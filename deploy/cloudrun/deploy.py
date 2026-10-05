"""Preflight and promote an image on an existing bounded Cloud Run service."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


def command(args):
    return subprocess.run(["gcloud", *args], check=True, text=True, stdout=subprocess.PIPE).stdout


def service_origins(origin, record):
    published = json.loads(record.get("metadata", {}).get("annotations", {}).get("run.googleapis.com/urls", "[]"))
    primary = record.get("status", {}).get("url")
    values = list(dict.fromkeys([origin, *([primary] if primary else []), *published]))
    for value in values:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Cloud Run returned an invalid service origin")
    return values


def fetch_bytes(url):
    request = Request(url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=30) as response:
        if response.geturl() != url:
            raise ValueError("Preflight must stay on the tagged revision")
        return response.read()


def preflight(origin, lease_timeout_seconds=180):
    """Check the exact tagged site's config and coordinator without creating jobs."""
    raw_config = fetch_bytes(origin + "/training/config.json")
    config = json.loads(raw_config)
    config_hash = hashlib.sha256(raw_config).hexdigest()
    model = config.get("modelFingerprint", "")
    if not isinstance(model, str) or not re.fullmatch(r"[a-f0-9]{64}", model):
        raise ValueError("Preflight config has an invalid model identity")
    status = json.loads(fetch_bytes(origin + "/api/training/status?compact=1"))
    downloaded = json.loads(fetch_bytes(origin + "/api/training/checkpoint"))
    expected_lease_seconds = min(config.get("contribution", {}).get("leaseSeconds", 1800), lease_timeout_seconds)
    if type(status.get("leaseSeconds")) not in (int, float) or status["leaseSeconds"] != expected_lease_seconds:
        raise ValueError("Preflight coordinator lease timeout differs from deployment policy")
    parameters = config.get("parameters", [])
    if not parameters or not all(isinstance(p, dict) for p in parameters):
        raise ValueError("Preflight config has no parameter schema")
    names = [p.get("name") for p in parameters]
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Preflight config has invalid parameter names")
    phases = config.get("trainingSequence", {}).get("phases")
    allowed_stages = {p["stage"] for p in phases} if phases else {config.get("stage")}
    for value in (status, status.get("checkpoint"), downloaded):
        if not isinstance(value, dict) or value.get("configHash") != config_hash or value.get("modelFingerprint") != model:
            raise ValueError("Preflight website and coordinator identities differ")
        if type(value.get("generation")) is not int or value["generation"] < 0 or value.get("stage") not in allowed_stages:
            raise ValueError("Preflight checkpoint progress is invalid")
    for value in (status["checkpoint"], downloaded):
        vector = value.get("parameters")
        if (value.get("schemaVersion") != 1 or value.get("algorithm") != config.get("algorithm")
                or value.get("parameterNames") != names or not isinstance(vector, list) or len(vector) != len(names)):
            raise ValueError("Preflight checkpoint schema differs from the website")
        for number, parameter in zip(vector, parameters):
            if (type(number) not in (int, float) or not math.isfinite(number)
                    or not parameter["min"] <= number <= parameter["max"]):
                raise ValueError("Preflight checkpoint parameters are invalid")
        if phases:
            completed = value.get("completedPhases")
            if (not isinstance(completed, list) or len(completed) > len(phases)
                    or any(not isinstance(entry, dict) or entry.get("phaseId") != phases[i]["id"] for i, entry in enumerate(completed))
                    or value["stage"] != phases[min(len(completed), len(phases)-1)]["stage"]):
                raise ValueError("Preflight sequence checkpoint progress is invalid")
    if phases:
        sequence = status.get("sequence", {})
        index = len(status["checkpoint"]["completedPhases"])
        if (type(sequence.get("phaseIndex")) is not int or sequence["phaseIndex"] != index
                or sequence.get("phaseId") != phases[min(index, len(phases)-1)]["id"]
                or status["stage"] != status["checkpoint"]["stage"]):
            raise ValueError("Preflight sequence phase is inconsistent")
    if status["generation"] != status["checkpoint"]["generation"] or downloaded["generation"] < status["generation"]:
        raise ValueError("Preflight checkpoint generation is inconsistent")
    if (downloaded["generation"] == status["generation"]
            and (downloaded["parameters"] != status["checkpoint"]["parameters"]
                 or downloaded["stage"] != status["checkpoint"]["stage"]
                 or downloaded.get("completedPhases") != status["checkpoint"].get("completedPhases"))):
        raise ValueError("Preflight checkpoint changed without a generation update")
    return {"configHash": config_hash, "modelFingerprint": model,
            "generation": downloaded["generation"], "parameterCount": len(names), "leaseSeconds": status["leaseSeconds"]}


def tagged_origin(record, tag, revision):
    targets = [target for target in record.get("status", {}).get("traffic", []) if target.get("tag") == tag]
    if len(targets) != 1 or targets[0].get("revisionName") != revision:
        raise ValueError("Cloud Run did not attach the preflight tag to the exact deployed revision")
    url = targets[0].get("url", "")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".run.app")
            or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port):
        raise ValueError("Cloud Run returned an invalid tagged revision URL")
    return url


def deploy(image, run_id, project="flyheaven", region="us-central1", service="fly-training",
           origin="https://flytrain.morisoba.moe", lease_timeout_seconds=180):
    if type(lease_timeout_seconds) is not int or not 120 <= lease_timeout_seconds <= 86400:
        raise ValueError("Lease timeout must be an integer between 120 and 86400 seconds")
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", run_id):
        raise ValueError("Invalid run identity")
    if not image.startswith(f"{region}-docker.pkg.dev/{project}/") or not re.search(r"@sha256:[a-f0-9]{64}$", image):
        raise ValueError("Use an immutable Artifact Registry image digest from the selected project")
    parsed = urlsplit(origin)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Public origin must be HTTPS without a trailing slash or path")
    base = [f"--project={project}", f"--region={region}", "--quiet"]
    existing = json.loads(command(["run", "services", "list", *base,
                                   f"--filter=metadata.name={service}", "--format=json"]))
    if len(existing) != 1:
        raise ValueError("Guarded deployment requires one existing Cloud Run service; bootstrap the service separately")
    # Keep working aliases permitted throughout an image update, including the
    # interval before gcloud returns the new revision's service URLs.
    origins = service_origins(origin, existing[0])
    env = {"GOOGLE_CLOUD_PROJECT": project, "TRAINING_RUN_ID": run_id,
           "ALLOWED_ORIGINS": ",".join(origins), "TRAINING_LEASE_TIMEOUT_SECONDS": str(lease_timeout_seconds)}
    tag = "verify-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="fly-cloudrun-") as temporary:
        env_file = Path(temporary) / "environment.json"

        def stage_revision():
            # A caller-supplied unique suffix identifies this operation even if
            # another deployment changes latestCreatedRevisionName meanwhile.
            suffix = "r" + uuid.uuid4().hex[:12]
            revision = service + "-" + suffix
            env_file.write_text(json.dumps(env))
            command(["run", "deploy", service, *base, f"--image={image}",
                 "--no-traffic", f"--tag={tag}", f"--revision-suffix={suffix}",
                 f"--service-account=fly-training-runtime@{project}.iam.gserviceaccount.com",
                 "--allow-unauthenticated", "--ingress=all", "--cpu=1", "--memory=512Mi",
                 "--concurrency=8", "--min=0", "--max=2", "--min-instances=0", "--max-instances=2",
                 "--cpu-throttling", "--cpu-boost", "--no-use-http2", "--timeout=300", "--port=8080",
                 "--startup-probe=httpGet.path=/healthz,httpGet.port=8080,initialDelaySeconds=0,periodSeconds=5,timeoutSeconds=3,failureThreshold=24",
                 f"--env-vars-file={env_file}"])
            record = json.loads(command(["run", "services", "describe", service, *base, "--format=json"]))
            return revision, record, tagged_origin(record, tag, revision)

        try:
            revision, status, preview_url = stage_revision()
            url = status["status"]["url"]
            published_origins = service_origins(origin, status)
            if published_origins != origins:
                env["ALLOWED_ORIGINS"] = ",".join(published_origins)
                revision, status, preview_url = stage_revision()
            verified = preflight(preview_url, lease_timeout_seconds)
        except Exception as error:
            # Removing our unique tag never changes the existing traffic split.
            try:
                command(["run", "services", "update-traffic", service, *base, f"--remove-tags={tag}"])
            except Exception as cleanup_error:
                error.add_note(f"Preflight tag cleanup failed: {cleanup_error}")
            raise
        command(["run", "services", "update-traffic", service, *base,
                 f"--to-revisions={revision}=100", f"--remove-tags={tag}", "--format=json"])
        # update-traffic may serialize a list of traffic targets rather than a
        # Service. Verify the actual service after the completed mutation.
        promoted = json.loads(command(["run", "services", "describe", service, *base, "--format=json"]))
        traffic = promoted.get("status", {}).get("traffic", [])
        assigned = [target for target in traffic if target.get("percent", 0) > 0]
        if (len(assigned) != 1 or assigned[0].get("revisionName") != revision
                or assigned[0].get("percent") != 100 or assigned[0].get("latestRevision")
                or any(target.get("tag") == tag for target in traffic)):
            raise RuntimeError("Cloud Run did not confirm 100% traffic on the verified revision")
    return {"project": project, "region": region, "service": service, "url": url,
            "publicOrigin": origin, "runId": run_id, "image": image, "revision": revision,
            "trafficPercent": 100, **verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="flyheaven")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--service", default="fly-training")
    parser.add_argument("--origin", default="https://flytrain.morisoba.moe")
    parser.add_argument("--lease-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(deploy(args.image, args.run_id, args.project, args.region, args.service, args.origin,
                            args.lease_timeout_seconds), indent=2))
