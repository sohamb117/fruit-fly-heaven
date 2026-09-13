"""Deploy a built image with bounded Cloud Run capacity and exact browser origins."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit


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


def deploy(image, run_id, project="flyheaven", region="us-central1", service="fly-training",
           origin="https://flytrain.morisoba.moe"):
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
    # Keep working aliases permitted throughout an image update, including the
    # interval before gcloud returns the new revision's service URLs.
    origins = service_origins(origin, existing[0] if existing else {})
    env = {"GOOGLE_CLOUD_PROJECT": project, "TRAINING_RUN_ID": run_id,
           "ALLOWED_ORIGINS": ",".join(origins)}
    with tempfile.TemporaryDirectory(prefix="fly-cloudrun-") as temporary:
        env_file = Path(temporary) / "environment.json"
        env_file.write_text(json.dumps(env))
        command(["run", "deploy", service, *base, f"--image={image}",
                 f"--service-account=fly-training-runtime@{project}.iam.gserviceaccount.com",
                 "--allow-unauthenticated", "--ingress=all", "--cpu=1", "--memory=512Mi",
                 "--concurrency=8", "--min=0", "--max=2", "--min-instances=0", "--max-instances=2",
                 "--cpu-throttling", "--cpu-boost", "--no-use-http2", "--timeout=300", "--port=8080",
                 "--startup-probe=httpGet.path=/healthz,httpGet.port=8080,initialDelaySeconds=0,periodSeconds=5,timeoutSeconds=3,failureThreshold=24",
                 f"--env-vars-file={env_file}"])
        status = json.loads(command(["run", "services", "describe", service, *base, "--format=json"]))
        url = status["status"]["url"]
        published_origins = service_origins(origin, status)
        if published_origins != origins:
            env["ALLOWED_ORIGINS"] = ",".join(published_origins)
            env_file.write_text(json.dumps(env))
            command(["run", "services", "update", service, *base, f"--env-vars-file={env_file}"])
    return {"project": project, "region": region, "service": service, "url": url,
            "publicOrigin": origin, "runId": run_id, "image": image}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="flyheaven")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--service", default="fly-training")
    parser.add_argument("--origin", default="https://flytrain.morisoba.moe")
    args = parser.parse_args()
    print(json.dumps(deploy(args.image, args.run_id, args.project, args.region, args.service, args.origin), indent=2))
