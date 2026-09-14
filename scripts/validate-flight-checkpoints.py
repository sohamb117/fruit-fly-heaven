"""Prepare pinned held-out jobs, or report their results; never execute a simulation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def report_path(value):
    result = (ROOT / value).resolve()
    require(result.is_relative_to(ROOT / "reports"), "Inputs and outputs must stay inside reports")
    return result


def read_json(file):
    file = report_path(file)
    data = file.read_bytes()
    value = json.loads(data, parse_constant=lambda constant: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    return value, {"file": str(file.relative_to(ROOT)), "sha256": sha(data)}


def write_json(file, value):
    with file.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_protocol(file):
    protocol, provenance = read_json(file)
    require(protocol["schemaVersion"] == 1 and protocol["kind"] == "flight-checkpoint-validation-protocol", "Wrong protocol")
    if "amends" in protocol:
        original, original_source = read_json(protocol["amends"]["file"])
        require(original_source == protocol["amends"], "Original predeclaration changed")
        require({key: value for key, value in protocol.items() if key not in ("amends", "amendment", "selection", "jobOrder")} == {key: value for key, value in original.items() if key != "jobOrder"}, "Amendment changes the original center comparison")
    else:
        require("amendment" not in protocol, "An amendment must identify the preserved original protocol")
    require(protocol.get("selection") == {"source": "exactly32acceptedTrainingJobs", "rank": "highest_score", "tieBreak": "lexical_job_id", "role": "secondary_selected_single_training_trial"}, "Unrecognized secondary selection rule")
    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) == 3 and all(type(seed) is int and 0 <= seed <= 0xffffffff for seed in seeds) and len(set(seeds)) == 3, "Require three distinct uint32 held-out seeds")
    records, sources = {}, {}
    for key in ("declaration", "bundle", "backend"):
        records[key], sources[key] = read_json(protocol["sources"][key]["file"])
        require(sources[key] == protocol["sources"][key], f"Pinned {key} bytes changed")
    declared, bundle, backend = (records[key] for key in ("declaration", "bundle", "backend"))
    config = json.loads(bundle["configText"])
    require(sha(bundle["configText"].encode()) == bundle["configHash"], "Bundle config hash mismatch")
    for record in (declared, backend, bundle):
        require(record["configHash"] == bundle["configHash"] and record["modelFingerprint"] == config["modelFingerprint"], "Protocol configuration mismatch")
    for asset, content in bundle["assets"].items():
        require(sha(content.encode()) == config["assets"][asset], f"Bundled asset hash mismatch: {asset}")
    require(backend["backend"] == declared["backend"] == "dawn-metal", "Explicit Dawn Metal required")
    require(protocol["criteriaVersion"] == 3, "Expected declared criteria 3")
    require(protocol["baselineGeneration"] == 0 and protocol["candidateGeneration"] == declared["validation"]["afterGeneration"] == 4, "Generation declaration mismatch")
    require(protocol["acceptedJobsRequired"] == declared["maximumInitialJobs"] == 32, "Training budget changed")
    require(protocol["seeds"] == declared["validation"]["seeds"], "Held-out seeds differ from the pinned declaration")
    require(protocol["stage"] == declared["validation"]["stage"] == config["stage"] == "landing", "Stage mismatch")
    require(protocol["durationSeconds"] == declared["validation"]["durationSeconds"] == config["durationSeconds"] == 8, "Full landing horizon required")
    require(any(stage["id"] == "landing" and stage["durationSeconds"] == 8 for stage in config["stages"]), "Missing full landing stage")
    require(len(config["parameters"]) == 27, "Expected 27-parameter contract")
    return protocol, provenance, records, config


def prepare(protocol_file, output):
    protocol, protocol_source, records, config = load_protocol(protocol_file)
    database = report_path(protocol["database"])
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM meta"))
        generations = [dict(row) for row in connection.execute("SELECT * FROM generations ORDER BY generation")]
        jobs = [dict(row) for row in connection.execute("SELECT job_id,generation,pair_id,sign,seed,parameters,parameters_hash,state,score,result_hash FROM jobs ORDER BY generation,pair_id,sign DESC")]
    finally:
        connection.close()
    require(metadata["configHash"] == records["bundle"]["configHash"] and metadata["modelFingerprint"] == config["modelFingerprint"], "Database configuration pins differ")
    require(json.loads(metadata["config"]) == config, "Database configuration content differs")
    require([row["generation"] for row in generations] == list(range(5)), "Wait for exactly generation 4; do not select an earlier or later checkpoint")
    completed = [row for row in jobs if row["state"] == "completed"]
    require(len(completed) == protocol["acceptedJobsRequired"], "Wait for exactly 32 accepted results")
    require(not any(row["state"] == "leased" for row in jobs), "Stop contributions at the generation boundary before freezing validation")
    count = 2 * config["optimizer"]["populationPairs"]
    require(count == 8, "Unexpected generation size")
    for number, generation in enumerate(generations):
        batch = [row for row in jobs if row["generation"] == number]
        require(len(batch) == count, "Incomplete generation job inventory")
        expected = "completed" if number < 4 else "pending"
        require(all(row["state"] == expected for row in batch), "Training must stop after the fourth complete generation")
        require(generation["status"] == ("unverified" if number < 4 else "evaluating"), "Unexpected generation status")
        pairs = defaultdict(list)
        for job in batch:
            pairs[job["pair_id"]].append(job)
            if number < 4:
                require(finite(job["score"]) and -10 <= job["score"] <= 10 and job["result_hash"], "Accepted job lacks finite scored result")
        require(len(pairs) == count // 2 and all(sorted(job["sign"] for job in pair) == [-1, 1] and pair[0]["seed"] == pair[1]["seed"] for pair in pairs.values()), "Antithetic pair inventory mismatch")
    require(not set(protocol["seeds"]) & {row["seed"] for row in completed}, "Held-out seed already used for training")
    selected = min(completed, key=lambda row: (-row["score"], row["job_id"]))
    centers = {f"g{row['generation']}": row["center"] for row in generations if row["generation"] in (0, 4)}
    centers["training-best"] = selected["parameters"]
    for raw in centers.values():
        vector = json.loads(raw)
        require(len(vector) == 27 and all(finite(value) and parameter["min"] <= value <= parameter["max"] for value, parameter in zip(vector, config["parameters"])), "Center violates parameter contract")
    require(json.loads(centers["g0"]) == records["declaration"]["initialParameters"] == [parameter["initial"] for parameter in config["parameters"]], "Generation 0 is not the declared initial center")
    snapshot = {"schemaVersion": 1, "database": protocol["database"], "readMode": "SQLite mode=ro, query_only, one BEGIN snapshot including WAL", "capturedAt": datetime.now(timezone.utc).isoformat(),
                "configHash": metadata["configHash"], "modelFingerprint": metadata["modelFingerprint"], "acceptedResults": len(completed),
                "generations": generations, "jobs": jobs, "storedParameterJson": centers, "selectedTrainingJob": selected}
    snapshot["snapshotSha256"] = sha(canonical({key: value for key, value in snapshot.items() if key != "capturedAt"}).encode())
    plan = {"schemaVersion": 1, "backend": "dawn-metal", "configHash": metadata["configHash"], "modelFingerprint": metadata["modelFingerprint"],
            "nativeWebGPU": records["backend"]["nativeWebGPU"], "scope": protocol["interpretation"],
            "validation": {"protocol": protocol_source, "snapshotSha256": snapshot["snapshotSha256"], "criteriaVersion": 3, "baselineGeneration": 0, "candidateGeneration": 4,
                           "heads": [{"id": "g0", "kind": "generation-center", "generation": 0}, {"id": "g4", "kind": "generation-center", "generation": 4},
                                     {"id": "training-best", "kind": "selected-single-training-trial", "generation": selected["generation"], "jobId": selected["job_id"], "trainingScore": selected["score"]}],
                           "centerEncoding": "Each parameters array is the exact stored SQLite center or candidate JSON, without manual rounding."}, "jobs": []}
    for seed in protocol["seeds"]:
        for head in centers:
            plan["jobs"].append({"name": f"{head}-seed-{seed}", "seed": seed, "stage": "landing", "durationSeconds": 8,
                                 "parameters": f"__EXACT_STORED_CENTER_{head}__", "capturePhysicsDigest": True})
    plan_text = json.dumps(plan, indent=2, allow_nan=False)
    for head, raw in centers.items():
        plan_text = plan_text.replace(json.dumps(f"__EXACT_STORED_CENTER_{head}__"), raw)
    require(all(job["parameters"] == json.loads(centers[list(centers)[i % 3]]) for i, job in enumerate(json.loads(plan_text)["jobs"])), "Exact center substitution failed")
    output = report_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "database-snapshot.json", snapshot)
    (output / "plan.json").write_text(plan_text + "\n")
    write_json(output / "preparation.json", {"protocol": protocol_source, "planSha256": sha((plan_text + "\n").encode()), "scriptSha256": sha(Path(__file__).read_bytes()), "snapshotSha256": snapshot["snapshotSha256"], "jobs": 9, "simulationExecuted": False})
    return {"plan": str((output / "plan.json").relative_to(ROOT)), "jobs": 9, "acceptedResults": 32, "selectedTrainingJob": selected["job_id"], "simulationExecuted": False}


def checked_evaluation(record, job, index, plan, config, plan_source):
    require(record.get("complete") is True and not record.get("error"), "Incomplete/error evaluation")
    require(record["assignment"] == job and record["index"] == index, "Evaluation assignment/order differs from plan")
    require(record["planSha256"] == plan_source["sha256"], "Evaluation plan bytes changed")
    expected_sources = {key: value for key, value in config["assets"].items() if not key.startswith("/body-model/")}
    require(record["sourceHashes"] == expected_sources, "Captured runtime source pins differ")
    require(record["backendLabel"] == "dawn-metal", "Wrong backend label")
    native = record["nativeWebGPU"]
    require(native["backend"] == "dawn-metal" and native["requestedBackend"] == "metal" and native["fallbackAllowed"] is False and native["adapter"]["isFallbackAdapter"] is False, "Wrong/fallback native adapter")
    require(all(native[key] == value for key, value in plan["nativeWebGPU"].items()), "Native backend pins differ")
    value = record["evaluation"]
    require(not value.get("cancelled") and value["backend"] == "webgpu" and value["bodyBackend"] == "mujoco-wasm", "Cancelled or incorrect execution backend")
    for key in ("configHash", "modelFingerprint"):
        require(record[key] == value[key] == plan[key], f"Evaluation {key} differs")
    for key in ("seed", "stage", "parameters"):
        require(value[key] == job[key], f"Evaluation {key} differs")
    require(value["dtMs"] == config["dtMs"] and value["bodyBlockMs"] == config["bodyBlockMs"], "Time-step contract differs")
    metrics = value["metrics"]
    require(not metrics.get("error") and metrics["actualNeuralBackend"] == "webgpu" and metrics["criteria"]["version"] == 3, "Wrong objective or evaluation error")
    for key in ("return", "simSeconds"):
        require(finite(value[key]), f"Nonfinite {key}")
    require(-10 <= value["return"] <= 10 and 0 < value["simSeconds"] <= 8 + 1e-7, "Invalid score or elapsed time")
    require(isinstance(value["steps"], int) and not isinstance(value["steps"], bool) and value["steps"] > 0 and abs(value["simSeconds"] - value["steps"] * config["bodyBlockMs"] / 1000) < 1e-7, "Incomplete body-step count")
    full = abs(value["simSeconds"] - 8) < 1e-7
    physical_failure = not value["success"] and value["terminated"] and value["reason"] in ("outside_habitat", "excessive_rotation", "overturned")
    require(full or physical_failure, "Evaluation must finish the full horizon or a known physical failure")
    require(not value["success"] or full, "Early success cannot qualify")
    require(value["reason"] not in ("simulation_error", "invalid_state", "external_force"), "Invalid physical result")
    require(metrics["success"] == value["success"] and metrics["reason"] == value["reason"] and metrics["return"] == value["return"], "Metric/result disagreement")
    require(metrics["duration"] == 8 and abs(metrics["elapsed"] - value["simSeconds"]) < 1e-7, "Metric horizon differs")
    for key in ("bestFlightSeconds", "landingSeconds", "wallSeconds", "setupWallSeconds", "executionWallSeconds"):
        require(finite(metrics[key]) and metrics[key] >= 0, f"Invalid {key}")
    require(metrics["bestFlightSeconds"] <= value["simSeconds"] + 1e-7 and metrics["landingSeconds"] <= value["simSeconds"] + 1e-7, "Impossible duration metric")
    require(isinstance(metrics["hasTakenOff"], bool), "Invalid takeoff flag")
    digest = record["physicsDigest"]
    require(digest["rows"] == value["steps"] + 1 and isinstance(digest["valuesPerRow"], int) and digest["valuesPerRow"] > 0 and len(digest["sha256"]) == 64, "Incomplete physics digest")
    return {"name": job["name"], "head": plan["validation"]["heads"][index % 3]["id"], "seed": job["seed"],
            **{key: value[key] for key in ("return", "success", "reason", "simSeconds", "terminated", "truncated", "steps")},
            **{key: metrics[key] for key in ("hasTakenOff", "takeoffTime", "bestFlightSeconds", "landingSeconds", "landingTime", "wallSeconds", "setupWallSeconds", "executionWallSeconds")},
            "diagnostics": metrics["diagnostics"], "finalObservation": metrics["finalObservation"], "physicsDigest": digest}


def report(plan_file, evaluations, output):
    plan, plan_source = read_json(plan_file)
    protocol, protocol_source, records, config = load_protocol(plan["validation"]["protocol"]["file"])
    require(protocol_source == plan["validation"]["protocol"], "Validation protocol bytes changed")
    preparation, _ = read_json(report_path(plan_file).parent / "preparation.json")
    snapshot, snapshot_source = read_json(report_path(plan_file).parent / "database-snapshot.json")
    require(preparation["planSha256"] == plan_source["sha256"], "Prepared plan changed")
    digest = sha(canonical({key: value for key, value in snapshot.items() if key not in ("capturedAt", "snapshotSha256")}).encode())
    require(digest == snapshot["snapshotSha256"] == preparation["snapshotSha256"] == plan["validation"]["snapshotSha256"], "Frozen database snapshot changed")
    require(plan["configHash"] == records["bundle"]["configHash"] and plan["modelFingerprint"] == config["modelFingerprint"] and plan["nativeWebGPU"] == records["backend"]["nativeWebGPU"], "Plan differs from original bundle/backend")
    require(len(plan["jobs"]) == 9, "Expected all nine jobs")
    completed = [row for row in snapshot["jobs"] if row["state"] == "completed"]
    require(len(completed) == 32, "Frozen snapshot must contain exactly 32 accepted results")
    selected = min(completed, key=lambda row: (-row["score"], row["job_id"]))
    require(snapshot["selectedTrainingJob"] == selected and snapshot["storedParameterJson"]["training-best"] == selected["parameters"], "Secondary head did not follow the predeclared training-only selection")
    heads = [{"id": "g0", "kind": "generation-center", "generation": 0}, {"id": "g4", "kind": "generation-center", "generation": 4},
             {"id": "training-best", "kind": "selected-single-training-trial", "generation": selected["generation"], "jobId": selected["job_id"], "trainingScore": selected["score"]}]
    require(plan["validation"]["heads"] == heads, "Checkpoint heads differ from frozen selection")
    for index, job in enumerate(plan["jobs"]):
        head, seed = heads[index % 3]["id"], protocol["seeds"][index // 3]
        require(job == {"name": f"{head}-seed-{seed}", "seed": seed, "stage": "landing", "durationSeconds": 8, "parameters": json.loads(snapshot["storedParameterJson"][head]), "capturePhysicsDigest": True}, "Plan jobs differ from frozen centers/declaration")
    found, sources, runs = {}, [], []
    for file in sorted(report_path(evaluations).glob("*.json")):
        value, source = read_json(file)
        if value.get("kind") == "native-checkpoint-evaluation":
            runs.append((value, source))
        elif value.get("kind") == "native-held-out-evaluation":
            name = value["assignment"]["name"]
            require(name not in found, "Duplicate evaluation; do not cherry-pick retries")
            found[name] = value
            sources.append(source)
    require(len(runs) == 1, "Require exactly one recorded evaluator run")
    run, run_source = runs[0]
    require(not run.get("error") and not run.get("stopped") and run.get("finishedAt"), "Evaluator run incomplete or errored")
    require(run["planSha256"] == plan_source["sha256"] and run["plan"] == plan and len(run["jobs"]) == 9 and all(job["complete"] and not job["error"] and not job["cancelled"] for job in run["jobs"]), "Run manifest differs or incomplete")
    require(set(found) == {job["name"] for job in plan["jobs"]}, "Need exactly all nine planned evaluations")
    rows = []
    for index, job in enumerate(plan["jobs"]):
        value = found[job["name"]]
        require(value["runId"] == run["runId"], "Mixed evaluation runs")
        rows.append(checked_evaluation(value, job, index, plan, config, plan_source))
    pairs = []
    comparisons = ("primary_center", "secondary_selected_training_candidate")
    for index in range(0, 9, 3):
        before = rows[index]
        for offset, comparison in enumerate(comparisons, 1):
            after = rows[index + offset]
            a, b = [found[row["name"]] for row in (before, after)]
            require(a["nativeWebGPU"] == b["nativeWebGPU"], "Paired native adapter differs")
            for key in ("initialCondition", "initialObservation", "criteria"):
                require(a["evaluation"]["metrics"][key] == b["evaluation"]["metrics"][key], f"Paired {key} differs")
            pairs.append({"comparison": comparison, "seed": before["seed"], "baseline": before, "candidate": after,
                          "difference": {key: after[key] - before[key] for key in ("return", "bestFlightSeconds", "simSeconds", "landingSeconds")},
                          "takeoffChanged": before["hasTakenOff"] != after["hasTakenOff"], "terminationChanged": before["reason"] != after["reason"]})
    summary = {}
    for comparison in comparisons:
        grouped = [pair for pair in pairs if pair["comparison"] == comparison]
        summary[comparison] = {key: {"meanDifference": statistics.mean(pair["difference"][key] for pair in grouped), "medianDifference": statistics.median(pair["difference"][key] for pair in grouped),
                                    "increasedSeeds": sum(pair["difference"][key] > 1e-9 for pair in grouped), "decreasedSeeds": sum(pair["difference"][key] < -1e-9 for pair in grouped),
                                    "unchangedSeeds": sum(abs(pair["difference"][key]) <= 1e-9 for pair in grouped)} for key in ("return", "bestFlightSeconds", "simSeconds", "landingSeconds")}
    summaries = {head["id"]: {"successes": sum(row["success"] for row in rows if row["head"] == head["id"]), "takeoffs": sum(row["hasTakenOff"] for row in rows if row["head"] == head["id"]),
                              "reasons": dict(Counter(row["reason"] for row in rows if row["head"] == head["id"]))} for head in heads}
    before, after = (json.loads(snapshot["storedParameterJson"][head]) for head in ("g0", "g4"))
    delta = [b-a for a, b in zip(before, after)]
    result = {"schemaVersion": 1, "kind": "held-out-flight-checkpoint-comparison", "criteriaVersion": 3, "allNineEvaluationGatesPassed": True,
              "provenance": {"plan": plan_source, "protocol": protocol_source, "snapshot": snapshot_source, "run": run_source, "evaluations": sources, "reportScriptSha256": sha(Path(__file__).read_bytes())},
              "configHash": plan["configHash"], "modelFingerprint": plan["modelFingerprint"], "heads": heads, "pairs": pairs, "pairedSummary": summary, "headSummary": summaries,
              "parameterMovementOnly": {"changedCount": sum(value != 0 for value in delta), "l2": math.hypot(*delta), "deltaByParameter": dict(zip((p["name"] for p in config["parameters"]), delta)), "interpretation": "Optimizer movement is not evidence of behavioral improvement."},
              "limits": [protocol["interpretation"], "The primary comparison is generation 4 versus generation 0. The secondary head is a single candidate selected only on training score; it is not the trained center.", "Longer time until a crash alone is not controlled flight. Use qualified continuous airtime and completed milestones alongside scores.", "No aggregate score improvement alone establishes maintained flight or landing.", "Physics digests cover initialization and completed 2 ms body blocks, not intervening native integration steps.", "No model simulation, coordinator write, lease or training upload was performed by this helper."]}
    output = report_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    text = "All nine held-out evaluations pass the pinned configuration, source, backend, assignment, horizon/physical-failure and paired initial-state checks. The primary comparison is generation 4 versus the exact generation-0 center. A pre-held-out amendment adds the single accepted candidate with highest training score (ties by lexical job ID) as a separate secondary head. Objective criteria 3 and the three seeds stay fixed.\n\n"
    text += f"The selected training candidate is {selected['job_id']}, with training score {selected['score']}. It is not the generation-4 center.\n\n"
    text += "| Seed | Candidate | Qualified flight g0 → candidate (s) | Takeoff | End time (s) | Score | Termination |\n|---|---|---:|---|---:|---:|---|\n"
    for pair in pairs:
        a, b = pair["baseline"], pair["candidate"]
        text += f"| {pair['seed']} | {b['head']} | {a['bestFlightSeconds']:.3f} → {b['bestFlightSeconds']:.3f} | {a['hasTakenOff']} → {b['hasTakenOff']} | {a['simSeconds']:.3f} → {b['simSeconds']:.3f} | {a['return']:.4f} → {b['return']:.4f} | {a['reason']} → {b['reason']} |\n"
    for comparison, values_by_key in summary.items():
        text += "\n" + comparison + ": " + "; ".join(f"{key}: mean paired difference {values['meanDifference']:+.6f}, {values['increasedSeeds']} increased / {values['decreasedSeeds']} decreased / {values['unchangedSeeds']} unchanged" for key, values in values_by_key.items()) + ".\n"
    text += "\n" + "; ".join(f"{head}: successes {values['successes']}/3, takeoffs {values['takeoffs']}/3" for head, values in summaries.items()) + ".\n\n"
    text += "The exact parameter center changed in " + str(result["parameterMovementOnly"]["changedCount"]) + " coordinates. This is reported separately from behavior and is not a learning claim. Three held-out seeds provide a limited paired check; they do not establish generalization or biological accuracy. A later crash without increased controlled flight is not successful flight.\n\n"
    text += "Full raw differences, criteria-3 diagnostics, source hashes and native digests are in [comparison.json](comparison.json). No simulation or training update was run by this report.\n"
    (output / "README.md").write_text(text)
    return {"report": str((output / "comparison.json").relative_to(ROOT)), "pairedSummary": summary, "headSummary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("protocol")
    prep.add_argument("output")
    compare = commands.add_parser("report")
    compare.add_argument("plan")
    compare.add_argument("evaluations")
    compare.add_argument("output")
    args = parser.parse_args()
    result = prepare(args.protocol, args.output) if args.command == "prepare" else report(args.plan, args.evaluations, args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
