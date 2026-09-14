"""Read-only behavioral summary of an operator-owned training database."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean


def parameter_update(generation, successor, rows, expected_jobs, names, acceptance=None):
    """Report actual stored-center movement, separately from trial rewards."""
    complete = len(rows) == expected_jobs
    result = {"fromGeneration": generation["generation"],
              "toGeneration": successor["generation"] if successor else None,
              "status": "incomplete" if not complete else "completed_without_successor",
              "completedJobs": len(rows), "expectedJobs": expected_jobs,
              "changedParameterCount": None, "deltaByParameter": None,
              "l2": None, "maxAbs": None, "equalScoresWithinEveryPair": None}
    if not complete or successor is None:
        return result
    before, after = json.loads(generation["center"]), json.loads(successor["center"])
    if any(len(vector) != len(names) or any(not math.isfinite(value) for value in vector)
           for vector in (before, after)):
        raise ValueError("Generation center does not match its parameter contract")
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pairId"]][row["sign"]] = row["score"]
    if len(pairs) * 2 != expected_jobs or any(set(pair) != {-1, 1} for pair in pairs.values()):
        raise ValueError("Completed generation lacks complete antithetic pairs")
    equal_scores = all(pair[1] == pair[-1] for pair in pairs.values())
    deltas = [right-left for left, right in zip(before, after)]
    changed = sum(delta != 0 for delta in deltas)
    result.update(status="updated" if changed else "completed_equal_scores_no_update" if equal_scores else "completed_no_update",
                  changedParameterCount=changed, deltaByParameter=dict(zip(names, deltas)),
                  l2=math.hypot(*deltas), maxAbs=max(map(abs, deltas), default=0),
                  equalScoresWithinEveryPair=equal_scores)
    if acceptance is not None:
        decision = acceptance.get("decision")
        if decision not in ("accepted", "rejected", "identical_parameters"):
            raise ValueError("Completed guarded generation lacks an acceptance decision")
        if decision in ("rejected", "identical_parameters") and changed:
            raise ValueError("Guarded checkpoint changed despite rejection or no-op")
        if decision == "accepted" and not changed:
            raise ValueError("Accepted guarded checkpoint did not change")
        result["acceptanceDecision"] = decision
        if not changed:
            result["status"] = "completed_rejected_proposal" if decision == "rejected" else "completed_identical_proposal"
    return result


def summarize(database):
    database = Path(database)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # Read metadata, generations, and jobs from one snapshot while the
        # active coordinator may atomically finish a generation in WAL mode.
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM meta"))
        config = json.loads(metadata["config"])
        generations = [dict(row) for row in connection.execute("SELECT * FROM generations ORDER BY generation")]
        jobs = connection.execute("SELECT job_id,generation,pair_id,sign,seed,parameters,state,score,result FROM jobs ORDER BY generation,pair_id,sign DESC").fetchall()
    finally:
        connection.close()
    grouped = defaultdict(list)
    trials = []
    for row in jobs:
        if row["state"] != "completed":
            continue
        payload = json.loads(row["result"])
        metrics = payload.get("metrics", {})
        diagnostics = metrics.get("diagnostics", {})
        trial = {"jobId": row["job_id"], "generation": row["generation"], "pairId": row["pair_id"],
                 "sign": row["sign"], "seed": row["seed"], "score": row["score"],
                 "parameters": json.loads(row["parameters"]), "success": metrics.get("success", False),
                 "reason": metrics.get("reason"), "simSeconds": metrics.get("simSeconds"),
                 "hasTakenOff": metrics.get("hasTakenOff"), "bestFlightSeconds": metrics.get("bestFlightSeconds"),
                 "landingSeconds": metrics.get("landingSeconds"), "landingTime": metrics.get("landingTime"),
                 "departures": diagnostics.get("departureCount"),
                 "poweredAirborneSeconds": diagnostics.get("poweredAirborneSeconds"),
                 "maximumAngularSpeed": diagnostics.get("maximumAngularSpeed"),
                 "wallSeconds": metrics.get("wallSeconds"),
                 "setupWallSeconds": metrics.get("setupWallSeconds"),
                 "executionWallSeconds": metrics.get("executionWallSeconds")}
        if config["schemaVersion"] == 2:
            trial["purpose"] = "checkpoint_comparison" if row["pair_id"].startswith(f"g{row['generation']}-a") else "search"
        grouped[row["generation"]].append(trial)
        trials.append(trial)
    names = [p["name"] for p in config["parameters"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate parameter names in database configuration")
    search_jobs = 2 * config["optimizer"]["populationPairs"]
    guarded = config["schemaVersion"] == 2
    by_generation = {generation["generation"]: generation for generation in generations}
    summaries = []
    for generation in generations:
        rows = grouped[generation["generation"]]
        acceptance = generation.get("acceptance")
        acceptance = json.loads(acceptance) if isinstance(acceptance, str) else acceptance
        expected_jobs = search_jobs + (6 if guarded else 0)
        if guarded and acceptance is not None:
            comparison_count = len(acceptance["comparisonJobIds"])
            if comparison_count != 6 and not (comparison_count == 0 and acceptance["decision"] == "identical_parameters"):
                raise ValueError("Unexpected guarded comparison inventory")
            expected_jobs = search_jobs + comparison_count
        summaries.append({"generation": generation["generation"], "status": generation["status"],
                          "center": json.loads(generation["center"]), "completed": len(rows),
                          "parameterUpdate": parameter_update(generation, by_generation.get(generation["generation"]+1), rows, expected_jobs, names, acceptance),
                          "meanScore": mean(row["score"] for row in rows) if rows else None,
                          "bestScore": max((row["score"] for row in rows), default=None),
                          "takeoffs": sum(row["hasTakenOff"] is True for row in rows),
                          "successes": sum(row["success"] is True for row in rows),
                          "bestFlightSeconds": max((row["bestFlightSeconds"] or 0 for row in rows), default=0),
                          "maximumPoweredAirborneSeconds": max((row["poweredAirborneSeconds"] or 0 for row in rows), default=0),
                          "reasons": dict(Counter(row["reason"] for row in rows))})
        if guarded:
            summaries[-1].update(acceptance=acceptance, completedByPurpose=dict(Counter(row["purpose"] for row in rows)))
    updates = [generation["parameterUpdate"] for generation in summaries]
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "database": str(database),
            "scope": "Stored parameter changes and training observations. Neither an optimizer update nor score improvement establishes held-out learning or flight.",
            "parameterDeltaUnits": "Stored optimizer coordinates (log multipliers for this contract); changed means an exact nonzero stored-coordinate difference.",
            "configHash": metadata["configHash"], "modelFingerprint": metadata["modelFingerprint"],
            "stage": config["stage"], "parameterNames": names, "parameterCount": len(names),
            "acceptedResults": len(trials), "jobs": dict(Counter(row["state"] for row in jobs)),
            "parameterUpdateSummary": dict(Counter(update["status"] for update in updates)),
            "behavior": {"takeoffs": sum(row["hasTakenOff"] is True for row in trials),
                         "successes": sum(row["success"] is True for row in trials),
                         "bestFlightSeconds": max((row["bestFlightSeconds"] or 0 for row in trials), default=0),
                         "maximumPoweredAirborneSeconds": max((row["poweredAirborneSeconds"] or 0 for row in trials), default=0)},
            "generations": summaries, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--last-generations", type=int, help="Limit console output; an output file still contains full history")
    args = parser.parse_args()
    if args.last_generations is not None and args.last_generations < 1:
        parser.error("--last-generations must be positive")
    report = summarize(args.database)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    console = {key: report[key] for key in ("timestamp", "parameterCount", "acceptedResults", "jobs", "parameterUpdateSummary", "behavior", "generations")}
    if args.last_generations:
        console["generations"] = console["generations"][-args.last_generations:]
    print(json.dumps(console, allow_nan=False))


if __name__ == "__main__":
    main()
