#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Report captured observation evidence; never control the observer or mark reviews.

uv run scripts/summarize-fly-observation.py --partial
uv run scripts/summarize-fly-observation.py  # refuses incomplete final evidence
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import struct
import sys
from urllib.parse import quote


DEFAULT_DIRECTORY = Path("reports/observation-60min-20260913")
RADIUS_LIMIT = 64.0  # Exact observer predicate: hypot(scene x, scene z) <= 64.
CLOCK_EPSILON = 1e-9
CONTROLLER_LABELS = {"banc_direct": "BANC direct motor control", "reference_policy": "Published FlyBody policy", "unknown": "Unknown controller"}
COMPLETION_LABELS = {
    "observerResultComplete": "Observer explicitly reports the session complete",
    "atLeast200DistinctCaptures": "At least 200 distinct captures are recorded",
    "uniqueCaptureIndicesAndPaths": "Capture indices and original image paths are unique",
    "allOriginalPngFilesPresent": "Every original PNG file is present",
    "observerDurationAtLeast3600Seconds": "Observer reports at least 3,600 elapsed seconds",
    "observerTimestampsSpanAtLeast3600Seconds": "Observer start/end timestamps span at least 3,600 seconds",
    "captureEndpointAtLeast3600Seconds": "Captures extend to at least 3,600 seconds after session start",
    "everyCapturedFrameVisuallyReviewed": "Every captured frame has an explicit visual review",
    "observerCountMatchesCaptureLog": "Observer total matches the capture log",
    "inputIntegrity": "Input logs pass consistency checks",
}
STALL_LABELS = {"reference_completed_hold": "Completed reference trajectory held", "faulted_body_clock_stall": "Body clock stopped with a runtime error",
                "paused_body_clock_stall": "Paused body clock", "unpaused_body_clock_stall": "Body clock stopped while unpaused"}
TARGET = "Food localization → approach → landing → probing/feeding → takeoff/flight"
SCOPE = (
    "An iterative real-wall-clock observation spanning recorded versions, reloads, "
    "and clock stalls. It is not an uninterrupted hour of simulated behavior. "
    "Reference-policy flight is separate from BANC motor control. Motion, task, "
    "and feeding fields are reported telemetry, not independently validated behavior."
)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo else None
    except ValueError:
        return None


def clean_json(value):
    """Do not emit JavaScript NaN/Infinity as if they were measured numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def load_inputs(directory, partial):
    issues, warnings, hashes = [], [], {}

    def read_bytes(relative, required=False):
        path = directory / relative
        if not path.exists():
            if required:
                issues.append(f"Missing input: {relative}")
            return None
        raw = path.read_bytes()
        hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
        return raw

    def read_json(relative, default, required=False):
        raw = read_bytes(relative, required)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            issues.append(f"Invalid JSON in {relative}: {error}")
            return default

    def read_jsonl(relative, required=False):
        raw = read_bytes(relative, required)
        if raw is None:
            return []
        rows = []
        lines = raw.splitlines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
            except (ValueError, UnicodeDecodeError) as error:
                message = f"Invalid {relative} line {index}: {error}"
                # An active append can be observed between its last bytes. A
                # partial report may omit only that unfinished trailing row.
                if partial and index == len(lines) and not raw.endswith(b"\n"):
                    warnings.append(message + " (unfinished trailing row omitted)")
                else:
                    issues.append(message)
        return rows

    session = read_json("session.json", {}, True)
    result = read_json("result.json", {})
    changes = read_json("changes.json", [], True)
    frames = read_jsonl("frames.jsonl", True)
    reviews = read_jsonl("visual-reviews.jsonl", True)
    manifests = {}
    for path in sorted((directory / "source-manifests").glob("*.json")):
        manifests[path.stem] = read_json(path.relative_to(directory), {})
    if not isinstance(session, dict):
        issues.append("session.json must contain an object")
        session = {}
    if not isinstance(result, dict):
        issues.append("result.json must contain an object")
        result = {}
    if not isinstance(changes, list) or any(not isinstance(c, dict) for c in changes):
        issues.append("changes.json must contain an array of change objects")
        changes = []
    return session, result, changes, frames, reviews, manifests, hashes, issues, warnings


def controller_kind(state):
    # Explicit observer metadata takes precedence over a version's informal name.
    if state.get("reference") is True:
        return "reference_policy"
    if state.get("reference") is False and state.get("source") == "BANC motor neurons":
        return "banc_direct"
    return "unknown"


def frame_errors(state):
    errors = state.get("errors", [])
    if not isinstance(errors, list):
        errors = [errors]
    if state.get("error"):
        errors = [*errors, state["error"]]
    return list(dict.fromkeys(str(error).strip() for error in errors if error and str(error).strip()))


def extent(values):
    values = [v for v in values if number(v)]
    return {"min": min(values), "max": max(values)} if values else None


def summarize(directory: Path, partial: bool):
    directory = directory.resolve()
    session, result, changes, records, reviews, manifests, hashes, issues, warnings = load_inputs(directory, partial)
    started = timestamp(session.get("startedAt"))
    if started is None:
        issues.append("session.startedAt is missing or is not a timezone-aware timestamp")
    frames, by_index, seen_files = [], {}, set()
    duplicate_indices, duplicate_files, missing_images, invalid_images = [], [], [], []
    image_groups = defaultdict(list)
    for row_number, original in enumerate(records, 1):
        index = original.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
            issues.append(f"Capture row {row_number} has an invalid frame index")
            continue
        if index in by_index:
            duplicate_indices.append(index)
            continue
        state = original.get("state", {})
        if not isinstance(state, dict):
            issues.append(f"Frame {index} state is not an object")
            state = {}
        file = original.get("file")
        valid_path = isinstance(file, str) and bool(file)
        if valid_path:
            path = (directory / file).resolve()
            valid_path = path.is_relative_to(directory) and Path(file).parts[0] == "frames"
        if not valid_path:
            issues.append(f"Frame {index} does not name an original file inside frames/")
            file, path = None, None
        elif file in seen_files:
            duplicate_files.append(file)
        if file:
            seen_files.add(file)
        exists = bool(path and path.is_file())
        image_hash, image_size = None, None
        if not exists:
            missing_images.append(index)
        else:
            raw = path.read_bytes()
            if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
                invalid_images.append(index)
            else:
                image_size = list(struct.unpack(">II", raw[16:24]))
                image_hash = hashlib.sha256(raw).hexdigest()
                image_groups[image_hash].append(index)
        elapsed, captured_at = original.get("elapsedSeconds"), timestamp(original.get("at"))
        if not number(elapsed) or elapsed < 0 or captured_at is None:
            issues.append(f"Frame {index} has invalid wall-time evidence")
        elif started is not None and abs(captured_at - started - elapsed) > .05:
            issues.append(f"Frame {index} elapsedSeconds disagrees with its timestamp/session start")
        position = state.get("position")
        radius = math.hypot(position[0], position[2]) if isinstance(position, list) and len(position) >= 3 and number(position[0]) and number(position[2]) else None
        up = state.get("upZ")
        tilt = math.degrees(math.acos(max(-1, min(1, up)))) if number(up) else None
        root = state.get("physicalRoot")
        frame = {
            "index": index, "at": original.get("at"), "elapsedSeconds": elapsed,
            "latenessMs": original.get("latenessMs"), "version": original.get("version", "unknown"),
            "file": file, "imageExists": exists, "imageSha256": image_hash, "imageSize": image_size,
            "controllerKind": controller_kind(state), "controllerSource": state.get("source"),
            "bodyTimeSeconds": state.get("bodyTime"), "neuralTimeMs": state.get("neuralTimeMs"),
            "paused": state.get("paused"), "ready": state.get("ready"), "population": state.get("population"),
            "position": position, "physicalRoot": root, "velocity": state.get("velocity"),
            "radiusSceneUnits": radius, "insideRadiusRecomputed": radius <= RADIUS_LIMIT if radius is not None else None,
            "insideHabitatReported": state.get("insideHabitat"), "upZ": up, "tiltDegrees": tilt,
            "bodyFiniteReported": state.get("bodyFinite"),
            "physicalRootFiniteRecomputed": all(number(x) for x in root) if isinstance(root, list) and len(root) == 7 else None,
            "motionReported": state.get("motion"), "airborneReported": state.get("airborne"),
            "onFoodReported": state.get("onFood"), "wingPower": state.get("wingPower"),
            "contactsReported": state.get("contacts"), "mouthContactReported": state.get("mouthContact"),
            "proboscis": state.get("proboscis"), "pump": state.get("pump"), "internal": state.get("internal"),
            "taskReported": state.get("task"), "trialCompleteReported": state.get("complete"),
            "terminationReason": state.get("terminationReason"), "appliedForceReported": state.get("appliedForce"),
            "errors": frame_errors(state), "reviewIds": [],
        }
        if frame["insideHabitatReported"] is not None and frame["insideRadiusRecomputed"] is not None and frame["insideHabitatReported"] != frame["insideRadiusRecomputed"]:
            warnings.append(f"Frame {index}: reported habitat bound differs from recomputed radius <= 64")
        by_index[index] = frame
        frames.append(frame)
    # Preserve capture order; do not silently hide an out-of-order log by sorting.
    for previous, current in zip(frames, frames[1:]):
        if current["index"] <= previous["index"]:
            issues.append("Capture indices are not in strictly increasing log order")
        if number(current["elapsedSeconds"]) and number(previous["elapsedSeconds"]) and current["elapsedSeconds"] < previous["elapsedSeconds"]:
            issues.append("Capture wall times run backward")
    if duplicate_indices:
        issues.append(f"Duplicate frame indices: {duplicate_indices}")
    if duplicate_files:
        issues.append(f"Duplicate original screenshot paths: {duplicate_files}")

    normalized_reviews, unknown_review_frames = [], set()
    for review_number, review in enumerate(reviews, 1):
        indices = review.get("frames", [])
        note = review.get("observations", review.get("observation", ""))
        reviewed_at = review.get("reviewedAt", review.get("at"))
        reviewed_timestamp = timestamp(reviewed_at)
        usable = isinstance(indices, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in indices) and isinstance(note, str) and bool(note.strip()) and reviewed_timestamp is not None
        if not usable:
            issues.append(f"Review {review_number} needs explicit integer frames, an observation, and a timezone-aware review timestamp")
            continue
        normalized = {"id": review_number, "frames": sorted(set(indices)), "at": reviewed_at, "observation": note, "sheet": review.get("sheet")}
        normalized_reviews.append(normalized)
        for index in set(indices):
            if index not in by_index:
                unknown_review_frames.add(index)
                continue
            captured_at = timestamp(by_index[index]["at"])
            if captured_at is not None and reviewed_timestamp < captured_at:
                issues.append(f"Review {review_number} predates capture {index}; it does not count as visual review")
                continue
            by_index[index]["reviewIds"].append(review_number)
    if unknown_review_frames:
        issues.append(f"Visual reviews name uncaptured frames: {sorted(unknown_review_frames)}")
    reviewed = sorted(f["index"] for f in frames if f["reviewIds"])
    unreviewed = sorted(f["index"] for f in frames if not f["reviewIds"])

    change_by_version = defaultdict(list)
    source_records = []
    for change in changes:
        version = change.get("id", "unknown")
        change_by_version[version].append(change)
        recorded_hashes = change.get("sourceSha256", {})
        supplemental = manifests.get(version, {})
        if not isinstance(recorded_hashes, dict) or not isinstance(supplemental, dict):
            warnings.append(f"Invalid source hash object for {version}")
            recorded_hashes, supplemental = {}, {}
        conflicts = {name: [recorded_hashes[name], digest] for name, digest in supplemental.items() if name in recorded_hashes and recorded_hashes[name] != digest}
        if conflicts:
            warnings.append(f"Source hash conflicts for {version}: {', '.join(conflicts)}")
        source_records.append({**change, "supplementalManifest": f"source-manifests/{version}.json" if version in manifests else None,
                               "supplementalSha256": supplemental, "hashConflicts": conflicts})
    for frame in frames:
        if frame["version"] not in change_by_version:
            warnings.append(f"Frame {frame['index']} has no recorded load/version source manifest")
    if result.get("changes") is not None and result.get("changes") != changes:
        warnings.append("result.changes differs from changes.json; both originals remain linked")

    segments, resets, stalls, error_episodes = [], [], [], []
    for frame in frames:
        if not segments or (frame["version"], frame["controllerKind"]) != (segments[-1]["version"], segments[-1]["controllerKind"]):
            segment = {"id": len(segments) + 1, "version": frame["version"], "controllerKind": frame["controllerKind"], "frames": []}
            segments.append(segment)
        segment["frames"].append(frame["index"])
        frame["segment"] = segment["id"]
    for segment in segments:
        members = [by_index[i] for i in segment["frames"]]
        first, last = members[0], members[-1]
        segment.update({"firstFrame": first["index"], "lastFrame": last["index"], "capturedFrames": len(members),
                        "firstWallSeconds": first["elapsedSeconds"], "lastWallSeconds": last["elapsedSeconds"],
                        "wallSpanSeconds": last["elapsedSeconds"] - first["elapsedSeconds"] if number(last["elapsedSeconds"]) and number(first["elapsedSeconds"]) else None,
                        "bodyClockStartSeconds": first["bodyTimeSeconds"], "bodyClockEndSeconds": last["bodyTimeSeconds"],
                        "sampledBodyProgressSeconds": 0., "reviewedFrames": sum(bool(f["reviewIds"]) for f in members),
                        "outsideRadiusFrames": [f["index"] for f in members if f["insideRadiusRecomputed"] is False],
                        "invertedFrames": [f["index"] for f in members if number(f["upZ"]) and f["upZ"] < 0],
                        "errorFrames": [f["index"] for f in members if f["errors"]],
                        "pausedFrames": [f["index"] for f in members if f["paused"] is True],
                        "radiusSceneUnits": extent(f["radiusSceneUnits"] for f in members), "upZ": extent(f["upZ"] for f in members),
                        "motionLabelsReported": dict(Counter(f["motionReported"] or "unknown" for f in members)),
                        "taskPhasesReported": dict(Counter((f["taskReported"] or {}).get("phase", "unknown") for f in members)),
                        "mouthContactFramesReported": [f["index"] for f in members if f["mouthContactReported"] is True],
                        "cropTelemetry": extent((f["internal"] or {}).get("crop") for f in members),
                        "ingestedTelemetry": extent((f["internal"] or {}).get("ingested") for f in members)})
        for previous, current in zip(members, members[1:]):
            old, new = previous["bodyTimeSeconds"], current["bodyTimeSeconds"]
            if not number(old) or not number(new):
                continue
            delta = new - old
            if delta < -CLOCK_EPSILON:
                resets.append({"kind": "unrecorded_body_clock_regression", "beforeFrame": previous["index"], "afterFrame": current["index"],
                               "version": segment["version"], "bodyBeforeSeconds": old, "bodyAfterSeconds": new})
            elif delta > CLOCK_EPSILON:
                segment["sampledBodyProgressSeconds"] += delta
            elif number(current["elapsedSeconds"]) and number(previous["elapsedSeconds"]):
                kind = ("reference_completed_hold" if current["controllerKind"] == "reference_policy" and current["trialCompleteReported"] is True
                        else "faulted_body_clock_stall" if current["errors"] else "paused_body_clock_stall" if current["paused"] is True
                        else "unpaused_body_clock_stall")
                if stalls and stalls[-1]["segment"] == segment["id"] and stalls[-1]["endFrame"] == previous["index"] and stalls[-1]["kind"] == kind:
                    interval = stalls[-1]
                    interval.update(endFrame=current["index"], endWallSeconds=current["elapsedSeconds"])
                else:
                    interval = {"segment": segment["id"], "version": segment["version"], "kind": kind, "startFrame": previous["index"],
                                "endFrame": current["index"], "startWallSeconds": previous["elapsedSeconds"], "endWallSeconds": current["elapsedSeconds"], "bodyTimeSeconds": new}
                    stalls.append(interval)
                interval["sampledWallSpanSeconds"] = interval["endWallSeconds"] - interval["startWallSeconds"]
                interval_start = by_index[interval["startFrame"]]
                interval["neuralTimeStartMs"] = interval_start["neuralTimeMs"]
                interval["neuralTimeEndMs"] = current["neuralTimeMs"]
                interval["neuralClockAlsoUnchanged"] = abs(current["neuralTimeMs"] - interval_start["neuralTimeMs"]) <= CLOCK_EPSILON if number(current["neuralTimeMs"]) and number(interval_start["neuralTimeMs"]) else None
        # Adjacent repeated error screenshots are one observed episode, not
        # repeated independent crashes. A later clear/error transition is new.
        active_errors = {}
        for frame in members:
            active_errors = {message: episode for message, episode in active_errors.items() if message in frame["errors"]}
            for message in frame["errors"]:
                if message not in active_errors:
                    episode = {"segment": segment["id"], "version": segment["version"], "message": message, "firstFrame": frame["index"], "frames": []}
                    error_episodes.append(episode)
                    active_errors[message] = episode
                active_errors[message]["frames"].append(frame["index"])
                active_errors[message]["lastFrame"] = frame["index"]
    for change_number, change in enumerate(changes):
        at = timestamp(change.get("at"))
        before = [f for f in frames if at is not None and timestamp(f["at"]) is not None and timestamp(f["at"]) < at]
        after = [f for f in frames if f["version"] == change.get("id") and at is not None and timestamp(f["at"]) is not None and timestamp(f["at"]) >= at]
        resets.append({"kind": "initial_recorded_load" if change_number == 0 else "explicit_recorded_reload", "version": change.get("id"),
                       "at": change.get("at"), "url": change.get("url"), "elapsedSeconds": at - started if at is not None and started is not None else None,
                       "beforeFrame": before[-1]["index"] if before else None, "afterFrame": after[0]["index"] if after else None,
                       "bodyBeforeSeconds": before[-1]["bodyTimeSeconds"] if before else None, "bodyAfterSeconds": after[0]["bodyTimeSeconds"] if after else None})

    count = len(frames)
    observed_elapsed = max((f["elapsedSeconds"] for f in frames if number(f["elapsedSeconds"])), default=0.)
    captured_endpoint = max((timestamp(f["at"]) - started for f in frames if timestamp(f["at"]) is not None), default=0.) if started is not None else 0.
    result_start, result_end = timestamp(result.get("startedAt")), timestamp(result.get("endedAt"))
    result_timestamp_duration = result_end - result_start if result_start is not None and result_end is not None else None
    if result_start is not None and started is not None and abs(result_start - started) > .05:
        issues.append("Observer result starts a different session from session.json")
    if number(result_timestamp_duration) and number(result.get("actualDurationSeconds")) and abs(result_timestamp_duration - result["actualDurationSeconds"]) > 1:
        issues.append("Observer result duration disagrees with its start/end timestamps")
    if result_end is not None and any(timestamp(f["at"]) is not None and timestamp(f["at"]) > result_end for f in frames):
        issues.append("A captured frame postdates the observer result's end timestamp")
    conditions = {
        "observerResultComplete": result.get("complete") is True,
        "atLeast200DistinctCaptures": count >= 200,
        "uniqueCaptureIndicesAndPaths": not duplicate_indices and not duplicate_files,
        "allOriginalPngFilesPresent": count > 0 and not missing_images and not invalid_images,
        "observerDurationAtLeast3600Seconds": number(result.get("actualDurationSeconds")) and result["actualDurationSeconds"] >= 3600,
        "observerTimestampsSpanAtLeast3600Seconds": number(result_timestamp_duration) and result_timestamp_duration >= 3600,
        "captureEndpointAtLeast3600Seconds": observed_elapsed >= 3600 and captured_endpoint >= 3600,
        "everyCapturedFrameVisuallyReviewed": count > 0 and not unreviewed,
        "observerCountMatchesCaptureLog": result.get("capturedFrames") == count,
        "inputIntegrity": not issues,
    }
    failures = [name for name, passed in conditions.items() if not passed]
    complete = not partial and not failures
    return clean_json({
        "schemaVersion": 1, "generatedAt": datetime.now(timezone.utc).isoformat(), "mode": "partial" if partial else "final",
        "generatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "complete": complete, "completionEligible": not failures, "completionConditions": conditions, "completionFailures": failures,
        "scope": SCOPE, "target": TARGET, "behavioralSuccessAssessed": False,
        "behavioralConclusion": "This generator makes no claim of a successful autonomous food-seeking cycle. Read version-specific visual observations; recorded task labels and pre-fix feeding telemetry are not physiological validation.",
        "session": session, "observerResult": result or None, "capturedFrames": count, "captureRows": len(records),
        "reviewedUniqueFrames": len(reviewed), "reviewedFrameIndices": reviewed, "unreviewedFrameIndices": unreviewed,
        "observedElapsedSeconds": observed_elapsed, "captureTimestampEndpointSeconds": captured_endpoint,
        "firstCaptureElapsedSeconds": frames[0]["elapsedSeconds"] if frames else None,
        "firstToLastCaptureSpanSeconds": frames[-1]["elapsedSeconds"] - frames[0]["elapsedSeconds"] if frames and number(frames[-1]["elapsedSeconds"]) and number(frames[0]["elapsedSeconds"]) else None,
        "resultTimestampDurationSeconds": result_timestamp_duration,
        "controllerFrameCounts": dict(Counter(f["controllerKind"] for f in frames)),
        "originalScreenshotChecks": {"missingFrames": missing_images, "invalidPngFrames": invalid_images, "duplicateIndices": duplicate_indices,
                                     "duplicatePaths": duplicate_files, "distinctImageHashes": len(image_groups),
                                     "identicalImageGroups": [v for v in image_groups.values() if len(v) > 1],
                                     "note": "Count distinct original capture IDs/files, not distinct pixels. Repeated images during stalls remain captured observations."},
        "boundsDefinition": {"formula": "hypot(scene position[0], scene position[2]) <= 64", "radiusSceneUnits": RADIUS_LIMIT,
                             "limitation": "This is the recorded radial habitat predicate, not a full collision or altitude test. Native root and scene coordinates are separate."},
        "segments": segments, "resets": resets, "bodyClockStalls": stalls,
        "stallInterpretation": "Intervals join adjacent captures with unchanged body clocks. They are sampled evidence, not proof of continuous behavior between screenshots. They never count as active stable flight.",
        "observedErrorEpisodes": error_episodes, "browserErrors": result.get("errors", []),
        "sourceRecords": source_records, "sourceManifestFiles": sorted(f"source-manifests/{name}.json" for name in manifests),
        "sourceProvenanceLimit": "Hashes record loaded-version provenance; they do not archive all source code. External assays are not counted as observed frames or behaviors.",
        "inputSha256": hashes, "visualReviews": normalized_reviews, "frames": frames,
        "issues": list(dict.fromkeys(issues)), "warnings": list(dict.fromkeys(warnings)),
    })


def plot_timeline(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"banc_direct": "#16764a", "reference_policy": "#b16b05", "unknown": "#686868"}
    frames = {f["index"]: f for f in summary["frames"]}
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for segment in summary["segments"]:
        rows = [frames[i] for i in segment["frames"]]
        for ax, field in zip(axes, ["bodyTimeSeconds", "radiusSceneUnits", "upZ"]):
            points = [(f["elapsedSeconds"] / 60, f[field]) for f in rows if number(f["elapsedSeconds"]) and number(f[field])]
            if points:
                ax.plot(*zip(*points), color=colors[segment["controllerKind"]], marker=".", markersize=3, linewidth=1.2)
        first = rows[0]
        if number(first["elapsedSeconds"]) and number(first["bodyTimeSeconds"]):
            axes[0].annotate(f"V{segment['id']}", (first["elapsedSeconds"] / 60, first["bodyTimeSeconds"]), xytext=(2, 5), textcoords="offset points", fontsize=8)
    for interval in summary["bodyClockStalls"]:
        for ax in axes:
            ax.axvspan(interval["startWallSeconds"] / 60, interval["endWallSeconds"] / 60, color="#c5c5c5", alpha=.3)
    for reset in summary["resets"]:
        if reset["kind"] == "explicit_recorded_reload" and number(reset.get("elapsedSeconds")):
            for ax in axes:
                ax.axvline(reset["elapsedSeconds"] / 60, color="#999", linestyle=":", linewidth=.8)
    errored = [f for f in frames.values() if f["errors"] and number(f["elapsedSeconds"]) and number(f["bodyTimeSeconds"])]
    if errored:
        axes[0].scatter([f["elapsedSeconds"] / 60 for f in errored], [f["bodyTimeSeconds"] for f in errored], color="#b91c1c", marker="x", s=22, zorder=4)
    axes[1].axhline(RADIUS_LIMIT, color="#b91c1c", linestyle="--", linewidth=1)
    axes[2].axhline(0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[2].set_ylim(-1.06, 1.06)
    for ax, label in zip(axes, ["Body clock (s)", "Scene radial position", "Body up · world up"]):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Observed elapsed wall time (minutes)")
    legend = [Line2D([], [], color=color, label=label) for label, color in [("BANC direct", colors["banc_direct"]), ("Published reference policy", colors["reference_policy"])]]
    legend += [Line2D([], [], color="#aaa", linewidth=7, alpha=.4, label="Unchanged body clock"), Line2D([], [], color="#b91c1c", marker="x", linestyle="none", label="Error shown")]
    axes[0].legend(handles=legend, loc="upper left", fontsize=8)
    prefix = "COMPLETE OBSERVATION LOG" if summary["complete"] else "PARTIAL OBSERVATION LOG"
    fig.suptitle(f"{prefix}: {summary['capturedFrames']} captures, {summary['reviewedUniqueFrames']} visually reviewed\nReloads separate runs; telemetry is not behavioral validation", fontsize=12)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if number(value) else "unknown"


def escaped(value):
    return html.escape(str(value if value is not None else "unknown"), quote=True)


def local_link(file):
    return quote(str(file), safe="/-_.")


def render_html(summary):
    title = "Complete observation log" if summary["complete"] else "Partial observation log"
    parts = [f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · one fly</title><style>
:root{{color-scheme:light;--ink:#183229;--muted:#566b61}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f2;color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:24px}}h1{{font-size:2rem;margin:0 0 8px}}h2{{margin-top:36px}}p{{max-width:110ch}}a{{color:#145ea8}}.status{{background:#fff4ce;border-left:6px solid #b77b0b;padding:18px}}.complete{{background:#e4f6e9;border-color:#16764a}}
.stats{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}.stats span{{padding:10px 16px;background:white;border:1px solid #d6dfd7;border-radius:6px}}.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px;text-align:left;border:1px solid #d6dfd7;vertical-align:top}}th{{background:#e7eee7}}.table-wrap{{overflow:auto}}
.timeline{{width:100%;height:auto;background:white}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0;position:sticky;top:0;background:#f3f6f2;padding:12px 0;z-index:1}}select,input{{font:inherit;padding:8px;max-width:100%}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:20px}}
article{{background:white;border:1px solid #cfd9d0;border-radius:8px;overflow:hidden;scroll-margin-top:100px}}article>header,article>.detail{{padding:12px 16px}}article h3{{margin:0;font-size:1.1rem}}.shot{{display:block;width:100%;height:auto}}.bad{{color:#a42020}}.badge{{display:inline-block;font-size:.83rem;font-weight:650;padding:2px 7px;border-radius:4px;background:#e9efea;margin:3px 4px 3px 0}}.policy{{background:#ffedcf}}.unreviewed{{background:#fee4e4}}dl{{display:grid;grid-template-columns:130px 1fr;gap:4px 10px;font-size:.91rem}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}}details{{margin:8px 0}}.review{{border-left:3px solid #94b39b;padding:4px 10px;margin:8px 0;font-size:.9rem}}footer{{margin-top:35px;font-size:.9rem}}[hidden]{{display:none!important}}
</style><main><header class="status {'complete' if summary['complete'] else ''}"><h1>{title}: one fly</h1>
<p>{escaped(SCOPE)}</p><p><strong>Target:</strong> {escaped(TARGET)}. <strong>No automated claim of task success.</strong></p></header>
<div class="stats"><span><strong>{summary['capturedFrames']}</strong> original captures / 200 minimum</span><span><strong>{summary['reviewedUniqueFrames']}</strong> unique captures reviewed</span><span><strong>{fmt(summary['observedElapsedSeconds']/60,2)}</strong> recorded wall minutes / 60 minimum</span><span><strong>{len(summary['segments'])}</strong> recorded segments</span></div>
<p class="muted">Generated {escaped(summary['generatedAt'])}. Browser version {escaped(summary['session'].get('browser'))}. Scheduled end {escaped(summary['session'].get('scheduledEnd'))}. Counts derive from captured files, not the session's requested totals.</p>
<p><a href="summary.json">Machine-readable summary</a> · <a href="frames.jsonl">Original capture log</a> · <a href="visual-reviews.jsonl">Original visual reviews</a> · <a href="session.json">Session</a> · <a href="changes.json">Recorded reloads and source hashes</a>{' · <a href="result.json">Observer result</a>' if summary['observerResult'] else ''}</p>
<h2>Completion checks</h2><ul>"""]
    for name, passed in summary["completionConditions"].items():
        parts.append(f"<li class=\"{'bad' if not passed else ''}\">{'Pass' if passed else 'Pending / failed'} — {escaped(COMPLETION_LABELS.get(name,name))}</li>")
    parts.append("</ul>")
    if summary["unreviewedFrameIndices"]:
        parts.append(f"<p><strong>Unreviewed captures:</strong> {escaped(', '.join(map(str,summary['unreviewedFrameIndices'])))}</p>")
    for issue in summary["issues"] + summary["warnings"]:
        parts.append(f"<p class=\"bad\">{escaped(issue)}</p>")
    if summary.get("timeline"):
        parts.append('<h2>Recorded progress and physical state</h2><a href="timeline.png"><img class="timeline" src="timeline.png" alt="Body time, radial position and attitude over elapsed wall time, separated by controller and reload"></a>')
    parts.append('<h2>Runs and recorded reloads</h2><p>Body-clock progress is summed only between adjacent captures in each run. No duration is joined across a reload.</p><div class="table-wrap"><table><thead><tr><th>Run / frames</th><th>Controller / version</th><th>Wall interval</th><th>Body clock</th><th>Bounds / attitude / errors</th><th>Source evidence</th></tr></thead><tbody>')
    for segment in summary["segments"]:
        manifest = f"source-manifests/{segment['version']}.json"
        source = f'<a href="{local_link(manifest)}">Supplemental hashes</a>' if manifest in summary["sourceManifestFiles"] else "Reload hashes only"
        parts.append(f"<tr><td>V{segment['id']} · <a href=\"#frame-{segment['firstFrame']}\">{segment['firstFrame']}–{segment['lastFrame']}</a></td><td><strong>{escaped(CONTROLLER_LABELS[segment['controllerKind']])}</strong><br>{escaped(segment['version'])}</td><td>{fmt(segment['firstWallSeconds']/60 if number(segment['firstWallSeconds']) else None,2)}–{fmt(segment['lastWallSeconds']/60 if number(segment['lastWallSeconds']) else None,2)} min</td><td>{fmt(segment['bodyClockStartSeconds'])} → {fmt(segment['bodyClockEndSeconds'])} s<br>sampled gain {fmt(segment['sampledBodyProgressSeconds'])} s</td><td>{len(segment['outsideRadiusFrames'])} outside radius; {len(segment['invertedFrames'])} inverted; {len(segment['errorFrames'])} error captures</td><td>{source}</td></tr>")
    parts.append('</tbody></table></div><details><summary>Explicit load/reload records and clock regressions</summary><ul>')
    for reset in summary["resets"]:
        parts.append(f"<li><strong>{escaped(reset['kind'])}</strong> · {escaped(reset.get('at'))} · {escaped(reset.get('version'))} · captured frame {escaped(reset.get('beforeFrame'))} → {escaped(reset.get('afterFrame'))}; body {fmt(reset.get('bodyBeforeSeconds'))} → {fmt(reset.get('bodyAfterSeconds'))} s.</li>")
    parts.append('</ul></details><h2>Unchanged body clocks and errors</h2><p>Fresh screenshots of a stopped body are observations of a stall, not evidence of stable active behavior. Repeated error screenshots are grouped into one observed error episode.</p><ul>')
    for interval in summary["bodyClockStalls"]:
        parts.append(f"<li>Frames {interval['startFrame']}–{interval['endFrame']}: <strong>{escaped(STALL_LABELS[interval['kind']])}</strong>, {fmt(interval['sampledWallSpanSeconds'],1)} wall seconds at body {fmt(interval['bodyTimeSeconds'])} s; V{interval['segment']}.</li>")
    if not summary["bodyClockStalls"]:
        parts.append("<li>No unchanged adjacent body-clock samples in the captured log.</li>")
    parts.append("</ul>")
    for episode in summary["observedErrorEpisodes"]:
        parts.append(f"<details><summary>Error episode, frames {episode['firstFrame']}–{episode['lastFrame']}, V{episode['segment']}</summary><pre>{escaped(episode['message'])}</pre></details>")
    if summary["browserErrors"]:
        parts.append(f'<details><summary>Browser error events recorded by the observer (raw events, not a crash count)</summary><pre>{escaped(json.dumps(summary["browserErrors"],indent=2))}</pre></details>')
    parts.append('<h2>All original screenshots</h2><p>Images below link directly to the original captures. Visual-review status comes only from the existing review log; this generator does not perform or invent visual review. Numeric state was sampled immediately before its screenshot.</p><div class="filters"><select id="controller" aria-label="Filter controller"><option value="">All controllers</option><option value="banc_direct">BANC direct</option><option value="reference_policy">Reference policy</option><option value="unknown">Unknown</option></select><select id="version" aria-label="Filter version"><option value="">All versions</option>')
    for version in dict.fromkeys(frame["version"] for frame in summary["frames"]):
        parts.append(f'<option value="{escaped(version)}">{escaped(version)}</option>')
    parts.append('</select><select id="flag" aria-label="Filter status"><option value="">All captures</option><option value="unreviewed">Unreviewed</option><option value="error">Errors shown</option><option value="outside">Outside radius</option><option value="inverted">Inverted</option></select><span id="visible-count"></span></div><div class="gallery">')
    review_by_id = {r["id"]: r for r in summary["visualReviews"]}
    for frame in summary["frames"]:
        flags = [name for name, condition in [("unreviewed", not frame["reviewIds"]), ("error", frame["errors"]), ("outside", frame["insideRadiusRecomputed"] is False), ("inverted", number(frame["upZ"]) and frame["upZ"] < 0)] if condition]
        parts.append(f'<article id="frame-{frame["index"]}" data-controller="{escaped(frame["controllerKind"])}" data-version="{escaped(frame["version"])}" data-flags="{escaped(" ".join(flags))}"><header><h3>Frame {frame["index"]:03d} · V{frame["segment"]}</h3><span class="badge {"policy" if frame["controllerKind"]=="reference_policy" else ""}">{escaped(CONTROLLER_LABELS[frame["controllerKind"]])}</span><span class="badge {"unreviewed" if not frame["reviewIds"] else ""}">{"Visually reviewed" if frame["reviewIds"] else "Not yet visually reviewed"}</span><div>{escaped(frame["version"])}</div><div class="muted">Wall {fmt(frame["elapsedSeconds"],1)} s · body {fmt(frame["bodyTimeSeconds"])} s · {escaped(frame["at"])}</div></header>')
        if frame["imageExists"] and frame["file"]:
            parts.append(f'<a href="{local_link(frame["file"])}"><img class="shot" loading="lazy" src="{local_link(frame["file"])}" width="1440" height="1000" alt="Original captured screenshot, frame {frame["index"]}"></a>')
        else:
            parts.append('<p class="bad">Original screenshot is missing.</p>')
        details = [
            ("Scene position", frame["position"]), ("Radial bound", f"r={fmt(frame['radiusSceneUnits'])}; limit=64; inside={frame['insideRadiusRecomputed']}"),
            ("Body attitude", f"upZ={fmt(frame['upZ'])}; tilt={fmt(frame['tiltDegrees'],1)}°; native root finite={frame['physicalRootFiniteRecomputed']}"),
            ("Native root", frame["physicalRoot"]), ("Clock / pause", f"body={fmt(frame['bodyTimeSeconds'])} s; neural/UI counter={fmt(frame['neuralTimeMs'],1)} ms; paused={frame['paused']}"),
            ("Reported motion", f"{frame['motionReported']}; airborne={frame['airborneReported']}; onFood={frame['onFoodReported']}"),
            ("Mouth telemetry", f"contact={frame['mouthContactReported']}; proboscis={fmt(frame['proboscis'])}; pump={fmt(frame['pump'])}; crop={fmt((frame['internal'] or {}).get('crop'))}"),
            ("Wing / contact", f"wing={fmt(frame['wingPower'])}; contacts={frame['contactsReported']}; appliedForce={fmt(frame['appliedForceReported'])}"),
        ]
        if frame["trialCompleteReported"] is not None or frame["terminationReason"]:
            details.append(("Trial state", f"complete={frame['trialCompleteReported']}; reason={frame['terminationReason']}"))
        parts.append('<div class="detail"><dl>')
        for label, value in details:
            text = "[" + ", ".join(fmt(v,6) if number(v) else str(v) for v in value) + "]" if isinstance(value, list) else value
            parts.append(f"<dt>{escaped(label)}</dt><dd>{escaped(text)}</dd>")
        parts.append("</dl>")
        for error in frame["errors"]:
            parts.append(f'<details open><summary class="bad">Error shown</summary><pre>{escaped(error)}</pre></details>')
        for review_id in frame["reviewIds"]:
            review = review_by_id[review_id]
            parts.append(f'<div class="review"><strong>Recorded visual review · {escaped(review["at"])}</strong><br>{escaped(review["observation"])}</div>')
        parts.append('</div></article>')
    parts.append('</div><footer><p>Coordinate note: scene position is in UI scene coordinates; native root is MuJoCo x,y,z followed by quaternion w,x,y,z in model coordinates. The radial limit is the observer\'s recorded predicate, not a complete collision test. No controller source is inferred from the appearance of a fly.</p><p>Original screenshots, capture logs and visual reviews are never rewritten by this report. Input and screenshot SHA-256 values are recorded in summary.json. Other agents\' assays do not count toward captured or reviewed frames.</p></footer></main><script>const selects=[...document.querySelectorAll("select")],cards=[...document.querySelectorAll("article")];function filter(){const c=document.querySelector("#controller").value,v=document.querySelector("#version").value,f=document.querySelector("#flag").value;let n=0;for(const card of cards){card.hidden=!!((c&&card.dataset.controller!==c)||(v&&card.dataset.version!==v)||(f&&!card.dataset.flags.split(" ").includes(f)));if(!card.hidden)n++;}document.querySelector("#visible-count").textContent=n+" / "+cards.length+" shown";}selects.forEach(s=>s.addEventListener("change",filter));filter();</script></html>')
    return "\n".join(parts)


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--partial", action="store_true", help="Generate a clearly incomplete snapshot while capture/review continues")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional matplotlib timeline")
    args = parser.parse_args(argv)
    summary = summarize(args.directory, args.partial)
    if not args.partial and not summary["completionEligible"]:
        print(json.dumps({"status": "final_refused", "completionFailures": summary["completionFailures"], "capturedFrames": summary["capturedFrames"],
                          "reviewedUniqueFrames": summary["reviewedUniqueFrames"], "observedElapsedSeconds": summary["observedElapsedSeconds"], "issues": summary["issues"]}, indent=2), file=sys.stderr)
        return 2
    directory = args.directory.resolve()
    summary["timeline"] = None
    if not args.no_plot:
        plot_timeline(summary, directory / "timeline.png")
        summary["timeline"] = "timeline.png"
    atomic_write(directory / "gallery.html", render_html(summary))
    atomic_write(directory / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete_observation_log" if summary["complete"] else "partial_observation_log", "capturedFrames": summary["capturedFrames"],
                      "reviewedUniqueFrames": summary["reviewedUniqueFrames"], "observedElapsedSeconds": summary["observedElapsedSeconds"],
                      "completionFailures": summary["completionFailures"], "summary": str(directory / "summary.json"), "gallery": str(directory / "gallery.html")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
