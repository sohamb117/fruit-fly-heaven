#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qualification-isolated-20260918/experiments/connectome_body
export UV_CACHE_DIR=/workspace/.uv-cache
export MUJOCO_GL=disable MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.mplconfig
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CONNECTOME_PREPARATION_REQUIRE_CACHE=1
run=runs/learning-qualification-cuda-v2
mkdir -p "$run/pre-isolation-attempt"
for name in started-at finished-at launcher-exit-code launcher.log launcher.pid qualification.log progress.json; do
 if test -f "$run/$name"; then mv "$run/$name" "$run/pre-isolation-attempt/$name"; fi
done
date -u +%FT%TZ > "$run/started-at"
code=0
timeout --signal=TERM --kill-after=15s 810s uv run --no-sync python -u scripts/qualify_learning_core.py --plan runs/learning-core-cuda-v2/plan.json --output "$run" --max-seconds 780 --case-max-seconds 180 --resume > "$run/qualification.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$run/launcher-exit-code"
date -u +%FT%TZ > "$run/finished-at"
tar -I 'gzip -1' -cf "$run-isolated.tar.gz" "$run" runs/learning-core-cuda-v2/plan.json validation/pod-setup-20260918
sha256sum "$run-isolated.tar.gz" > "$run-isolated.tar.gz.sha256"
printf 'ARCHIVE_READY\n'
exit "$code"
