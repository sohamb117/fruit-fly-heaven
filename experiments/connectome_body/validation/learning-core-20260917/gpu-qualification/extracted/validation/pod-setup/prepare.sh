#!/usr/bin/env bash
set -euo pipefail
cd /workspace/experiments/connectome_body
export UV_CACHE_DIR=/workspace/.uv-cache
export MUJOCO_GL=disable MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.mplconfig
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while kill -0 170 2>/dev/null; do sleep 2; done
uv run --no-sync python -c 'import torch, pyarrow, mujoco; print(torch.__version__)'
uv run --no-sync python -m connectome_body.cli bootstrap > validation/pod-setup/flybody.log 2>&1 &
fly_pid=$!
uv run --no-sync python -m connectome_body.adaptation.cli assets --download > validation/pod-setup/assets.log 2>&1 &
assets_pid=$!
while ! grep -q PUBLIC_DOWNLOADS_COMPLETE validation/pod-setup/public-download.log; do
  if ! kill -0 592 2>/dev/null; then cat validation/pod-setup/public-download.log; exit 1; fi
  sleep 2
done
uv run --no-sync python -m connectome_body.cli prepare banc > validation/pod-setup/banc.log 2>&1 &
banc_pid=$!
uv run --no-sync python -m connectome_body.cli prepare malecns > validation/pod-setup/malecns.log 2>&1 &
male_pid=$!
wait "$fly_pid"
wait "$assets_pid"
wait "$banc_pid"
wait "$male_pid"
uv run --no-sync python scripts/paper_preflight.py --device cuda --expected-gpu 'RTX 4090' --timeout 60 --output validation/pod-setup/cuda.json
uv run --no-sync python -m connectome_body.compatibility.learning_study plan --config configs/paper/learning_core.json --output runs/learning-core-cuda-v1
printf '%s\n' PREPARATION_COMPLETE
