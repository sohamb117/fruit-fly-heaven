#!/usr/bin/env bash
# Run only on the user's allocated GPU. No provisioning or billing actions.
set -euo pipefail
cd /workspace/experiments/connectome_body
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
export MUJOCO_GL=disable MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.mplconfig
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p validation/pod-setup
# Preserve the official template; create an isolated, lockfile-pinned environment.
uv sync --frozen --no-dev --python 3.12 > validation/pod-setup/install.log 2>&1
uv run --no-sync python scripts/paper_preflight.py --device cuda --expected-gpu 'RTX 4090' --timeout 60 --output validation/pod-setup/cuda.json
uv run --no-sync python -m connectome_body.compatibility.learning_study plan --config configs/paper/learning_core.json --output "${CB_PLAN_DIR:-runs/learning-core-cuda-v2}"
# Training is launched from an always-on supervising host using
# scripts/qualification_ssh_job.py. It arms a separate provider-stop watchdog
# before SSH, downloads a verified checkpoint archive, and stops even on errors.
# Prepare and upload data/graphs/.prepared on CPU before paying for GPU time.
