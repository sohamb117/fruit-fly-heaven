# Checkpoint-preserving CUDA short test

The user-provided RTX PRO 4000 Blackwell pod `9bmscgp5qzbtbo` completed the bounded BANC engineering test and was stopped after about 27 minutes, with persistent storage retained. CUDA parity, all benchmark cases and interrupted/resumed training passed. The final checkpoint contains 925 of 1,000 requested updates; the native student achieved 0/3 hover successes. All 37 retrieved artifacts match their remote checksums. See [the completed report](runs/brief-pro4000-short-20260915/REPORT.md). The image's original PyTorch `2.9.1+cu130`, Python `3.12.3`, and CUDA 13.0 were retained. No replacement GPU was provisioned.

Artifacts are under `runs/brief-pro4000-short-20260915/`. See [the measured benchmark](runs/brief-pro4000-short-20260915/BENCHMARK.md) and the mirrored records in `retrieved/short-test/`. BANC has 175,401 neurons and 13,542,180 directed edges; the nominal 5k adapter has 4,912 trainable parameters. At batch 16 / BPTT 32, the initial benchmark measured 3,630.6 forward+backward recurrent transitions/second with 1.101 GiB peak PyTorch allocation. A transition is one whole-graph recurrent substep for one batch member. These timings exclude MuJoCo and optimizer updates.

## Scope and safeguards

This is the authorized 30-minute engineering test, not a completed scientific MVP cell. It includes native teacher preflight, sparse GPU benchmarking, offline imitation, checkpoint resumption, and a short native student rollout. Training uses the original immutable teacher data, 1,000 requested updates, batch 4, BPTT 32, burn-in 64, validation every 100 updates, and a checkpoint every 25 updates. The independent local MPS campaign is unchanged.

The pod started at 2026-09-15 19:41:57.859 UTC. The harness ended training at its 20:06:30 UTC cutoff and completed the rollout/audit. The isolated batching probe then tested batches 16/32/64/128 at BPTT 32 with no competing GPU process. Batch 64 reached 7,638.7 transitions/s at 3.375 GiB; batch 128 provided no further throughput gain. The GPU was stopped and verified `EXITED` around 20:09:07 UTC. The later 20:11:27.859 UTC stop guard was canceled, so it cannot affect a future restart.

Artifacts were copied from persistent storage to this computer approximately every 50 seconds. Each downloaded `latest.pt` was loaded and checked for finite weights, optimizer/RNG state, and manifest identity agreement. Final verification matched all 37 files against stable remote SHA-256 hashes before shutdown; the local backup loop was then canceled. For future runs, stop the GPU while retaining its persistent disk and never terminate the only checkpoint copy.

## Environment and data

The image is `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`. The project is `/workspace/connectome-body`; checkpoints are in `runs/short-test/experiment/stage-00` beneath it. Dependencies are installed at `/opt/connectome-venv` with system site packages to retain the GPU stack. The environment is rebuildable; persistent storage holds the data, checkpoints, and installed-version records.

The original archive is `runs/brief-4090-short-20260915/input.tar.gz`, SHA-256 `7fb55c693c3737b038664b2cb3b16b0c36515e01ddde97bf6791a94c8330b311`. All 342 input files were verified before applying the recorded deployment overlay. Do not install the archive's original Torch 2.8 lock into this environment: it would replace PyTorch 2.9.1. `deployment-manifest.json` and `retrieved/deployment/installed-requirements.txt` record the separate software profile.

BANC, the controller, dynamics, training data, and optimization rules are unchanged. CUDA/Torch 2.9.1 and earlier MPS/CPU/Torch 2.8 throughput ratios compare complete execution stacks, not hardware alone.

## Body portability boundary

Linux and macOS generated MJCF XML differ in 17 position/quaternion attributes with maximum absolute difference 1.11e-16. The engineering equivalence check requires matching non-XML identities, identical XML structure and other attributes, and only finite position/quaternion differences within 1e-14. Both native fingerprints are retained; neither identity is overwritten.

The native teacher passed both engineering preflight episodes. This equivalence validation applies only to the short test. The full scientific DAgger pipeline still enforces strict body identity; promoting cross-platform runs requires a separately versioned qualification/identity decision. No DAgger or full held-out GPU evaluation is represented by this offline test.

## Resuming

`latest.pt` contains actor weights, optimizer state, framework/sampling RNG state, update/experience counters, learning curves, and best-so-far weights. Source, data, graph, and configuration identities guard resumption. `best.pt` is an evaluation artifact; use `latest.pt` to resume.

After restoring the same project path and recorded environment on an explicitly authorized pod:

```sh
cd /workspace/connectome-body
```

If `/opt/connectome-venv` was lost on stop/restart, rebuild it on the same recorded image before resuming:

```sh
uv venv --python "$(command -v python)" --system-site-packages /opt/connectome-venv
uv pip install --python /opt/connectome-venv/bin/python --link-mode copy \
  -r /workspace/deployment/science-requirements.in
uv pip check --python /opt/connectome-venv/bin/python
```

Then resume the saved offline run:

```sh
MUJOCO_GL=disable MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python /opt/connectome-venv/bin/python \
  -m connectome_body.adaptation.cli offline \
  --config configs/brief/runpod-short.json \
  --output runs/short-test/experiment/stage-00 --resume
```

Do not restart the complete supervisor into the existing output directory: benchmark reports are immutable. Resume the offline CLI directly without changing its recorded source/config.

## Earlier attempts

The preceding RTX 4090 `k6wzxe5lhk2i41` failed native CUDA initialization with error 999 independently of project code. NVIDIA reported `GPU Recovery Action: Reboot`; a user container restart did not clear it. That pod was stopped with persistent storage retained and diagnostics copied/checksum-verified; its stop guard was canceled. See [the report](runs/brief-4090-torch24-20260915/REPORT.md).

The earlier RTX 3080 Ti attempt also failed CUDA initialization; see [its report](runs/brief-3080ti-short-20260915/REPORT.md). Neither produced CUDA training results. The working PRO 4000 run supersedes the earlier infrastructure-readiness-only status.
