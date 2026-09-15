# Sequential training and eye views on flytrain

The sequential sensorimotor trainer is deployed at **https://flytrain.morisoba.moe/train.html** in Google Cloud project `flyheaven`. Cloud Run revision `fly-training-r0b7a72087eed` received 100% of traffic after the tagged revision passed configuration, checkpoint and coordinator preflight.

## Release identity

- Configuration: `284e053d1a7b14e6a6fbe8d1ac08302350f90f55391c478feae61addef3bb446`
- Model: `7ec004c7f6f7e8021ccb329a2e85fe618505800dd169b0f177048ce7aa2e8b53`
- Image: `us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:324f99174bd6edab893415c19d7aad9a716a9c41a576c3bac74499bb3a527388`
- Cloud Build: `7b2ad0fd-5cff-4b06-a6cf-474c43577d96`
- Build context: `dist/cloudrun/sequence-eyes-20260915`; build ID `a9050b10eaf98530`
- Frozen experiment: `reports/sequential-training-20260915/v4/sequence.bundle.json`
- Contributor archive: `dist/training-sequence-eyes-client/fruit-fly-training-client-284e053d1a7b.zip`; SHA256 `81e75ff29e709d1937d4088153162c6bf9a1c1f8960505a8320a9206e3b165bc`

The image serves one browser WASM worker per contributor. Firestore owns assignments, results, phase progress and checkpoints under `training_sequences/banc888-sensorimotor-wasm-20260915`. Active jobs renew their 180-second leases every 60 seconds. The previous `training_runs/banc888-motor-decoder-wasm-20260914` namespace remains intact with 104 accepted results and a generation-10 checkpoint; no old scores were imported.

The nine phases train leg, antenna, haltere, wing-strain and visual feedback; fit the existing anatomical decoder; then train recovery, takeoff and landing. The vector contains 24 sensory coordinates and 672 decoder coefficients. Candidate improvements and task completion are separate acceptance decisions. Full details and limitations are in [the sequence contract](../../docs/training-sequence.md).

## Eye views

The **Eyes** tab sits beside Fly and Brain in the existing preview panel. It displays separate left/right 256 × 128 RGB images copied from the sensory renderer's last consumed frame. The motion pathway uses luminance derived from those images. Capture time is displayed independently of the current body clock.

Eye observation performs no new camera rendering or neural/physics steps. Owned image copies are sent at most twice per wall second, only while Eyes is selected and visible. They never enter coordinator uploads, checkpoint persistence or the worker progress watchdog. The existing scientific model/configuration identity is unchanged by this presentation feature. See [observation details](../../docs/training-brain-view.md#eyes).

## Verification

- 30 Cloud Run tests passed, including a 95,644-byte decoder-fit upload, idempotent retry and oversized-request handling.
- 14 eye/brain observer tests and 23 eye UI/client tests passed.
- Contributor packaging verified all 110 files over HTTP and checked archive integrity. The managed package exposes 108 public files.
- All 15 public HTTPS transport checks passed after deployment; eight deployed UI/config/observer assets matched the release manifest byte for byte.
- Local Safari displayed both actual eye images with capture time advancing from 0.0 to 0.4 seconds. Pause/Resume retained and labeled the last image correctly; Stop released its local assignment.
- Public Safari loaded all nine phases, 696 current parameters, and both eye images. Retinal capture time advanced from 0.1 to 3.8 seconds with changing images; the existing 3D and Brain views remained available.
- The first public Safari trial completed its full five-second scored horizon and was durably accepted in Firestore at 2026-09-15 07:28 UTC. Its score was 0.0588 and task success was false. All 696 applied values matched the assignment, the pinned neural WASM and MuJoCo body provenance matched, and only the five active leg sensory parameters differed from the initial configuration. The browser displayed one completed/uploaded trial and automatically started trial two.
- The generation-zero checkpoint still matched the initial configuration: no candidate comparison or checkpoint improvement had finished. The trainer was left running at 60% intensity with the user's High preview quality retained.

Machine-readable evidence is in `deployment.json`, `cloud-build.json`, `public-assets.json`, `http-verification.json`, `browser-observed-assignment.json`, `browser-result-audit.json` and `release-validation.json`. Generated bundles, raw evidence and checkpoint snapshots are local artifacts. The result audit verifies accepted browser-reported execution; it is not remote execution attestation.

This release establishes the deployed training and observation paths. It does not establish that sensory parameters match measured physiology or that the fly has learned the complete behavioral target.
