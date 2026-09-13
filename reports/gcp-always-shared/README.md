# Shared training and checkpoint download update

Target: https://35.209.223.157/train.html, 2026-09-13.

The contributor page always joins its shared coordinator. Metadata connects automatically; neural/body compute waits for Start. There is no sharing checkbox, local search choice, or disconnect control. Public HTTPS uses its own origin; the downloaded localhost client uses the GCP endpoint. Failed connections do not fall back to unshared work. Internal scientific test APIs remain separate from this contributor interface.

**Download shared checkpoint** performs a fresh GET, validates model/config identity and parameters, then preserves the exact server JSON metadata. It never exports a stale local vector. The direct endpoint https://35.209.223.157/api/training/checkpoint sends an attachment filename based on its current generation and `Cache-Control: no-store`.

The database remains `/var/lib/fly-training/coordinator.sqlite3` on the retained persistent disk of `fly-training-1`, project `monereko-20260809`, zone `us-central1-a`. `generations.center` stores parameter vectors and `jobs` retains contributions. The small public JSON references the frozen model through fingerprints; the full history stays private. [Download and backup instructions](../../docs/training.md#download-checkpoints-and-full-history).

Completed results awaiting upload retain their original lease when Stop or a transient error occurs, so restarting can retry submission. Incomplete work is released. The normal 30-minute expiry still applies.

Local validation passed: 44 JavaScript tests, 27 coordinator tests, seven browser origins, failed coordinator before/after connection with zero worker creation, exact fresh checkpoint downloads, and 1440/390/320 px layouts. UI review is recorded under `reports/training-shared-ui/` and `reports/training-ui-idle/`.

Deployed artifact: `dist/gcp/fly-training-gcp-37b4afa356ec8af8.tar.gz`, 116,823,036 bytes, SHA256 `d0b5628ef6e5a17648c3a9d5063a1745adf3d7573aa2cac4b8a4ee892872e8d7`. Contains 139 verified payload files and no database bootstrap. A stalled large SFTP upload was replaced by a 54,669-byte delta sent through SCP; the VM reconstructed the full release from the known initial archive and verified all 139 file hashes before installation. Delta SHA256: `3954b4d339e3230a0922637e9c3fc406a4387b8b757385761b11fde1d59c950a`.

The installer preserved generation 1 and all nine existing accepted results. Pre-update server snapshot: `/var/lib/fly-training/before-always-shared-20260913.sqlite3`.

Contributor ZIP SHA256: `6dea72f31803ae5d24e24c8e117c838850480ed328935dfd19677bab6ec78f72`. The canonical model/config identities are unchanged.

Live verification passed on the deployed release: auto-connect with zero idle compute, coordinator outage blocking Start without local fallback, exact fresh checkpoint download including original metadata, one actual WebGPU/native FlyBody contribution, and release of the next incomplete lease. Accepted results increased **9 to 10**, with generation 1 retained. Desktop and mobile screenshots were visually inspected; no page, console or network errors occurred.

The episode produced 13 native frames and ended after 0.166 simulated seconds with excessive rotation and reward -5.03. This establishes working shared training infrastructure and downloads; posture remains unvalidated. Raw browser evidence and screenshots are generated local artifacts in this directory and are excluded from the source commit.
