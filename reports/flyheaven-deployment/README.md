# Managed hosting migration

Work performed on 2026-09-13 for project `flyheaven` (`753887928769`). Cloud Run serves the training website and coordinator; visitors execute the BANC/FlyBody simulation. Firestore `(default)` in `us-central1` provides durable history. The target domain is `flytrain.morisoba.moe`.

## Reviewed release

- Config SHA256: `1bad7d5c80d5129dcbe4f94b66fc725ec343053fb50bf70769c4a8e75510d00e`.
- Model fingerprint: `8bfd755893d540ed11e249c36ad7ae73d6a955c78bc57a3899ad18cc8c2bee13`.
- Final Cloud Run package: `f89544b92d69ca55`, 68 public paths, 305,599,371 bytes including compressed variants and private server code.
- The contributor UI uses the simplified Training / Start / Pause / Stop interface. No public coordinator address or engineering-status prose is added.
- Runtime identity: `fly-training-runtime@flyheaven.iam.gserviceaccount.com`, with `roles/datastore.user` only.
- Service capacity: request-based billing, 1 vCPU, 512 MiB, concurrency 8, minimum 0 and maximum 2 instances.
- Firestore deletion protection is enabled. No persistent training state belongs to the Cloud Run filesystem.

## Validation before cutover

Ten Cloud Run serving/packaging tests passed, including a 33 MiB+17 byte chunked response with a matching hash, gzip and conditional requests, HEAD, private-path rejection, API body/origin rules, and isolation headers. This tests the streaming path required for the 207 MiB graph file. The allowlisted package excludes databases, credentials and repository metadata from public paths.

The [Firestore validation record](../cloudrun-firestore/README.md) covers eleven emulator cases plus twenty-seven SQLite coordinator tests. Fifteen simultaneous clients received eight unique assignments and seven waiting responses, with a measured maximum reply time of 8.783 seconds. Migration verifies every historical job before exposing the destination run.

The empty, stopped VM and its task-specific disk, address, firewall rules, subnet and network were removed from `flyheaven`. Its default network was retained. After verified migration and a real contribution, the original task VM, disk, address, two firewall rules, subnet and network were also removed from `monereko-20260809`. Its default network, 42 default subnets and four default firewall rules remain unchanged. The private migration backup was rechecked before deleting the detached source disk.

These infrastructure checks do not establish learned food-seeking, feeding or flight behavior.

## Build and history migration

Cloud Build `292d29b2-8adf-40a9-b2e1-35159fe7e2d8` succeeded. The immutable image is `us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:d74f29a2e3c7700dc77c8f4a2ac6ad05e01867c5859168189e2509639a8c4d36`. Its dedicated builder identity can read the source bucket, write this image repository and write build logs; it has no runtime database role.

The source coordinator in `monereko-20260809` was stopped before its consistent SQLite backup. The private downloaded backup is `migration.sqlite3` beside this report (ignored by Git), SHA256 `6f3a7cdb9bad80b64efe2d39e8258c6db3ee1cb0430c3e99ff579a5ef624285b`.

The verified import into `training_runs/banc888-v1-20260913` preserved **14 accepted results, two generations, sixteen jobs and five contributors**. Every generation, job and contributor field was compared with the source before any scheduling change. Current checkpoint parameters and identities remained identical. The old browser origin cannot finish assignments against the new service automatically, so its one uncompleted active lease was explicitly released using the normal coordinator method. The resulting generation has six completed and two available jobs. No accepted result was removed or replaced.

The canonical migration-content digest is `fc3c83bbdcedee8e0c60d4906dedea69897155b212e3f63e3c563f95c1445618`. Detailed import and comparison outputs remain private under this report directory.

## Live browser and persistence checks

The real HTTPS browser check passed at `https://fly-training-yekt6i27nq-uc.a.run.app/train.html`, using normal certificate verification. Idle and intercepted-outage pages started no worker, WebGL context or model download. The checkpoint button downloaded the exact fresh server response. The normal training path loaded the pinned graph and native body through the streaming server, rendered **80 advancing frames**, and uploaded exactly one real result. The one-second posture trial met its implemented criterion with score `7.994386258168675`; the complete behavioral target remains unvalidated. The check took 42.886 seconds including pause, desktop/mobile screenshots, resuming and releasing a second uncompleted assignment, and Stop cleanup.

Desktop (1440 px) and mobile (390 px) screenshots were visually inspected. The fly, floor/food surface, controls and chart render in the existing theme without horizontal overflow. The preview buffer is 320×180. Public copy contains none of the removed coordinator/model-status prose. No page, network or substantive console errors occurred.

Firestore now holds **15 accepted results**. Deploying a fresh revision, `fly-training-verified-20260913`, preserved that count and the exact checkpoint. Private server/configuration/database paths returned 404, expected browser origins passed preflight, and an unapproved origin was rejected.

The internal startup probe `/healthz` passes inside Cloud Run. Public requests to that exact path receive a Google HTML 404 while the application, API and other paths work; external health checks therefore use `/api/training/status`.

An extra live boundary check found that closing an oversized upload before consuming its body could turn the intended 413 into a gateway 502. A segmented TCP test reproduced the reset locally. The wrapper now drains at most 1 MiB with a ten-second deadline before returning the original 413; the accepted request limit remains 262144 bytes. Thirteen serving/packaging tests pass, including the segmented upload, drain cap and stalled-upload deadline. The final boundary-fix package is `f89544b92d69ca55`; the public assets and scientific identities are unchanged.

Cloud Build `891afbed-d2bf-4692-a4bd-f86761c49eeb` published the final image `us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:c419594b5209dd86d5e49ffd80dea1791f37db27aed4f41a15fd5654cd6b009c`. Revision `fly-training-00006-kqd` serves it. All fifteen live transport checks passed, including the formerly failing oversized request now returning 413, and the persisted count remained fifteen. The deployment helper preserves both Google-reported service aliases while updating images.

## Custom domain

Google's mapping connects `flytrain.morisoba.moe` to `fly-training`. The authoritative Cloudflare servers and public resolvers return the required DNS-only CNAME to `ghs.googlehosted.com`; no restrictive CAA record was found. Google marked the domain certificate provisioned at 22:04:10 UTC. Public TLS and browser verification follow before the canonical domain is reported live. No insecure browser or certificate overrides are used.
