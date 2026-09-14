These three prepared one-fly plans separate adapter compatibility from the effect of structurally freezing nonwing joints. They are diagnostics, not optimizer updates, saved checkpoints or evidence of successful flight.

| Plan | Body | Amplitude | Gate / comparison |
| --- | --- | --- | --- |
| `full-adapter-equivalence/plan.json` | Original full XML and amplitude-profile metadata; new body adapters with reduction disabled | 0.90 | Require saved physics digest `47ebf3b68a2266884fe8d75aca48a7dd53a47ed4d81f0aa7158c38d93feed207` and compare complete motor history against common-0.90 reference. |
| `reduced-amplitude-0p90/plan.json` | Source-pinned free root plus six dynamic wing joints | 0.90 | Distinct reduced-plant diagnostic after the compatibility gate. |
| `reduced-amplitude-0p91/plan.json` | Identical reduced model/config | 0.91 | Only job parameter 0 differs from reduced-0.90. |

Every job uses seed 1290888, the unchanged maintained-flight scorer, 0.5 s of live-BANC/event/muscle warm-up, root pose `[0,0,3.5,1,0,0,0]`, and five scored seconds or physical failure. The reset fixture holds only the root during unscored warm-up; after release the root is free and its direct-write count must remain unchanged. No root weld, prescribed motor stream, direct muscle-force injection or corrective flight controller is added.

All plans retain the complete 27-coordinate `activation-amplitude-v1` migration and the same proposed wing source. Fixed activation gain is `1.999999638880142`, with learned amplitude `exp(theta[0]) <= 1`; initial parameter 0 is `log(.90)`. The second reduced job uses `log(.91)`. All other 26 coordinates and all neural/event/tegula settings remain exact. Historical vectors with the old power-coordinate meaning and optimizer histories are not reused.

The full model's XML and metadata bytes match the power-parameterization parent. Reduced XML exactly matches `model/model.xml`, and reduced metadata differs from its validated source only by adding the **same** `wing_actuation.power_transfer`. Its explicit body variant is `fixed-nonwing-flight` in the reset contract and `{schemaVersion:1,kind:'fixed-nonwing-flight-v1'}` in body metadata. Every config's initial-condition hash matches its actual metadata asset. The reduced pair has identical config hashes/fingerprints; its job vectors supply the amplitude difference.

Full-adapter config SHA: `0d3c1ac1029242d540d90b483dc8548d6d0a6b68feac9b3c4ea51c3883c9d937`.
Reduced config SHA: `ebd369b5de6d7f3242e4a44addbb03872cf1c2cc337030eccddbb07ea7881f7c`.
Reduced metadata SHA: `68dc8650e87b30061dd70e45c5e5b30bfc4c6828cf7f3db1f3124ab74cd30caf`.
Reduced XML SHA: `6c255aa9806f324894f105afef36919429b2a74b366125f922c3538cb416112c`.

`../prepare-plans.mjs` verifies all original and staged source pins, adapter validation, reset contracts, native model validation v2, parameter migration, asset digests and readback before handing off. It pins the unchanged native runner and archives small preparation inputs. The source-pinned native validation was performed separately by the parent task: compilation and zero-time forward passed for the exact reduced XML, with all 44 removed hinges absent and body/inertial/geometry/site identities and reference transforms retained. No dynamic native timestep was advanced by that validation.

Seven adapter mock tests and eight fresh reset compatibility mock tests pass. The fresh reset output is `../plan-preparation-reset-tests.tap`. Those eight tests cover the helper/contract; the environment mouth-anchor fallback receives a syntax check, while its native compatibility is the full-adapter evaluation's purpose. Preparation does not replace that native gate. The reference physics includes an eventual physical failure, so matching it establishes compatibility rather than successful maintained flight.

The original source-pinned artifacts remain unchanged. `verification.json` records preparation checks, and `ledger.json` records plan hashes, source hashes and native-validation provenance. This preparation task started no native simulation/server, edited no live sources and wrote no training checkpoint. The reduced variant cannot establish coordinated leg-assisted takeoff, landing, probing or feeding.
