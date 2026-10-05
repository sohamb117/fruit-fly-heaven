These are two staged **full-body** diagnostics for the explicit `activation-amplitude-v1` power contract. They contain no reduced-body adaptations, optimizer updates, saved checkpoints, deployment, or training-state migration.

| Plan | Job parameter 0 | Purpose |
| --- | --- | --- |
| `full-equivalence/plan.json` | `log(.90) = -0.10536051565782628` | Require exact native-physics parity with the saved common-0.90 diagnostic; compare its complete motor-event history separately. |
| `full-amplitude-0p91/plan.json` | `log(.91) = -0.09431067947124129` | A single predeclared amplitude change after the equivalence gate. All other 26 parameters remain exact. |

Both plans reuse the unchanged maintained-native runner and staged full-body environment/scorer/reset. They retain seed 1290888, 0.5 s of unscored live-BANC warm-up, root pose `[0,0,3.5,1,0,0,0]`, and 5 scored seconds or physical failure. XML is byte-identical to the existing full-body baseline (`697aadc8a4df253537f1a03a786ece20fa57f22883d55ea24320a59c15b14583`). Metadata changes only by adding `wing_actuation.power_transfer` from the frozen proposal. All neural/event/tegula settings remain unchanged.

The new request is `exp(theta[0]) * clamp(rawSideForce * 1.999999638880142)`. The fixed activation normalization is a numerical prior retained from the old gain. Parameter 0 now controls amplitude after that clamp, so its maximum is 0 in log space. Both configs use the complete 27-parameter migration with initial `log(.90)`; only the second job overrides this initial coordinate to `log(.91)`. The old saved vector is incompatible with this semantic contract and is not silently reused. No aerodynamic monotonicity or successful-flight claim follows from this change alone.

The exact equivalence reference is `reports/flight-common-power-staging/scale-0p90-run1/result.json`, file SHA `7de3a954ceeeb662be838fbc0f43b6c8778eb7fbcf7c0ae35d3537d8c6ec0fc1`. Its physics digest is `47ebf3b68a2266884fe8d75aca48a7dd53a47ed4d81f0aa7158c38d93feed207` across 1,226 rows of 630 values, including warm-up. The expected physics object is in the equivalence job. Both plans pin the reference path, file digest and digest of all 1,226 motor packets under `diagnostic.equivalenceReference`; the event digest uses UTF-8 `JSON.stringify(motorEvents)` without a trailing newline. Physics parity and complete motor-history equality remain native/saved-record follow-up gates, not preparation results.

`config.json` SHA: `af2db8f153e7799724ac682ebe6835fbe29c8d47f28244ef6de4600a4137e041`.
Model fingerprint: `f4ccdcafeb543f6e5e8a3c49644a925d02d98c16dd0d0a88b81de3758e50f01a`.
Metadata SHA: `d0041dc4843771477f75f40bdd7223a18e3991736a0de4f34e361f5c8ea7a4ea`.
Proposed wing-source SHA: `863b214fd5cb145b703ec04e29b066339a4a35ec88c1bbb955692d893c73020e`.

`../prepare-full-plans.mjs` verifies the frozen eight-test proposal, original baseline and common-0.90 reference, every inherited source pin, unchanged assets, new overrides, complete parameter bounds and metadata before writing. It refuses an existing output directory. The ledger, small source/input archive and `verification.json` preserve preparation evidence; the large reference trajectory remains at its pinned original path. Byte-exact output readback and final source-pin verification passed. No native run, server, coordinator request or live-source edit was performed by this preparation task.
