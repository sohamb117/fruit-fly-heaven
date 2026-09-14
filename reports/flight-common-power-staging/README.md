# Common post-clamp power diagnostic

Prepared only: three separate one-fly plans scale the two existing clamped wing-power requests by **0.85, 0.90 or 0.95**. No native/neural evaluation has been run by this preparation. This is a numerical mechanics ablation, not measured physiology, a new training contract, or model promotion.

The complete source difference in each override is:

```js
// Original
requested=[clamp(left*p.powerGain),clamp(right*p.powerGain)]
// Diagnostic; S is a fixed source literal
requested=[S*clamp(left*p.powerGain),S*clamp(right*p.powerGain)]
```

All 27 assigned parameters remain exactly equal to the saved baseline, including power log gain `0.693147` (approximately 2×). When both original requests saturate, both diagnostic requests equal S; below saturation their independent live differences remain. Zero input remains zero. This is not an imposed constant-power command.

Steering, frequency, target/servo limits, deployment and ramp code are unchanged. The existing `requested > 0.01` deployment gate receives the scaled request, so its equivalent original-request threshold becomes `0.01/S`; this is a declared consequence of the expression change. Changed power selects different wing-table interpolation and multiplies the downstream steering residual, so this is not simply a scalar lift change. The intervention is active throughout warm-up from time zero, and subsequent physical feedback can change BANC output, muscle state and power. The plans do not promise that those trajectories remain equal.

The source is pinned to original `web/flybody-wings.js` SHA-256 `10a44cb2ed11835ed3a151081586095e6bd24527f0c20926c9a006783fec7084`. Each config changes only its wing executable asset digest and the derived model fingerprint. Original metadata/XML, the full native body, neural/event profiles, seed **1290888**, all parameters, and the **0.5 s live-BANC restrained warm-up followed by 5 scored seconds or physical failure** are retained from `plans-v1/maintained-baseline`.

The canonical custom runner reads the overridden code at the canonical module URL, preserving relative imports. Each plan pins that runner, builder, source-transform helper, tests, generated source, parent plan/config, and all inherited sources. Preparation verified 57 assets and 82 inherited/preparation source pins; each plan adds its own generated wing source pin. Existing output directories are refused, and every saved artifact is read back against its digest. No server/coordinator, optimizer or production files are changed.

Prepared plans:

- [0.85](plans-v1/scale-0p85/plan.json)
- [0.90](plans-v1/scale-0p90/plan.json)
- [0.95](plans-v1/scale-0p95/plan.json)

[The ledger](plans-v1/ledger.json) records exact plan/config/source/model hashes. It also archives the original source, parent plan/config and preparation scripts.

Four pure tests passed: exact source-delta validation, scalar-1 byte equality over 2,048 interpreter steps, saturation/bilateral/zero behavior, and the unchanged deployment threshold's interaction with scaling. These execute the JavaScript wing interpreter only, without a native body or BANC allocation. `prepare.mjs` also passed the syntax check. These results validate implementation and provenance, not flight.

An independent read-only review of the builder, transform and tests found no blocking defect. It confirmed the deployment threshold, retained bilateral asymmetry below saturation, and the wing-table/steering consequences described above.

```sh
node --test reports/flight-common-power-staging/variant.test.mjs
node reports/flight-common-power-staging/prepare.mjs reports/flight-common-power-staging/NEW_PLAN_DIRECTORY
```

The parent may execute each prepared plan serially with the existing `reports/flight-maintained-native-staging/evaluate.mjs` runner after review. No automatic evaluation is part of this builder.
