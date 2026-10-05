# Takeoff coordination identification

Eighteen deterministic replays show that the motor-to-body interface needs separate wing-power and claw-adhesion calibration. Reducing every wing motor rate is not an adequate substitute. Moderate hind-adhesion reductions can avoid inversion for the recorded 400 ms without changing the wing commands, while complete release still fails. **None of these runs establishes sustained flight.**

## Reproduce

```sh
node scripts/audit-takeoff-coordination.mjs
```

The assay reads `reports/flybody-solid-wing-repair/after-com/onset/capture.json`, reuses its scene XML and all 200 recorded 2 ms BANC motor updates, and stops at exactly 400 ms. Seventeen source/model artifacts must match the capture hashes before execution. The baseline then must reproduce every recorded qpos, qvel, actuator control and muscle-state element exactly; all four maximum errors were zero. No production file, collision geometry, neural weight, external root force, or root pose is changed.

These measurements preceded the optional interface branch and WASM rebuild in this checkout. Reproduction requires the capture's matching executable sources; running the historical command against the changed checkout intentionally fails its hash guard. The current live-neural candidate tests are recorded in [interface training validation](../interface-training-validation/README.md).

`result.json` contains the input and script hashes, 18 case specifications, native event metrics, 2 ms state/force snapshots, the first wing impact, and the 136 ms comparison. It is approximately 12 MB and deliberately excludes the full original capture. `summary.json` is the compact event summary. Both are generated reports. Native contacts, inversion and other events are checked at 50 microsecond resolution; 2 ms output samples must not be treated as sufficiently sampled wing-cycle force averages.

`verification.json` records eight passing numerical/provenance checks, including reproduction of all 200 root positions and velocities from each of the prior baseline, all-claws-zero and hind-claws-zero experiments. Those three trajectories match bitexact. It also checks identical wing commands for claw-only interventions, noncompounding adhesion factors, the exact executed script hash, all 18 final horizons, and zero applied forces. The script passes `node --check`.

Every sample records requested DLM/DVM rates; muscle excitation, activation, fatigue and force; wing opening, deployment, power, targets, actual angles and actuator torques; root acceleration and angular velocity; COM; supporting/claw contact forces; raw/applied claw controls; and fluid, constraint and actuator wrenches about COM. Gravity is included separately. The applied-force arrays remain zero in all cases. Contact normal forces and commanded adhesion magnitudes are separate quantities. COM and wrench records use the native force-evaluation configuration; root poses describe the immediately following integrated state.

The claw interventions multiply the native adhesion control at every solver step. Fractional factors are restored between steps, so they do not compound accidentally. Power interventions multiply only the recorded DLM/DVM motor rates before the existing 80 Hz excitation calibration, WASM activation/fatigue and wing mechanics. They are not torque multipliers. Other recorded motor rates remain untouched; limb controls still respond through their original body-dependent equations. Wing power, opening, deployment and target arrays are exactly equal to baseline at every 2 ms sample for every claw-only intervention.

## Results

Times are milliseconds from the captured start. A dash means the event did not occur within 400 ms, not that it cannot occur later. Tilt is the angle between the body up axis and world vertical; it is not pitch alone.

| Intervention | Tilt at 136 ms | First wing impact | First inversion | Peak angular speed, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Original | 66.03° | 138.05 | 143.75 | 2473.28 |
| Hind adhesion ×0 | 18.31° | 224.65 | 304.35 | 1475.52 |
| Hind adhesion ×0.25 | 37.55° | — | — | 56.43 |
| Hind adhesion ×0.5 | 48.41° | 313.95 | — | 115.43 |
| Hind adhesion ×0.75 | 57.67° | — | — | 71.72 |
| All adhesion ×0 | 16.88° | 237.75 | 246.05 | 2399.72 |
| All adhesion ×0.5 | 50.05° | 170.95 | 282.30 | 2018.50 |
| DLM/DVM rates ×0 | 20.82° | — | — | 3.68 |
| DLM/DVM rates ×0.25 | 35.41° | 279.70 | 280.70 | 1813.87 |
| DLM/DVM rates ×0.5 | 37.58° | 203.60 | 208.55 | 1541.21 |
| DLM/DVM rates ×0.75 | 59.45° | 190.80 | 195.75 | 1509.17 |
| Hind adhesion ×0; power rates ×0.5 | 59.77° | 145.90 | 151.45 | 1412.18 |
| Hind adhesion ×0.5; power rates ×0.5 | 47.25° | 161.90 | 301.00 | 1485.51 |
| Hind release starts 0 ms | 16.85° | 366.00 | 374.60 | 1266.89 |
| Hind release starts 20 ms | 18.27° | 205.15 | 250.20 | 2571.03 |
| Hind release starts 40 ms | 14.79° | 232.10 | 242.15 | 1363.07 |
| Hind release starts 60 ms | 8.67° | 199.70 | 208.90 | 1789.03 |
| Hind release starts 80 ms | 47.26° | — | — | 91.30 |

The release experiments use a 40 ms smoothstep ramp from full to zero hind adhesion, starting at the listed time. They are explicitly open-loop identification probes, not a proposed deployed controller or an inferred biological release program. Their times were chosen to bracket the original interval during which front/middle support is lost and hind contacts remain.

Three observations matter for training:

1. **The shortest posture score can choose the wrong candidate.** The 60 ms release has only 8.67° tilt at 136 ms yet inverts at 208.90 ms. The 80 ms release has 47.26° tilt at 136 ms and does not invert within the capture. Starting the release at 0 ms postpones inversion to 374.60 ms: even a 300 ms check would miss this failure.
2. **Weakening wing drive and calibrating attachment are different interventions.** Every tested nonzero rate attenuation alone still inverts. Hind adhesion ×0.25 and ×0.75 avoid wing impacts without any wing-command difference. Combining two individually plausible interventions can make the onset worse: half-rate drive plus complete hind release inverts at 151.45 ms, substantially earlier than either intervention alone.
3. **Surviving the horizon is not flight success.** The hind ×0.25, ×0.5, ×0.75 and 80 ms-release conditions never rise above their initial height and end with 9, 9, 12 and 10 environmental proximity contacts, respectively. Their recorded maximum body tilts remain approximately 57°, 57°, 60° and 55°. Zero power avoids launch altogether. An optimizer that rewards only non-inversion can select a stationary or descending fly.

At 90 ms the original body has only two environmental contacts, both hind claws. Their actuator-force magnitudes are 0.4512 and 0.4264 g·cm/s², together approximately 0.91 bodyweights. Their separately measured contact normal forces are 0.3811 and 0.4181 g·cm/s². This is a directly measured attachment condition during the pre-impact loss of balance, not a claim that normal force and adhesion should be summed as support. At 136 ms, effective wing power is 0.6862 per side in every claw-only case.

## Calibration and training implications

Keep the existing coordinator and contributions, but keep the new parameter contract and checkpoints separate from the earlier neural-only run. This single trajectory cannot identify a biological parameter set or justify deploying its best outcome.

| Candidate interface parameter | Initial exploration range | Evidence and boundary |
| --- | --- | --- |
| Hind-claw effective adhesion gain, independent of front/middle gains | 0.25–1 × current | This assay directly probes the range; the apparent benefit is specific to this initial pose and replay. Zero remains an ablation control, not the recommended default. |
| Front and middle claw gains, independently selectable | 0.25–1 × current | An assumed search range. The global 0.5 control underperforms a hind-only 0.5 change, so one global claw gain hides a useful distinction. Individual front/middle gains were not identified here. |
| Claw activation/deactivation time constants | 15–80 ms | An assumed calibration range, not fitted physiology. Timed smooth release demonstrates sensitivity to timing, but a scripted clock ramp is not a measured muscle time constant and must not become the training policy. |
| Separate DLM and DVM rate-to-excitation parameters | Defined by the controlled muscle-transfer assay, not this replay | Both groups were attenuated together here. The results demonstrate that this adjustment cannot substitute for attachment calibration; they do not identify a correct Hz-to-force curve. |

First calibrate attachment strength and dynamics against independent force or kinematic constraints, alongside the wing transfer assay. Then evaluate the resulting model with fresh, longer BANC rollouts and multiple initial poses/surfaces. A takeoff score needs sustained upright separation, acceptable displacement/velocity, and explicit collision/inversion penalties; brief contact loss alone does not qualify. Use at least a 1 s follow-up horizon, because this replay ends at 400 ms and other corrected-model tests have failures beyond 700 ms. Do not synthesize extra recorded motor input by repeating the final frame.

No parameter from this assay has been promoted into production, and no successful food-localization → approach → landing → feeding → takeoff sequence is claimed.
