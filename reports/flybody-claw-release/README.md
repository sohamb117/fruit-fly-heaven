# Claw-adhesion causal replay

Retained hind-claw adhesion contributes to the early loss of balance in this captured BANC trajectory. Releasing it delays the wing impact and inversion, but **does not produce successful flight**. This is a diagnostic result; production controls and contacts remain unchanged.

Run from the repository root:

```sh
node scripts/experiment-flybody-claw-release.mjs
```

[result.json](result.json) records source hashes, controls, muscle state, wing state, native contact forces, root motion, and a 50 µs onset trace. The actual after-COM capture supplies all 200 × 2 ms motor-rate updates. Baseline qpos, qvel, controls, and muscle state match exactly. Seventeen mechanics/model sources are checked against the capture, and the original scene XML is reused unchanged.

The two interventions zero only the six native claw-adhesion controls, or only the two hind-claw controls, immediately before each native step. All other controls retain their original MN → WASM muscle → native actuator equations and body feedback. No geometry, collision eligibility, model parameter, root pose, or external root force is changed. Wing target, power, opening, and deployment arrays remain exactly equal to baseline at every recorded control sample in both interventions.

| Condition, 0.4 s | First 60° body tilt | First wing impact | First inversion | Peak angular speed | Maximum rise |
| --- | --- | --- | --- | --- | --- |
| Original | 0.12265 s | 0.13805 s | 0.14375 s | 2473.28 rad/s | 3.9116 cm |
| All six adhesion controls zero | 0.24145 s | 0.23775 s | 0.24605 s | 2399.72 rad/s | 0.3066 cm |
| Hind adhesion controls zero | 0.22435 s | 0.22465 s | 0.30435 s | 1475.52 rad/s | No rise |

The baseline last positive supporting contact before its wing impact occurs at 66.25/58.10 ms for the left/right front legs, 44.70/35.30 ms for the middle legs, and 116.30/120.70 ms for the hind legs. At 80–100 ms, only the hind claws retain environmental contacts. At 90 ms their control values are 0.4581/0.4329; with native gain 0.985, actuator-force magnitudes are 0.4512/0.4264 g·cm/s². Together those magnitudes are about 0.91 bodyweights. Contact normal forces are logged separately from actuator adhesion and must not be conflated with these commanded magnitudes.

At 136 ms, baseline body tilt is 66.03°, versus 16.88° with all claws released and 18.31° with only hind claws released. Tilt here combines roll and pitch; the corresponding forward-axis pitch-up angles are 38.08°, 15.57°, and 17.24°. Effective wing power is identical at approximately 0.686 on both sides. Hind release therefore materially changes the body motion before the original wing collision without reducing the wing program.

This establishes contribution in the matched recorded-input trajectory. It does not establish the correct physiological claw-release controller, prove that all pre-impact imbalance comes from adhesion, or predict a closed-loop neural rerun. Both interventions still overturn later. The result supports investigating takeoff coordination and the claw motor interface; it does not justify permanently disabling adhesion.
