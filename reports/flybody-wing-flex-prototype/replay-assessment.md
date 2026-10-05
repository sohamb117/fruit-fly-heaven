# Passive wing-flex prototype: retain as research, do not integrate

The single corrected motor-neuron replay **fails** the behavior and calibration-range checks. Production files remain unchanged. The static spring experiment is valid within its measured load range; it does not establish a useful dynamic wing model.

## What was verified

The independent design review found the neutral split, full rotated inertia/parallel-axis calculation, hinge direction, and unit conversion consistent. Native checks subsequently preserved whole-wing and whole-body mass/COM/inertia, with tensor error at most `2.65e-23 g·cm²`. Sixteen bilateral signed load probes returned `0.0240013–0.0240853 N/m`; their applied root generalized force was zero. The nominal conversion is `0.024 N/m × 1000 × (0.114 cm)² = 0.311904 g·cm²/s²/rad`.

This stiffness describes a particular wing-vein load point over a low-force range, not the location or damping of this artificial hinge. The box-tip proxy, damping prior, and lack of coupled aerodynamic deformation remain explicit. [Primary stiffness measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6361194/).

The first replay exposed an unintended model-construction change. Adding massive distal bodies also enabled their default inertia-box fluid forces, on top of the unchanged proximal whole-wing aerodynamic ellipsoid. MuJoCo selects its fluid model separately for every massive body, including welded children. [MuJoCo 3.13 fluid selection](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_passive.c#L807).

[fluid-fallback-audit.json](fluid-fallback-audit.json) checks 70 identical captured states before the original first wing impact. The unwanted fluid term changed root-Y force by up to `0.5456814084 g·cm/s²`; spring and damper differences were zero. A positive-negligible ellipsoid interaction sentinel suppresses only this accidental fallback. After that explicit correction, original named fluid and total passive force entries match exactly, with bias error at most `9.55e-15` and shared mass-matrix error `1.70e-21`.

Both corrected prototype XMLs require the recorded post-compile `geom_fluid[0] = 1e-300` overrides. XML alone is insufficient. The assay asserts the audited Euler integrator and expected initial coefficient before applying each override. This is numerical suppression of duplicate aerodynamic force, not an added stabilizer. The failed first artifacts are preserved under [before-distal-fluid-correction](before-distal-fluid-correction/).

## Corrected matched motor-neuron replay

[mn-replay.json](mn-replay.json) contains the single corrected comparison. It uses the actual after-COM capture's 200 × 2 ms motor-rate updates, real WASM muscles, native joint-dependent feedback, and the existing wing-step implementation. It restores joint qpos/dofs, actuator and activation slots, body force buffers, rest pose, and body/site metadata by native name; new passive bends start at zero. It checks actuator transmission types and named targets. Original habitat assets/geoms and the exact geom-to-food resolver are retained. No neural rerun or fixed-control substitution occurs.

The original baseline is bitexact for qpos, qvel, controls, and muscle state. Applied root forces remain zero in all conditions.

| Condition, 0.4 s | First wing impact | First inversion | Peak angular speed | Maximum rise |
| --- | --- | --- | --- | --- |
| Original body | 0.13805 s | 0.14375 s | 2473.28 rad/s | 3.9116 cm |
| Corrected rigid split | 0.13810 s | 0.14430 s | 2276.21 rad/s | 0.06264 cm |
| Corrected flexible split | 0.15205 s | **0.12420 s** | 1732.62 rad/s | No rise |

Before the original first impact, the rigid split agrees over 2760 native steps to `3.79e-13` in root qpos and `1.67e-10` in root qvel. This resolves the initial construction confound. Its later collision trajectory differs substantially, demonstrating why a rigid-split shape control was necessary: changed mesh contact geometry alone changes the launch.

The flexible body instead overturns before any wing–environment impact. Its static calibration range is exceeded at **0.05605 s**, with no wing–environment contact. Maximum pre-impact bend reaches `0.5663 rad` (32.4°); overall bend reaches `2.9993 rad` (171.8°). Maximum normal tip displacement is 1.140 mm. At least one bend is outside the static-range displacement bound for **5709/8000 native steps**. These flags compare against the reported 3.3 µN linear-load deflection; they are not a dynamic failure law.

The compliant segment therefore changes inertial wing behavior before contact and then reaches very large, unvalidated deformations. A lower maximum rise cannot be credited as a successful contact remedy. Retain the implementation, native calibration, named-state mapping, and failed trajectories as reproducible research artifacts. Do not deploy this prototype or tune damping merely to obtain an upright fly.

## Reproduction and provenance

```sh
uv run --no-project --with mujoco==3.13.0 --with numpy scripts/experiment-flybody-wing-flex.py
node scripts/experiment-flybody-wing-flex-replay.mjs
```

The replay checks the native-probe report and all source hashes before proceeding. Model XML hashes for the corrected comparison are `f47a1267906493872fe0e741095df544812539c6d41032746352bf22dd2db444` (flex) and `80489491b5b2eb55b9a096d74945c7fcac6fbced9e897f12b02d1b4e487a6bc6` (rigid split). Exact scene hashes and required native overrides are in each replay case. The production native XML remains `8246f5ff573dcb60d05e8d4500b102a4dbac577ff1c73d716f5d8ecf3d09abd7`.
