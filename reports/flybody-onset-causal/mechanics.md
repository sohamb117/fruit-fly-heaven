The original BANC onset contains two distinct failures: an early powered tumble, followed by much larger motion amplified by wing–terrain impacts. Disabling wing–terrain contacts does not prevent the first tumble. It eliminates almost all upward launch in this matched replay. Wing aerodynamics advance the tumble from 0.25155 s to 0.11005 s; powered wing motion can still destabilize the supported body when both wing fluid forces and wing–terrain contacts are suppressed.

All five conditions replay the same 200 recorded motor-rate frames (0.4 s), initial body/muscle state, terrain, six wing servos with gain 18, other actuator equations, and muscle/internal-state equations. Controls still respond to each condition's physical joint state through the unchanged muscle/servo equations. The instrumented baseline reproduces recorded qpos, qvel, controls and muscle state **bit for bit**. Every condition has zero external applied root/body force. These are immediate mechanics counterfactuals with recorded neural input; sensory feedback to a new BANC run is not simulated.

| Condition | First airborne (s, 2 ms sampling) | First up-Z < 0 (s, 50 µs sampling) | Peak angular speed (rad/s) | Maximum rise from initial root (cm) |
|---|---:|---:|---:|---:|
| Original | 0.104 | 0.11005 | 2720.56 | 2.23289 |
| Only wing–environment contact disabled | 0.104 | 0.11005 | 154.11 | 0.03212 |
| Only wing fluid force suppressed | 0.254 | 0.25155 | 2085.80 | 3.17579 |
| All wing contact disabled | 0.104 | 0.11005 | 154.11 | 0.03212 |
| Wing–environment contact and wing fluid suppressed | 0.254 | 0.25155 | 117.99 | ≤ 0 |

Baseline and the wing–environment-disabled condition have identical root/wing positions and velocities through 0.11265 s. Their first difference occurs at 0.11270 s, the first positive wing–terrain force, after the first overturn. No wing self-contact occurs anywhere in the baseline. Disabling all wing contact produces exactly the same samples as disabling only wing–environment contact.

The later baseline maximum is 2720.56 rad/s at 0.33780 s, where the 50 µs trace contains three positive wing–terrain contacts. The original 2 ms recorder sampled a slightly smaller maximum of 2663.97 rad/s at 0.338 s, after those contacts ended. These refer to the same trajectory at different observation frequencies. They must not be described as the angular speed at initial takeoff.

Contact coordinates below use native world XYZ in **cm**. Normals are dimensionless; normal force uses native **g·cm/s²** (multiply by 1e-5 for N). For original Three.js world coordinates in mm, map `(X,Y,Z)` to `(10X,10Z,10Y)`.

| Contact event | Time (s) | Ground–wing geom IDs | Contact XYZ (cm) | Normal | Normal force |
|---|---:|---|---|---|---:|
| First positive wing contact | 0.11270 | 0–82, right membrane | (-3.196573, -1.515250, 1.659960) | (-0.339834, -0.205404, 0.917781) | 0.50970 |
| Stronger contact in that same first step | 0.11270 | 0–82, right membrane | (-3.195814, -1.506966, 1.661943) | (-1.000000, 0.000000265, ≈0) | 13.34220 |
| Strongest onset-window wing contact | 0.11340 | 0–82, right membrane | (-3.194846, -1.555371, 1.649470) | (-0.569211, -0.208950, 0.795197) | 72.49664 |
| Strongest wing contact over 0.4 s | 0.32560 | 0–78, left membrane | (-1.921566, 1.543018, 0.370005) | (0.113095, 0.602065, 0.790397) | 350.20108 |

The near-horizontal normal at the first impact deserves comparison with the original fruit mesh and native terrain. These mechanical experiments alone do **not** establish that the contacted surface is an invisible fill or a spurious terrain edge. They do not justify deleting or softening production wing collisions. Native wing colliders are inherited from the published FlyBody asset: 77/78 are left brown/membrane, 81/82 right brown/membrane; body IDs are 9/10. Environment IDs are 0–65 (ground, ceiling and walls). A pair-mask audit verifies that the targeted contact condition changes only these 264 wing–environment pair eligibility entries; all other pair eligibility remains unchanged. No geometry, inertia, damping, timestep, actuator or contact-response coefficient changes.

Fluid suppression changes only interaction coefficients of geom 79 and 83 from 1 to 1e-300. This preserves the ellipsoid branch and suppresses wing generalized fluid torque to at most 1.87e-300, retaining non-wing fluid forces. Exactly zero would enable MuJoCo's inertia-based fluid fallback; zeroing drag/lift coefficients would retain viscosity. The diagnostic asserts Euler integration because MuJoCo 3.13's implicit fluid derivatives are not scaled by the interaction coefficient. See the [native fluid force implementation](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_passive.c#L870) and [fluid derivative implementation](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_derivative.c#L2920).

The six actual wing control trajectories, wing states, root states, fluid/passive/actuator/constraint/bias forces, and every native contact are saved at 50 µs resolution from 0.075–0.130 s. Two-millisecond samples cover the entire run; explicit first/peak events retain full contact data outside the onset window. The pitch controls reach their original ±1 limit for 17.80% of the onset-window samples; yaw/roll controls do not saturate there. This is evidence about the current reduced controller, not a demonstrated new actuator indexing or units error.

Reproduce with `node scripts/diagnose-flybody-onset-mechanics.mjs`. It first checks all 16 captured mechanics/runtime source hashes and requires the formal capture/replay gate. [mechanics.json](mechanics.json) records exact hashes, mutations, event data and results. Production files and UI were not changed.
