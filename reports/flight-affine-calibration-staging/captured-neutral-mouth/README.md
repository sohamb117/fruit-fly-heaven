# Completed restrained force-space calibration

The **54-measurement** run completed and met its predeclared four-output trim
tolerances on the first measured trim correction. Center and final-trim repeats
matched selected state and full wrench/wing traces exactly; source pins were
unchanged. [result.json](result.json) contains the full evidence and named T.

| Measured quantity | Result |
|---|---:|
| Common power, unit power gain | 0.7920781393 |
| Vertical aerodynamic force / body weight | 0.9964163166 |
| COM torque x / (BW × 0.27 cm) | +0.0000116028 |
| COM torque y / (BW × 0.27 cm) | −0.0000299605 |
| COM torque z / (BW × 0.27 cm) | −0.0000510475 |
| Unconstrained horizontal force / body weight | **0.0327932700** |
| Jacobian row-scaled rank / condition | 4 / 2.4706597 |

The tolerances were ±0.01 BW vertically and ±0.0001 for each normalized torque.
The unscaled normalized-output Jacobian condition number was 122.2403; the
smaller row-scaled condition does not imply that force and torque have equal
physical sensitivity.

The unchanged full model retained all 51 joints and 56 actuators. The rig held
the root level at z=10 cm and 42 leg hinges at their captured 100 ms positions.
Rostrum and haustellum were explicitly set to metadata.neutral after the original
captured-mouth fixture produced an active mouth/head contact. No collision was
disabled. Wings evolved dynamically through their existing actuators, with
synthetic normalized force inputs, unit steering gains, 227.204212186366 Hz,
100 ms warmup and approximately 16 measured wingbeats per observation. There
were no contacts or instability warnings in the accepted run.

At the measured trim, pitch shape-residual clipping occurred in **9.09% left /
18.18% right** of 352 wing updates. Pitch servo control clipped in **13.64%** of
updates on each side. Other axes and all joint targets had zero measured
clipping. These clamps remain part of the measured nonlinear plant.

This is a **restrained aerodynamic trim**, not stable hover or demonstrated
free-flight control. Horizontal force remains nonzero and unconstrained. Native
event-muscle dynamics and BANC were bypassed for force-space identification;
attainability and startup behavior require separate evidence. F0 is the recorded
event-force reference, while T is this measured force-space trim. A proposed
later `T + G(F−F0)` mapping must preserve that distinction.

Plan SHA-256: `c92cf51fa042b03fa5231cc5cc638421c9e737ae993ace029630fc6cc6b794c7`.
Result SHA-256: `d9ae600d2adbd3ca0683122a7a4e9670025c4937dc256b588bc05eda31a6eb66`.
