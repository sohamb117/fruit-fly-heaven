Both v1 snapshots independently pass all timing, event-order, current-readback, duplicate-sham and sign-folded-control gates. The files `vision-airflow-v1.assay-{0,1}.independent-audit.json` preserve the detailed checks and side-specific pulse statistics. No original trace or runtime source was changed.

Each table entry is the number of wing motor neurons with a different exact event history; parentheses give the subset with different net spike counts. Phase labels are offsets of the prescribed sensory prior, not identical absolute phases across the two snapshots.

| Current cap(pA) |500ms,phase0|500ms,phaseπ|600ms,phase0|600ms,phaseπ|
|---|---:|---:|---:|---:|
|100|0(0)|0(0)|0(0)|0(0)|
|200|2(0)|0(0)|0(0)|0(0)|
|400|5(0)|5(3)|8(1)|10(1)|
|800|7(2)|15(1)|12(1)|0(0)|

The signed sensory perturbation can reach actual wing-MN event timing and the event-aware decoder. Many effects would be missed by net counts alone. Responses depend on dose, phase and conditioned neural state; the400/800pA comparison does not support a monotone dose-to-useful-feedback curve.

Native left/right haltere powers were `[0,0.603522]` at500ms and `[0,0.447134]` at600ms. Consequently all171 left afferents receive zero direct haltere-prior current. The157 right afferents remain active. This limits the tested sensory route to the right side; it does not remove left neurons or prevent network effects on left motor neurons. In the500ms,800pA,phaseπ arm, negative pitch removes a23ms left-B1 spike shared by sham and positive pitch.

At500ms, the800pA first wing-event contrasts occur12.5/18.5ms after current onset; first decoder-feature/output contrasts occur13/19ms. At600ms,400pA yields17.5/23.5ms wing-event onsets and18/24ms feature onsets. The600ms,800pA,phase0 onset is18ms(19ms for features); phaseπ changes two afferent event histories without a wing-event/output difference in the40ms observation window.

Maximum recorded decoder-output differences across the two phases are0.0040616 at500ms and0.0010852 at600ms. None of the tested outputs clips at its power or steering boundaries, so clipping does not explain these small or absent output contrasts. At100pA—and600ms/200pA—full state buffers still differ despite matching selected event histories. These are subthreshold/latent differences, not evidence of no information.

The phase0-versusπ mean odd-feature cosine is0.0512 at500ms/800pA and−0.1263 at600ms/400pA. These descriptive values do not establish either useful phase generalization or impossibility of learning a phase-conditioned response. Signed probes replace pitch rate with±5rad/s; at600ms the native pitch rate was1.552575rad/s, so the even contrast against native sham includes an operating-point shift. No body is integrated during these branches, and output direction has not been checked against corrective body torque.

V1 remains pilot evidence with its recorded metadata caveat. The final v2 records and exact initial-state comparison remain pending. The data establish conditional causal sensitivity, not a physiological calibration or stable-flight controller.
