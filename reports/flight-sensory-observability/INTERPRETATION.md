# What the current flight feedback can distinguish

The diagnostic uses the recorded 200 ms airborne native pose and muscle gates, `FlyBodyWorld.copyPose`, the complete prepared sensory manifest, and the production `SensoryEncoder`. It creates static native-state probes and refreshes forward caches; it does not advance the neural network or integrate a new trajectory. Every probe uses a fresh encoder. The `dt=0` snapshot convention avoids inventing finite-difference joint motion between probes; explicitly measured native joint velocities remain available.

All 6,837 requested-rate entries agree exactly between positive and negative root angular velocities at 1, 10, 25 and 50 rad/s on each axis, and under a mixed-axis sign reversal. For each fixed neuron index, the current mapper is a deterministic function of this rate, so equal vectors necessarily produce equal added currents. The current mapper was inspected, not recalibrated or simulated for this assay.

| Native information | What reaches added sensory current |
|---|---|
| Three root-local angular-velocity components | Their Euclidean magnitude feeds 449 rotation afferents. The third component also contributes through its absolute value to 50 fallback self-motion cells. No component sign remains. |
| World linear velocity | A scalar speed norm, converted to display millimetres per second, enters 50 self-motion and 579 antenna fallback inputs. Its contribution clips at 30 mm/s. Direction and independent COM vertical velocity are absent. |
| Root attitude | Unsigned tilt `acos(up)` enters antenna fallback inputs. Heading affects two spatial odor samples. Signed roll and pitch, a gravity vector and direct attitude error do not enter the body-sense mapping. |
| Root position | Position and heading influence the bilateral odor field. Altitude is reported but is not a direct sensory input. |
| Leg configuration | 403 position afferents use absolute raw tibia/coxa angles. The 50 fallback cells use side-averaged joint-angle norms and finite-difference speed magnitudes. |
| Leg velocity | 149 velocity afferents use absolute tibia velocity; 403 vibration afferents use collision-gated absolute tibia velocity. Tibia reversal at fixed posture gives identical encoder output. |
| Leg contact | Per-leg native normal load drives 61 load afferents; nonclaw collision magnitude drives 2,965 touch afferents. Native organ contacts separately route taste. Contact force direction and each bristle's local receptive field are not represented. |
| Wing and haltere activity | Organ and side remain distinct. Haltere asynchronous force and maximum named steering force gate its afferents; deployed wing power gates wing-base afferents. |
| Actual wing angles/velocities, wingbeat phase/frequency | Native values exist or are exposed in the feedback object, but the current rotation mapping does not consume them. It uses activation proxies rather than local strain. |
| Vision | Disabled in this training environment; its 1,093 mapped visual inputs receive zero added drive. |

Roll and pitch of equal speed produce the same whole input vector. Yaw magnitude can differ on the 50 fallback cells, so it is inaccurate to say every rotation axis is completely indistinguishable. Switching a unilateral wing gate from left to right changes all 121 wing-base afferents. Bilateral spatial identity is preserved even though rotation direction is absent.

At full power the rotation-rate formula clips at 100 Hz once angular speed reaches 23.75 rad/s. The recorded 100–280 ms window contains 91 feedback samples: left and right wing-base rates clip on two and one samples respectively. Haltere rates do not clip in that window. More significantly, the left asynchronous haltere power is zero on all 91 samples: its 171 afferents receive only a 2.73–2.98 Hz steering-gated baseline, with no rotation-dependent term. The right haltere remains active. These are findings for this captured sequence, not a claim about every candidate.

The downstream body-channel cells are spiking with a 2 ms refractory parameter. Their rate-to-current profile allows up to 200 Hz, while the body encoders already cap requests at 100 Hz. Positive requests below 2 Hz map to the mapper's 2 Hz floor; zero is zero added current. Recurrent activity, intrinsic activity and internal-state modulation remain separate from this added current.

The anatomical labels support more grouping than the current four organ/side scalars: 34 cell-type labels, 35 organ-by-type groups and 65 organ-by-side-by-type groups. `SApp11` occurs in both organs, so organ must be part of a grouping key. The raw-data join agrees with the prepared organ/side/type labels for all 449 cells. Annotation subclasses identify 328 haltere campaniform cells, 102 wing-base campaniform cells and 19 wing-base orphan cells. Neither prepared nor joined raw records supply receptor-field, preferred-axis, direction-sign or firing-phase labels; raw cluster/morphology grouping columns are null for this set.

This is an immediate feedback limitation, not a proof that all recurrent control is impossible. Motor history, evolving contacts and bilateral odor changes can provide indirect information over time. However, the 27 trainable wing gains cannot add a signed signal that the current encoder removes. Any proposed signed receptive-field or phase mapping must be an explicit, independently testable modeling prior; the existing anatomical type names do not establish those directions.
