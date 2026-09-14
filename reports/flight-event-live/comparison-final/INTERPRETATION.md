The real BANC event path runs correctly, but this single comparison does not show a flight improvement. Qualifying flight increased by 8 ms (166 → 174 ms), while excessive-rotation termination occurred 88 ms earlier (480 → 392 ms). Neither arm maintained one second of qualifying flight or landed successfully. The 0.024 reward increase reflects the unchanged progress formula; it is not evidence of stability or learning.

The baseline reproduced the earlier corrected-sensory trial exactly: all 241 motor packets, the 241×630-value physical digest, initial/final frames, initial conditions, parameters and behavioral metrics match. Both new arms used identical runtime sources, neural/backend configuration, native XML, sensory mapping and 27 parameters. Only the explicit event-interface configuration and its metadata/fingerprint changed. This is the no-adhesion development benchmark with operator-selected power 1.5 and steering 0.05; no fitting occurred here. [Verified comparison and hashes](comparison.json).

The event interface changed muscle transfer without fixing neural recruitment. In the common (100,280] ms window, mean raw DLM rates were 118.89/117.78 Hz left/right for the baseline and 110.00/107.78 Hz for events. Both b1 motor neurons produced zero events throughout both runs. Event timing first diverged at 63–64 ms: the 64 ms packet contains two spikes shifted by 0.5 ms, with counts still equal. A cumulative count first differs at 78 ms. Later rates therefore reflect each arm's own feedback trajectory; their difference cannot be attributed to the effector at fixed neural input.

Nearby preview samples illustrate the transfer change, but are not matched-time force measurements:

| Quantity | Baseline at 316 ms | Events at 314 ms |
|---|---:|---:|
| DLM excitation, L / R | 1.000 / 1.000 | 0.9989 / 0.9992 |
| DLM normalized force, L / R | 0.9794 / 0.9794 | 0.9788 / 0.9790 |
| DVM normalized force, L / R | 0.4204 / 0.4470 | 0.5530 / 0.6754 |
| i1 normalized force, L / R | 0.9787 / 0.9788 | 0.3863 / 0.7296 |
| b2 normalized force, L / R | 0.9792 / 0.9793 | 0.6338 / 0.5873 |
| Requested wing power, L / R | 1 / 1 | 1 / 1 |
| Body upright component | 0.5860 | 0.6310 |
| Environment contacts | 0 | 0 |

The event route retains a substantial bilateral steering contrast that the old saturated commands flattened in these samples. DLM excitation remains near maximal, and stronger DVM force does not create more requested power once that quantity has clipped to one. The body is already tilted beyond the 45° flight criterion at both nearby previews. These data establish that native forces changed; sparse previews do not establish time-averaged force improvements, the first loss-of-stability time, or a unique cause of the later crash.

At termination, the baseline has one nonfoot environment contact and angular speed 603.85 rad/s; the event arm has two nonfoot contacts and 764.85 rad/s. Neither has foot support. The event arm is at radius 6.27 cm, but this snapshot alone does not identify which surface was hit or order impact versus rotation growth. The stored flight-window RMS is not a substitute for the final instantaneous failure measurement. No root force was applied in either record.

Active execution costs normalize to 15.260 versus 15.442 seconds per simulated second, about 1.19% more for events in this pair. Both remain in the same performance range. The shorter event run used less total wall time because it ended earlier; it is not evidence of a speedup. This is one run per condition, with setup excluded and no attempt to isolate compute overhead from different contact trajectories.

The result supports retaining the optional event route as a tested implementation of the declared effector prior. It does not support promoting it as a solved flight controller or training around the remaining defects. The [frozen DLM membrane comparison](../../flight-dlm-published-model/reference/replay/run-numerical-001/README.md) and [signed-feedback audit](../../flight-sensory-observability/INTERPRETATION.md) still identify separate boundaries: motor-cell excitability and missing immediate directional information. The synthetic positive control shows that these mechanics can be stabilized when a controller actually receives suitable body-state information; BANC has not demonstrated that behavior here.
