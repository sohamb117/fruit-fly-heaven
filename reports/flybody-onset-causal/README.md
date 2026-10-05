# Actual BANC launch: controlled replay

The actual first 400 ms were captured from one direct BANC fly in the original 3D UI. Complete source hashes, including the XML model, match. Both continuous full-control replay and the full MN-rate→WASM-muscle→native-body replay reproduce qpos, qvel and controls exactly; muscle-state error is also 0. No state corrections were made after the initial restore.

| Replay | First airborne | First inverted | Maximum root rise | Minimum uprightness | Maximum angular speed |
|---|---:|---:|---:|---:|---:|
| Recorded MN output | 104 ms | 112 ms | 2.232590 cm | -0.993417 | 2663.969 rad/s |
| Only DLM/DVM rates zero | None in 400 ms | None in 400 ms | -0.000639 cm | 0.883785 | 3.509 rad/s |
| DLM/DVM zero + six wing controls zero | None in 400 ms | None in 400 ms | -0.000639 cm | 0.883748 | 3.509 rad/s |

The table uses the recorder’s 2 ms sample spacing. A subsequent [50 µs mechanics audit](mechanics.md) resolves first inversion at 110.05 ms and first positive wing impact at 112.70 ms. Removing only wing–environment contact leaves the initial tumble unchanged but reduces peak rotation from 2720.56 to 154.11 rad/s and maximum upward rise from 2.233 to 0.032 cm. It separates the initial powered tumble from the later contact-amplified motion; no diagnostic contact removal was applied to production.

The recorded DLM/DVM-driven wing path is necessary for this mechanical launch: removing those rates prevents loss of support and inversion over the captured horizon. Removing the remaining active wing servo changes little in that counterfactual. The zero-power restoring servo alone does not explain this onset. All three cases have zero externally applied forces.

This does **not** identify whether the recorded wing command, the muscle/hinge surrogate, environmental wing contacts or their combination is biologically correct. The extreme peak body angular velocity warrants a separate mechanical check. Counterfactual body motion would change sensory input in a live brain; this assay deliberately holds the recorded MN inputs fixed. It is not a recovered closed-loop behavior, and no flight-muscle silencing was applied to production.

The 200 samples here are 2 ms physical input/state records over 0.4 simulated seconds; they are separate from the 200 screenshots over 60 real minutes in the [visual audit](../observation-60min-20260913/README.md). The browser capture took 30.03 wall seconds.

[Capture](capture.json) · [Replay and source checks](replay.json) · [Capture-end screenshot](onset-end.png) · [Protocol](../flybody-onset-causal-protocol.md)

![Continuous physical replay: root height and uprightness](replay-timeline.png)

Rebuild this figure with `uv run --offline scripts/plot-flybody-onset.py`.

The [contact geometry classification](contact-geometry.md) confirms later contacts in invisible banana fill and at a native cliff 6.43 mm above the visible bowl. The largest sampled normal force also hits the real bowl, so contact amplification alone is not a defect. The first overturn precedes these impacts; fixing the terrain alone does not establish recovered control.
