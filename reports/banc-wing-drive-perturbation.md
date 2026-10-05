# Why the fly leaves the food

A single fly was tested in the original 3D console, using BANC v888, Reference physiology, direct motor-to-muscle coupling, the neural body clock, and an off-food initial pose. No production parameters were changed.

All four external sensory pathways (odor, taste, vision and body sense) were disabled **from worker initialization**, before the first neural step. Intrinsic graded release, recurrent connectivity and internal-state currents remained active. Inputs were then enabled and disabled again in the same fly.

| Phase | Neural time | Observed wing output and body response |
|---|---|---|
| Sensory inputs off from startup | 0–164 ms | Wing output reached approximately 207/208 Hz by 54 ms; the fly was already airborne. Wing output peaked near 290 Hz. |
| Sensory inputs on | 164–374 ms | Flight continued, ending at approximately 267/247 Hz left/right wing output. |
| Sensory inputs off again | 374–628 ms | Flight continued, ending at approximately 204/188 Hz and wing power 0.953. |

The motor-to-muscle adapter saturates at 80 Hz. With wing power near 1, the reduced flight model produces upward acceleration from lift of approximately 2400 cm/s², compared with gravity of 981 cm/s². It also applies forward thrust proportional to lift. Persistent wing drive therefore produces ascent and forward motion into the arena boundaries. It does not establish food seeking or a decision to take off after feeding.

The experiment establishes that external sensory stimulation is **not necessary** for the excessive wing drive. It narrows the remaining diagnosis to intrinsic/recurrent neural dynamics and their motor-to-muscle calibration. It does not separately identify the contribution of graded baseline release, synaptic gains, individual cell parameters or internal-state currents. Disabling external senses is a diagnostic perturbation, not a proposed fix. The behavioral target remains unmet.

Historical raw samples, initialization flags, source hashes and application errors: [banc-wing-drive-before-mujoco.json](banc-wing-drive-before-mujoco.json). This report describes the previous reduced body and conductance setting. Run a diagnostic on the current implementation with:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/diagnose-banc-wing-drive.mjs
```
