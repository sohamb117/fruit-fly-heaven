The saved stable event control does **not** lose motion qualification to the instantaneous `COM verticalSpeed >= -1 cm/s` gate. None of its 750 saved 2 ms samples cross the threshold. A 50 ms velocity average changes neither the result nor its longest continuous motion-qualified interval: 1.452 s, covering 726 qualifying samples from 50 ms through 1.5 s.

| Saved arm | Instantaneous samples below -1 cm/s | 50 ms means below -1 cm/s | Instantaneous velocity range (cm/s) | Mean velocity range (cm/s) | Longest motion-qualified interval, instant / mean |
| --- | --- | --- | --- | --- | --- |
| `historical_native` | 0 / 750 | 0 / 726 | -0.802 to 0.536 | -0.291 to 0.149 | 1.452 / 1.452 s |
| `causal_native` | 0 / 750 | 0 / 726 | -0.806 to 0.533 | -0.295 to 0.149 | 1.452 / 1.452 s |
| `event_closed_loop` | 0 / 750 | 0 / 726 | -0.765 to 1.325 | -0.176 to 0.955 | 1.452 / 1.452 s |
| `event_open_loop` | 175 / 750 (23.33%) | 168 / 726 (23.14%) | -6.200 to 1.356 | -4.343 to 0.830 | 0.524 / 0.948 s |

The two percentage denominators differ because a complete average first exists at 50 ms. On the same 726 complete-window samples, the open-loop instantaneous failure fraction is 175 / 726 = 24.10%. All stable arms remain at zero under either denominator.

For `event_closed_loop`, the 50 ms inferred support is 0.974–1.027 body weights, and the 20 ms RMS of saved raw angular speed is 3.147–9.237 rad/s. All complete windows pass the current support, attitude and angular-rate gates. The source records zero contacts across every native step, no root writes or applied external forces after release, maximum tilt 7.198 degrees, COM rise 0.833 cm, and horizontal drift 2.168 cm. This was stable by that assay's declared criterion; it was not precise stationary hover.

The failed open-loop control does exhibit short dips that averaging suppresses: 41 samples fail the instantaneous velocity gate while the mean and other reconstructed motion gates pass. Its first such dip is at 160 ms, with instantaneous velocity -1.105 cm/s and 50 ms mean -0.0779 cm/s. Smoothing increases its longest interval to 0.948 s but does not establish stable flight; this arm reaches 43.519 degrees of tilt and fails the original positive-control criterion. Attribution of those dips specifically to wingbeat recoil would need a separate phase-resolved analysis. They are not evidence that the successful control is rejected.

This is a read-only motion-gate audit, not a full current-scorer replay. The trajectory is a 1.5 s reduced-body assay with fixed nonwing joints, prepared trim/event history, synthetic events and an explicit frozen classical controller; BANC is not running. It begins near 10 cm altitude and does not supply the present task's ceiling or complete observation record. Saved native muscle power remains above 0.75, but actual per-sample realized `wingPower` and total COM speed are absent. No 5 s task success, habitat compliance or takeoff credit is asserted.

The comparison averages linearly interpolated saved COM velocities with trapezoid integration over 25 completed 2 ms intervals. Angular RMS uses the saved raw angular-speed magnitude over ten samples; inferred support uses the same 50 ms endpoint-velocity formula as the current scorer. These are 2 ms samples, not all 50 μs native velocities, and cannot exclude unsaved between-sample excursions. Qualification credit follows the scorer convention of 2 ms for each qualifying endpoint, hence 726 × 2 ms = 1.452 s.

Evidence: [saved source assay](../flight-event-control/run-001/README.md), [analysis script](analyze.mjs), [per-sample calculations and pinned hashes](result.json). Execute `node reports/flight-event-control-scoring-audit/analyze.mjs` to prepare the calculation; it refuses overwriting an existing result. No criterion changes, native runs, or live-source edits were made. This record provides no basis for changing the velocity threshold to rescue the successful positive control.
