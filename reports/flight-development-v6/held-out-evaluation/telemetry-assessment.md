The selected `g3-p0-neg` (`training-best`) improves recorded qualified airborne duration on all three v6 held-out seeds. It does not achieve sustained-flight or landing success: all nine episodes terminate with `excessive_rotation`, and all record zero landing duration.

This assessment uses only run `58ad2c1a-ba0b-4792-b31e-e80ef04489b9` and its nine existing records. [The companion extraction](telemetry-assessment.json) preserves full-precision sampled values, record SHA256 hashes, criteria and provenance. Times below identify the recorded 2 ms body boundaries; native floating-point timestamps and unrounded values remain in that extraction. No simulation or source change was made. Input record hashes were checked unchanged after extraction.

| Head | Longest qualified bouts, seeds 490888 / 590888 / 690888 (s) | Mean (s) | Confirmed takeoffs | Termination times, same seed order (s) |
|---|---|---:|---:|---|
| g0 | 0.136 / 0.082 / 0.172 | 0.130 | 1/3 | 0.402 / 0.416 / 0.392 |
| g4 | 0.112 / 0.110 / 0.110 | 0.110667 | 0/3 | 0.318 / 0.326 / 0.314 |
| training-best | 0.338 / 0.354 / 0.352 | 0.348 | 3/3 | 0.686 / 1.106 / 0.720 |

All nine records are complete and error-free, with identical config, model, plan and source hashes. For each seed, initial physical observations and native root `qpos/qvel/qacc` are exactly equal across heads; parameter-dependent wing frequency differs. The recorded backend is Dawn Metal neural compute with a MuJoCo WASM body, in the v6 no-adhesion benchmark.

The selected candidate's early airborne samples support the improvement within the modeled flight criteria:

| Seed | Time (s) | Current qualified bout (s) | Tilt (degrees) | Angular RMS (rad/s) | COM vertical speed (cm/s) | Inferred support (body weights) |
|---|---:|---:|---:|---:|---:|---:|
| 490888 | 0.282 | 0.166 | 15.320346 | 13.068978 | +8.464039 | 1.056813 |
| 590888 | 0.280 | 0.160 | 13.982940 | 12.952008 | +9.095365 | 1.044529 |
| 690888 | 0.282 | 0.164 | 16.948668 | 12.156887 | +8.624868 | 1.040626 |

Each sample has zero environment contacts, essentially unit wing power, and a full 50 ms inferred-support window. Recorded qualification requires contact-free flight, tilt at most 45 degrees, 20 ms angular RMS at most 20 rad/s, vertical speed at least −1 cm/s, support at least 0.8 body weights and wing power above 0.1. Every saved observation has `externalForce=false`; every saved native claw adhesion gain is zero. These are powered, supported intervals under the recorded criteria, rather than evidence based solely on altitude. In contrast, g4 at 0.282 s is higher (4.409–4.524 cm COM height), but tilted 98.379–104.208 degrees, descending 12.944–17.349 cm/s, with support only 0.267900–0.331188 body weights. Inferred support is not a direct aerodynamic force measurement.

The end of the longest qualified bout is not captured directly in any selected record:

- **490888:** At 0.584 s, current qualification is zero, the retained best is 0.338 s and phase is `unqualified_airborne`. Rejected touchdowns increased from zero at 0.282 s to one. The current snapshot is contact-free and passes the displayed tilt/RMS/vertical-speed/support gates (27.765346 degrees, 15.789345 rad/s, +5.089954 cm/s, 1.073662 body weights). This records lost qualification and an intervening contact rejection; it does not establish whether that contact caused the first break or followed another failed gate.
- **590888:** At 0.582 s, current qualification is only 0.002 s while the retained best is 0.354 s: a longer bout ended and qualification restarted. Rejected touchdowns remain one and departures remain two from 0.280 s. The first failed gate is unrecorded. At **0.880 s**, vertical speed is **−2.609515863 cm/s**, below the −1 criterion, so descent explicitly explains nonqualification at that sample. Tilt, RMS and support then pass (41.757259 degrees, 13.709314 rad/s, 1.094235 body weights).
- **690888:** At 0.584 s, current qualification is 0.010 s versus a retained best of 0.352 s, again showing interruption and requalification. Rejected touchdowns remain zero. The sampled RMS is 17.210296 rad/s and support 1.099382 body weights; the first failed gate and its exact time are unrecorded.

Selected terminal angular speeds are 315.944049, 1096.429741 and 741.651312 rad/s at 0.686, 1.106 and 0.720 s respectively, exceeding the recorded 300 rad/s failure threshold. Their terminal COM radii are 6.135183, 5.268885 and 6.105433 cm. All three terminal samples show zero current contacts; rejected-touchdown totals are 2, 2 and 0. Other records directly show terminal non-foot contacts: g0/690888 has ten; g4/490888 and g4/690888 have four each. Contact interactions therefore remain a plausible contributor to some catastrophic rotation, especially near the habitat boundary. Neither these samples nor the counters resolve the exact contact surface, impulse timing or causal ordering. A contact-free terminal sample does not exclude a brief native contact between observations.

Only 3–5 preview frames exist per episode. The per-body-step physics digest is a hash, not recoverable trajectory data. No exact first qualification-loss time or gate is inferred across frame gaps, and terminal rolling diagnostics are not assumed to include the fatal observation. Saved selected wing power remains essentially one, so the frames show no withdrawal of power; they cannot establish the intervening force history. The supported conclusion is a longer modeled qualified interval for one selected parameter set on three matched seeds. It does not establish one second of continuous controlled flight, successful landing, biological fidelity, or a causal benefit from closed-loop sensory feedback. This assessment contains no v7 evidence.
