# Affine transfer: two failed cold replays

The candidate failed at **262 ms** with excessive rotation, **no confirmed takeoff and zero qualified flight**. This is not a mechanical improvement. Its early contact loss is a descending departure followed by an impact, not successful launch or landing.

The accepted static calibration and the cold replay answer different questions. The combined intervention did not transfer successfully to the captured starting state and rising motor drive. The replay changes both steering trim and power mapping, so it cannot identify either component as the sole cause.

A second saved replay keeps the original power mapping and applies the same steering reference/trim. That arm also regresses: **176 ms best qualified flight, then failure at 456 ms**. Its first flight loss is an attitude excursion at 300 ms, well before its first recorded non-foot contact.

## Paired sequence through the first failure

All times below are observations on the existing 2 ms grid, not exact contact onset times.

| Interval / observation | Original transfer | Affine + power candidate |
| --- | --- | --- |
| 0–38 ms | Identical recorded observations | Identical recorded observations |
| 40 ms | First recorded trajectory difference | First recorded trajectory difference; both reported wing-power pairs are still zero |
| 54–60 ms | Left wing power first observed nonzero at 54 ms; L/R reach 0.526/0.466 at 60 ms | First observed nonzero at 56 ms; L/R reach 0.410/0.397 at 60 ms |
| First departures | No environment contact at 66–68 ms; feet briefly return at 70 ms; continuously contact-free on the sample grid from 72 ms | Brief contact-free samples at 88 and 92 ms, interspersed with loaded feet; still on one loaded foot at 104 ms |
| 100–106 ms | At 100 ms: COM 1.812 cm, upward velocity +1.394 cm/s, up 0.975 | At 100 ms: COM 1.728 cm, downward velocity −4.372 cm/s, up 0.757. Tilt exceeds 45° at 104 ms while a foot remains loaded. At 106 ms all environment contacts disappear, with velocity −6.339 cm/s |
| 106–150 ms | COM rises 0.198 cm; mean vertical acceleration +96.34 cm/s² | COM falls 0.568 cm; mean vertical acceleration −238.70 cm/s²; every sampled observation is contact-free |
| 152–160 ms | Continues rising without sampled contact | Still descending at −15.45 cm/s at 152 ms. Feet contact at 154 ms; upward foot load peaks at **12.45 body weights** at 156 ms. Velocity turns upward by 158 ms. This is an impact and rebound |
| 194 ms | No environment contact | First recorded non-foot environment contact: 6 non-foot contacts among 18 total. Angular speed rises from 74.23 at 192 ms to 197.51 rad/s. The record does not identify the contacting body part |
| 228–260 ms | Continues supported airborne ascent | Short sampled contact-free interval 228–244 ms; feet return 246–254 ms; another departure at 256 ms. Up falls to 0.620 by 260 ms |
| 262 ms | COM 2.920 cm; up 0.998; angular speed 4.53 rad/s; **142 ms qualified flight so far** | COM 1.275 cm; angular speed **398.85 rad/s**, up 0.856; no sampled environment contact. Physical failure ends scoring |

By 500 ms the original transfer has a confirmed takeoff and 380 ms continuous qualified flight in this captured prefix. Its original objective horizon remains 8 seconds; this is not a complete successful flight/landing evaluation. Candidate physics continued to the capture boundary only for diagnostics; score stayed frozen at the 262 ms failure.

## What the saved evidence establishes

The original and candidate use the same initial observation, all 805 recorded motor-rate inputs, all 48 wing-neuron event streams, raw L/R wing drives and oscillator phases. **All 28 native wing-muscle activation, fatigue and force values match exactly at all 250 checkpoints.** There are no external applied forces or pose corrections. Pure rescoring reproduces every saved candidate score and the original 500 ms score exactly.

Thus the observed regression is downstream of the fixed neural/event/native-muscle signals. The candidate reduces interpreted power during startup: at 100 ms L/R are 0.652/0.611 versus 0.996/0.996 originally. It also changes deployment timing because the existing deployment predicate uses the scaled requested power. For example, evaluating that predicate on the recorded 40 ms right raw drive gives 0.01359 originally and 0.00668 with the candidate, on opposite sides of the `> 0.01` threshold. The exact within-block crossing time was not logged. This intervention therefore changes startup release as well as eventual power and trim; it is not simply a steady-flight offset.

The falling interval precedes both the heavy foot impact and the first recorded non-foot contact. Across sampled contact-free 106–150 ms, endpoint COM acceleration corresponds to average nongravity upward force of approximately **0.757 body weights**, versus **1.098** originally. This is a net force estimate, not a directly recorded aerodynamic wrench. Attitude also degrades before the impact: the candidate first exceeds 45° tilt at 104 ms. Later impacts/rebounds occur after launch has already failed to develop into sustained ascent.

All 48 affine bias/amplitude coordinates stay within [0,1] over the full 500 ms (range 0.3101–0.7702, 2,500 wing updates). That rules out coordinate extrapolation in this particular run; it does not make the resulting motion stable.

## Steering-only replay with original power

`steering-only-result.json` uses exactly the same `T` and `F0` and sets both power scales to 1. Its full 250 interpreted L/R power pairs equal the original transfer exactly, as do the frozen motor inputs, oscillator phases and all 28 native activation/fatigue/force values. The source and score checks pass independently.

| Observation | Steering-only result |
| --- | --- |
| 68–74 ms | Brief departure at 68 ms, feet again at 72 ms, then no sampled environment contacts from 74 through 454 ms |
| 124–298 ms | Accumulates 176 ms uninterrupted qualified flight; takeoff is confirmed at 272 ms and attributed to the departure at 74 ms |
| 300 ms | Qualification stops because **up=0.7041 is below cos45**. Every other instantaneous/window qualification check passes: ΩRMS=15.50 rad/s, vertical speed +6.17 cm/s, inferred support 0.8148, full 20/50 ms windows, wing power approximately 1, no environment contact. Original transfer up is 0.9603 here |
| 324 ms onward | Height peaks near 324 ms and subsequently falls. At 340 ms vertical speed is −7.40 cm/s and up=0.516; original transfer remains ascending at +6.13 cm/s with up=0.921 |
| 450–454 ms | Angular speed reaches 289.59 rad/s at 450 ms, and up becomes negative at 452 ms. No environment contact is recorded on these samples |
| 456 ms | First recorded non-foot environment contact (2 contacts), Ω=343.71 rad/s, up=−0.349; excessive rotation terminates scoring. COM radius is 6.228 cm, but the specific contact geometry is absent from this record |

At the same 456 ms endpoint, the original transfer has 336 ms continuous qualified flight and no physical failure. This arm establishes that the copied trim can degrade attitude even when power remains unchanged; it does not establish how the same trim would behave at its calibration operating point. **The static trim was measured at power approximately 0.792; the steering-only replay operates near 1.** It is not a calibration validation at that higher power, nor evidence that feedback is intrinsically incapable of stabilizing flight.

## What remains unresolved

The observer stores static-world contact counts and upward tarsal loads, not geom IDs, self contacts or full contact/actuator/aerodynamic wrenches. It samples every 2 ms from the last native 50 μs step. Accordingly:

- The 194 ms non-foot contact cannot be specifically called a wing collision from these records.
- Zero recorded contact at 260 and 262 ms does not exclude an intervening collision, and self contacts are outside this observer. The specific impulse behind the final angular-speed spike is unresolved.
- The combined intervention does not separate reduced startup power/deployment from changed steering trim. The steering-only arm isolates the trim change at the original power, which differs from the measured static calibration power.
- Frozen neural signals make this a mechanical counterfactual; it is not a fresh autonomous brain/body feedback trajectory.

No additional simulation was run for this analysis. Reproduce the saved-record comparison with:

```sh
node reports/flight-affine-replay-staging/analyze-candidate.mjs
```

`candidate-comparison.json` contains exact paired milestones, all contact-category intervals through each arm's first failure, acceleration windows, verification gates and source hashes for all three records. Inputs include `baseline-result.json` (SHA256 `94abc6ef6ca487b76ffecd500a830a44a2e8a4d0d2765c0ac4556693ceb86689`), `candidate-result.json` (`dc3a48626eab841d655182d86996f58cda18b5edb80abc3489508be64e2883f4`) and `steering-only-result.json`. The measured profile hash is `6302ba08230362ad0c2d8c2dc8b17b089c311c3db84e072bed82b530db71c39d`.
