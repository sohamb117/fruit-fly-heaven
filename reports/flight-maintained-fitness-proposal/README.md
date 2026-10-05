# Durable maintained-flight fitness proposal

Status: report only. No canonical source, configuration, database, running job or reserved held-out record is changed or evaluated. The current two-generation experiment and its held-out evaluation remain the predeclared experiment. Consider this proposal only afterward if their outcome warrants a new experiment.

## Recommended minimal change

Keep every current qualification, success, physical-termination and observation-validity gate. Let **T** be the existing `totalQualifiedFlightSeconds` counter during the five scored seconds. It increases only for powered, contact-free flight meeting the attitude, angular-rate, support and mean-vertical-speed criteria. Set earned progress to **P = T**, in score points per qualified second.

| Outcome | Proposed return |
| --- | --- |
| Still running | T |
| Five-second timeout without hard success | T |
| Existing physical failure | T - 1 |
| Existing hard success at the full horizon | T + 5 |
| Invalid observation or unexpected external force | -10 |

Apply the bonus or penalty once. Do not sum the absolute running score over steps: the incremental progress is the newly earned qualified duration, not the previously accumulated total again. The proposed terminal physical penalty is the existing one-second minimum continuous-flight duration, expressed in score units. The success bonus equals the five-second horizon. These are explicit utility choices, not fitted or measured physiological quantities.

The maximum earned progress is five. Every success has at least one qualified second and therefore scores at least six, above every nonsuccess. Normal physical failures range from -1 to 4; invalid/external-force failures remain -10; the existing [-10, 10] score bounds suffice. Qualification begins only after fresh post-release support windows, so attainable maximum progress can be slightly below five.

This replaces two separate defects: the timeout's use of only the final bout, and the three-point physical-failure cost that can exceed meaningful earned flight. A three-second qualified episode followed by a physical failure scores **2**; an episode that never qualifies and rests on the floor scores **0**. A late gate crossing stops new credit and resets the current bout exactly as before, but cannot erase the previous three seconds. A crashing episode with less than one qualified second can still rank below grounded zero; that is the explicit one-second failure cost, not a hidden survival or posture reward.

## Continuity remains required for success

Keep the current final-bout requirement: at five scored seconds the fly must currently qualify and have a continuous qualified bout of at least one second. No early success is introduced. Existing failures still stop the episode immediately; no new integration after a crash is requested. The warmup remains unscored, and the initial airborne placement receives neither takeoff nor landing credit.

The primary proposal deliberately gives equal earned credit to equal amounts of qualified airtime. Several short qualified bouts can earn partial progress, while still failing the hard sustained-flight criterion. If later evidence shows that this fragments flight excessively, an explicitly optional alternative is **P = 0.8 T + 0.2 B**, where B is the existing best continuous qualified bout. It remains durable, monotone in both counters and bounded by five, with the same terminal bonus/penalty. Do not introduce or fit this blend during the current experiment. The primary recommendation is the simpler T-only rule.

## Why this is denser, and what it does not solve

The current score can drop from several earned points to zero when a single two-millisecond sample ends a bout. The proposal instead earns 0.002 points for that qualified step, or zero for an unqualified step; previously earned progress remains. This removes the discontinuous loss of history. It is dense in *qualified duration*, not differentiable through the physical qualification thresholds. The hard success bonus and physical-failure penalty remain deliberate discontinuities.

Upright posture, time spent on the floor, unqualified falling, warmup time and mere survival earn no positive progress. Nevertheless, this reward cannot repair a missing feedback pathway, weak steering authority, a phase-sensitive release or thresholds that classify physically useful flight as unqualified. It also cannot prove five entirely qualified seconds: the preserved success definition is the full horizon plus a qualified final continuous second.

Retaining the old three-point physical penalty with T-only progress would require more than three qualified seconds before a crashed episode beats grounded zero. The proposed one-point penalty instead makes this tradeoff visible as one qualified second. Keeping a larger progress range and the old +3 success bonus would let some nonsuccesses outrank hard successes; the 0-to-5 progress range and +5 success bonus avoid that additional ambiguity.

## Saved-record comparison

`aggregate-replay.json` contains only already completed contributor artifacts from the current training run. It excludes all reserved validation/test seeds. It pins each artifact, the current scorer and configuration, reconstructs the old terminal return, and applies the primary and optional formulas to the stored full-resolution counters. The primary rule above is specified before that comparison; the comparison is not a weight search.

These aggregate counters are sufficient to calculate the proposed **terminal returns for the existing trajectories** because no qualification or termination rule changes. They are insufficient to reconstruct every dense reward increment, identify the precise cause of each gate crossing, or predict the behavior of candidates that a new optimizer would choose. Sparse previews are not substituted for the counters. No new body or neural state is computed, and no held-out result is used to select a weight, penalty or candidate.

## Minimal conditional next experiment

If the current two generations and held-out evaluation fail to establish useful improvement, first review the saved old/new rankings against qualified duration, continuous bouts, contacts and survival. If the proposed ordering matches the stated objective, implement and test only the scoring contract in a fresh configuration/checkpoint namespace. Required pure regressions are: earned progress survives a late gate dip/contact; no-qualified-flight timeout stays zero; a three-second qualified crash beats grounded zero; physical penalties apply once; every hard success outranks nonsuccess; invalid/external force remains -10; warmup/takeoff credit stays zero; all existing hard gates and five-second timing remain unchanged.

Then run at most **one guarded generation** from the exact declared reference vector, with the same body, brain, sensory mapping, phase initialization, 27-coordinate bounds/scales and backend. Predeclare fresh, nonreserved training seeds and allow the same eight search plus six matched acceptance evaluations. Report both old and proposed returns from every trajectory, all acceptance-pair differences, total/best/current flight and hard success. Keep a retained incumbent if the existing guard rejects the proposal. One short search is an experiment in learnability, not sufficient evidence of a trained or physiologically validated fly; keep independent held-out validation separate. No such run or implementation has been prepared here.
