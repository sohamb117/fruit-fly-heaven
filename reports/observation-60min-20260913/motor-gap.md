# Recorded BANC motor-output diagnosis

Analyzed frames 4, 5, 6, 7, 8, 9, 10, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76; cutoff 76. Read-only offline analysis. Exact input hashes and per-frame values are in motor-gap.json.

The fly is producing substantial motor output. The low displayed wing power is downstream of MN output, and strong pumping is separate from a correctly placed, extended proboscis. These snapshots do not establish that the required behavior has been solved.

Excitation below means clamp(mean named-group MN rate / 80 Hz, 0, 1). It is the production decoder input, not measured muscle force. Rates are already smoothed over 50 ms; snapshots are separated by roughly 18 wall seconds and 0.3 simulated seconds. No claim about missing fast rhythms follows from this sample cadence.

## Proboscis and pumping

| Excitation channel | Old feeding frames 8–10 mean | Corrected after 400 ms mean |
|---|---:|---:|
| rostrumExtend | 0.002 | 0.097 |
| rostrumRetract | 0.000 | 0.000 |
| haustellumExtend | 0.000 | 0.000 |
| haustellumRetract | 0.000 | 0.000 |
| labellarExtend | 0.000 | 0.000 |
| labellarAbduct | 0.462 | 0.606 |

Old pooled reach contributions: proboscis_m8_muscle 60.2%, proboscis_m7_muscle 39.6%, proboscis_m9_muscle 0.2%. The present decoder keeps m7 as labellar abduction and m8 unsupported; neither extends rostrum/haustellum. Old displayed proboscis mean 0.090 therefore did not prove correctly routed reaching.

Observed pump mean: 0.484 → 0.526; corrected mouth-contact snapshots: 0/17. The three active pump groups are listed by name and rate in the JSON; five groups share one pump average.

## Wing outputs

| Side | DLM excitation mean | DVM mean | III1 mean | B1 mean | Opening proxy mean | Below 0.85 proxy threshold |
|---|---:|---:|---:|---:|---:|---:|
| left | 0.967 | 0.347 | 0.336 | 0.000 | 0.664 | 15/17 |
| right | 0.976 | 0.256 | 0.416 | 0.007 | 0.591 | 15/17 |

Corrected observed wingPower mean 0.014. The hinge uses 1 - clamp(III1 force - B1 force) for opening, then requires deployment >0.85 before beating. High power excitation can thus coexist with suppressed wing motion. Actual force/fatigue/deployment were not recorded, so these numbers identify a candidate gap rather than reproduce exact physical output.

## Opposing leg outputs

| Joint | Positive excitation mean | Negative mean | Signed difference | Coactivation mean | Both >0.5 snapshots |
|---|---:|---:|---:|---:|---:|
| coxa_T1_left | 0.257 | 0.000 | 0.257 | 0.000 | 0/17 |
| coxa_T1_right | 0.599 | 0.002 | 0.597 | 0.002 | 0/17 |
| coxa_T2_left | 1.000 | 0.000 | 1.000 | 0.000 | 0/17 |
| coxa_T2_right | 0.282 | 0.002 | 0.280 | 0.002 | 0/17 |
| coxa_T3_left | 0.974 | 0.000 | 0.974 | 0.000 | 0/17 |
| coxa_T3_right | 0.441 | 0.002 | 0.439 | 0.002 | 0/17 |
| femur_T1_left | 0.016 | 0.372 | -0.356 | 0.016 | 0/17 |
| femur_T1_right | 0.000 | 1.000 | -1.000 | 0.000 | 0/17 |
| femur_T2_left | 0.015 | 0.511 | -0.496 | 0.011 | 0/17 |
| femur_T2_right | 0.000 | 0.516 | -0.516 | 0.000 | 0/17 |
| femur_T3_left | 0.053 | 0.572 | -0.519 | 0.053 | 0/17 |
| femur_T3_right | 0.758 | 0.021 | 0.738 | 0.021 | 0/17 |
| tarsus_T1_left | 0.000 | 0.000 | 0.000 | 0.000 | 0/17 |
| tarsus_T1_right | 0.000 | 0.000 | 0.000 | 0.000 | 0/17 |
| tarsus_T2_left | 0.000 | 0.000 | 0.000 | 0.000 | 0/17 |
| tarsus_T2_right | 0.000 | 0.000 | 0.000 | 0.000 | 0/17 |
| tarsus_T3_left | 0.000 | 0.000 | 0.000 | 0.000 | 0/17 |
| tarsus_T3_right | 0.000 | 0.004 | -0.004 | 0.000 | 0/17 |
| tibia_T1_left | 0.211 | 0.047 | 0.164 | 0.047 | 0/17 |
| tibia_T1_right | 0.000 | 0.835 | -0.835 | 0.000 | 0/17 |
| tibia_T2_left | 0.005 | 0.339 | -0.334 | 0.005 | 0/17 |
| tibia_T2_right | 0.012 | 1.000 | -0.988 | 0.012 | 0/17 |
| tibia_T3_left | 0.059 | 0.467 | -0.408 | 0.045 | 0/17 |
| tibia_T3_right | 0.000 | 0.282 | -0.282 | 0.000 | 0/17 |

Names of each positive/negative target, per-muscle BANC IDs, rates, and all per-frame values are retained in the JSON. These are opposition proxies; different fatigue, length and velocity terms can change actual torque.

## Taste and physiology gaps

All 532 sugar GRNs were driven from generic on-fruit contact in these recorded builds: raw annotations identify 87 labellar, 152 wing-margin, and 293 leg cells. This also stimulates the labellar sugar pathway while mouthContact is false. The association with pumping is mechanistically plausible but has not been isolated experimentally here.

Nine labellar taste-peg cells were omitted from io.sensory by its taste-bristle-only classifier. They are restored as a separate verified annotation supplement; one has null side. Across all 532 sugar cells, 23 have missing laterality and the new mapper abstains. Nerve strings are retained in the diagnosis; they are not used to invent a side assignment.

Prepared motor profile groups: 805 cells sharing one 16-parameter profile. The model uses common 80 Hz excitation saturation, 15/40 ms muscle activation kinetics and shared fatigue/force coefficients. Energy remains above the 0.1 force-limiting reserve in these frames; fuel depletion does not explain the low wing output. Numerical physiology and relative actuator strength remain priors, not calibrated BANC measurements.

Next discriminating evidence is dense synchronized recording of named MN rates, muscle activation/force, wing deployment, reach-joint controls/positions and organ contacts, after the compartment-specific taste fix. No motor gain, desired movement, or artificial forcing was changed for this diagnosis.
