# Saved-record controllability audit

The amplitude parameter is now observably effective, but these traces do not prove that the 27-parameter controller can stabilize flight. All five saved physics digests and every final scorer-state field replay exactly. The new 0.90 amplitude trajectory and all motor packets are also exactly equal to the old common-0.90 control. No native simulation, brain stepping or optimization was run for this audit.

All times below are scored seconds after the same 0.5-second live warm-up; all runs use seed 1290888 and the original instantaneous-vertical-speed scorer.

| Amplitude | Best qualified bout | Total qualified | First contact | First tilt >45° | First COM vz <−1 cm/s |
|---|---:|---:|---:|---:|---:|
| 0.85 common scale | 0.016 | 0.194 | 0.984 | none | 0.058 |
| 0.90 old/new exact pair | 0.602 | 1.248 | 1.838 | 1.936 | 0.652 |
| 0.91 new amplitude | 0.602 | 1.188 | 2.074 | 1.208 | 0.652 |
| 0.95 common scale | 0.498 | 0.712 | 0.548 | 0.634 | 0.590 |

Contact precedes the >45° tilt at 0.90 and 0.95. At 0.91 the tilt precedes contact by 0.866 seconds. These are temporal observations; neither collision causation nor a universal contact-first explanation follows. Increasing 0.90 to 0.91 changes the trajectory and delays first contact but does not improve the first/best qualified bout.

For both new-amplitude records, the fixed activation normalization saturates on both sides in every precontact scored sample. The new outer amplitude remains active despite that saturation. Pure decoder perturbations by ±0.1 log units (clipped to existing bounds) give maximum-across-trials target RMS changes around 0.03 rad for power and 0.13 rad for frequency. The largest current steering-coordinate response is about 0.00010 rad. This conditioning gap is a local command observation, not a torque/stability guarantee.

Two coordinates, `iii3` bias and amplitude, are exactly inert under both recorded inputs: the mapped left/right MNs and forces are present but silent (indices 113078 and 27969; zero cumulative spikes). `iii1` MNs also have zero scored spikes; their 2/3 warm-up spikes leave maximum force residuals only 4.36e−6/6.23e−6. Corresponding target RMS sensitivity is about 2.4–4.5e−11 rad. Deployment tau changes post-warm-up targets only at floating-point epsilon (~4e−16 rad RMS); this assay does not rule out its effect during warm-up. `iv2` is weak/sparse, with no scored events at 0.90 and one left event at 0.91. Other steering dimensions do change commands, at their retained small gains.

The search can adjust common power/frequency and reweight the existing force-driven steering basis. Bilaterally tied, positive coefficients cannot create an independent side-specific offset or a command from an exactly zero force. Sparse channels are not proof of anatomical absence or bad physiology. Rapid changes in 27 numeric parameters would therefore not imply 27 effective control directions.

`analyze.mjs` saves exact source hashes, MN identities/counts, precontact force ranges, all coordinate perturbations, and score/digest checks in `result.json`. The pure assay samples every fifth precontact row; each sample restores recorded phase/deployment, holds qpos and recorded muscle force constant, and runs ten 0.2 ms decoder steps. It does not replay native dynamics, continuous force history or brain feedback. No stabilizability, multi-seed generalization or learning claim is made. Run the script with an absent result file to reproduce.
