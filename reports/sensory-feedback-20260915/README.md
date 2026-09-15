# Sensory feedback: local implementation and measured limits

The trainer now has opt-in 256 × 128 images per eye at 50 Hz, a compact directional visual front-end, and signed virtual antennal airflow feedback. Inputs pass through mapped BANC sensory cells and the existing brain–VNC–individual-MN–muscle interface. No direct corrective body-state input was added to the decoder. No weights were fitted and no cloud results were submitted in this experiment.

This is a better sensory interface, not a biologically calibrated proprioceptive system. The existing leg encoder still discards angle/velocity sign in several channels. Tegula feedback remains a load proxy. Antennal mechanics, receptor preferred directions, visual retinotopy and gains are explicit modeling assumptions. In the haltere snapshots tested here, the left organ has zero modeled motor power, which prevents its mechanical transducer from providing any added current.

## Flight comparison

All rows use the same fitted-v2 672-parameter decoder, original native body, BANC graph and neural parameters, reset seed 2590888, 500 ms warm-up, and five-second maintained-flight objective. Values below use scored time after release. Actual browser neural/physics WASM modules ran locally in Node.

| Sensory configuration | Trial duration | Total qualifying flight | Longest uninterrupted qualifying flight | First contact | Goal |
|---|---:|---:|---:|---:|---|
| Original feedback, vision off | 1.830 s | 0.574 s | 0.412 s | 1.426 s | Failed: excessive rotation |
| Vision + airflow, unfiltered v1 pilot | 5.000 s | 3.776 s | 1.400 s | 3.952 s | Not met |
| Vision + airflow, filtered v2 | 5.000 s | 4.028 s | 0.918 s | None | Not met |

The original-feedback control exactly reproduces the earlier trial's score, duration, failure reason and complete final native observation. The v2 fly ends upright, without environment contact, at approximately 2.03 cm height. Its interruptions in qualification still prevent it passing the sustained-flight goal. Score is −2.4232 / +0.5964 / +0.6944 respectively; score alone does not capture the difference between longest and total flight.

These are one-reset fixed-parameter evaluations. They do not establish generalization or distinguish the contribution of vision from antenna feedback. The antenna replacement also changes tonic drive substantially: at the matched 0.718 s input sample, the old proxy requested roughly 61.7 Hz per side, whereas the new channel means are roughly 4.8–4.9 Hz. A better outcome could partly reflect this baseline shift rather than use of richer directional information. Component and constant-input controls remain necessary before making that causal claim.

The new inputs reach real neural state: in v2, the mean actual T4/T5 firing rate varies approximately 0.010–2.504 Hz over the sampled trajectory. Requested rates and actual connected-network firing are recorded separately. These T4/T5 cells currently use the prepared spiking model, and the existing current calibration compresses positive requests below its 2 Hz knot. Detailed retinal pixels do not by themselves establish faithful physiological coding.

[Control](off-v1.evaluation.json) · [Pilot](vision-airflow-v1.evaluation.json) · [Filtered trial](vision-airflow-v2.evaluation.json) · [V2 sensory trace](vision-airflow-v2.sensory.json) · [Full v2 native motion/contact trace](vision-airflow-v2.observations.json).

## What the directional test establishes

Matched +5/−5 rad/s pitch pulses change only the mechanical haltere inputs in branched copies of real flight-conditioned neural states. All other subsequent sensory/body/internal inputs are held fixed. Spike times, individual-MN excitation histories and the actual decoder are retained; the body is not integrated in these branches. Duplicate shams and sign-folded controls must be exact.

At the 500 ms snapshot, the left/right haltere powers are `[0, 0.603522]`. At an 800 pA transducer cap, opposite pitch signs change the exact event histories of 7 or 15 wing motor neurons, depending on sensory phase. First different motor events occur at 12.5 or 18.5 ms, and decoder features/outputs differ at 13 or 19 ms. Only 2 or 1 motor neurons differ in total spike count: most of the response is timing information that rate/count averaging would hide. The decoder does not clip in these branches.

At 100 pA, selected motor events and decoder outputs remain equal within 40 ms, but afferent voltage and full recurrent neural state differ. This is subthreshold information, not proof that the signal is absent. Higher current settings recruit more receptors and produce some motor responses; they do not identify a biologically correct gain or a stabilizing control sign. No cap is selected merely for producing more spikes.

The left B1 example at 800 pA and phase π is modulation of an existing event: sham and positive pitch both contain its 523 ms spike, while negative pitch removes it. This demonstrates a right-driven network effect on a left motor neuron in this state; it does not establish newly recruited B1 activity, a monosynaptic pathway or corrective torque.

The gain sweep therefore establishes that the candidate pathway can deliver signed information through BANC into the decoder. It does not establish a calibrated bilateral reflex. Increasing the current cap cannot restore the zero-powered left haltere, and the response depends on phase and operating point. The existing [left-power audit](../flight-haltere-live/haltere-power-audit/README.md) found the motor/muscle mapping intact; previous neural observations showed a silent model motor signal. Replacing that silence with an arbitrary opposite-side fallback would not resolve the calibration question.

[500 ms final-v2 reduction](vision-airflow-v2.assay-0.analysis.json) · [Independent final-v2 timing/event audit](vision-airflow-v2.assay-0.independent-audit.json) · [Interpretation details](assay0-independent-findings.md).

The final-v2 500 ms snapshot exactly matches the pilot in both complete neural-state hashes, the native sample, decoder history, event-adapter state and delivered-current schedule. The measured dose/phase responses also match. Thus the 500 ms finding is independently reproduced under the corrected final-v2 declaration and optics; it does not depend on relabeling the pilot.

At 600 ms the final-v2 state has diverged from the unfiltered pilot. Its 100/200 pA branches show no selected event or decoder contrast in the 40 ms window. At 400 pA, 4/1 afferents change their timed events across the two phase conditions, but no wing-MN event or decoder output differs. At 800 pA, phase zero changes 18 wing-MN event histories, with first motor differences at 21 ms and decoder differences at 22 ms; phase π changes two afferents but no wing-MN events or decoder outputs in the same window. Both neural runs finish and pass final asset verification. This later-state result strengthens the calibration concern: detectable peripheral direction can fail to produce a different motor-event input to the decoder. It does not establish that no information exists elsewhere in the CNS or beyond 40 ms.

| Final-v2 state | Current cap | Wing MNs with changed timed events, phase 0 / π | First motor-event contrast, phase 0 / π |
|---|---:|---:|---:|
| 500 ms | 100 pA | 0 / 0 | — / — |
| 500 ms | 200 pA | 2 / 0 | 21.5 ms / — |
| 500 ms | 400 pA | 5 / 5 | 24.0 / 31.5 ms |
| 500 ms | 800 pA | 7 / 15 | 12.5 / 18.5 ms |
| 600 ms | 100 pA | 0 / 0 | — / — |
| 600 ms | 200 pA | 0 / 0 | — / — |
| 600 ms | 400 pA | 0 / 0 | — / — |
| 600 ms | 800 pA | 18 / 0 | 21.0 ms / — |

[600 ms final-v2 reduction](vision-airflow-v2.assay-1.analysis.json) · [Independent final-v2 audit](vision-airflow-v2.assay-1.independent-audit.json). A dash means no observed contrast within this finite window, not proof of permanent silence. These tests do not justify promoting a new haltere calibration or retraining against an assumed working reflex.

## Image quality, cost and review

Each eye uses its own 32,768 rays, rather than enlarging the old raster. V2 removes high-frequency procedural-texture aliasing using surface pixel footprints; the offline supersampled comparison shows a 62.3% reduction in RGB error at the saved release pose. This is an optical engineering check, not a biological eye validation. The renderer approximates native geometry and omits self-occlusion and shadows. [Before/after/reference](antialias-comparison.png).

The complete v2 sensory update path took 21.776 s of the 608.908 s evaluation, about 3.6% of wall time. The full BANC/physics trial remains much slower than real time. This is a measured breakdown for this run, not a controlled comparison across machines or proof of a global speedup.

The [local review](http://127.0.0.1:7884/) opens in Safari with the existing 3D preview and reconstructed eye images. It does not train or contact a coordinator. The first two active runner versions saved only their first 200 observer poses; the page explicitly labels this incomplete articulated-pose range. Full 2 ms root-motion/contact records and full trial results are preserved. The runner now decimates observer frames across the entire trial and always retains the final frame for future runs. Reconstructed eye views are labeled separately from actual stored sensory frames.

## Next fitting stage and validation boundary

The [protocol](PROTOCOL.md) walks through recovery fitting: fix and version the sensory model; gather varied recoverable tilts/descents; fit causal MN features against state-aware teacher labels; hold out whole trajectories; and test the decoder alone in closed loop. Teacher/body state supplies training labels, not deployed decoder inputs. Use failure states for further demonstrations when distribution shift dominates. Train takeoff/landing after recovery and maintained flight are reliable.

The source archives and v2 bundles preserve exact JavaScript, body XML/metadata and asset identities; prepared graph/binaries remain external hash-verified dependencies. V1 is pilot evidence because its legacy flag said vision was disabled despite actual retinal injection. V2 corrects that declaration and enables filtered optics. The final working-tree follow-up changes only eye telemetry and floating-point tolerance at the unused 100 ms cadence boundary; a synthetic 20 ms replay produces bit-identical complete afferent-input and motion buffers. Those reporting fixes do not relabel the frozen v2 source hash. [Exact source/config difference](working-tree-verification.json).

All **75 focused tests pass**, covering image motion/antialiasing, cadence, input routing/identity, antennal mechanics, sensory declarations/assets, neural-event timing, decoder snapshots, and existing episode/parameter contracts. The assay unit fixtures are small networks; the separately recorded assays above use the actual full BANC model. [Test log](final-tests.tap).

No claim here establishes takeoff, landing, food localization, feeding, calibrated electrophysiology, or broad behavioral success.
