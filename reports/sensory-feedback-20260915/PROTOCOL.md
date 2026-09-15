# Local sensory feedback and directional identification

This work changes sensory transduction, then tests whether a signed disturbance reaches the existing motor decoder. The biological graph, neural parameters, 672 fitted-v2 decoder coefficients, native body, muscles, contacts and flight score stay fixed. These are evaluations, not a new learning run. The diagnostic host allows local asset reads only and never contacts the coordinator.

## Implemented sensory path

Each eye now supplies a genuine 256 × 128 raster, 64 times the old 32 × 16 pixel count. Images are rendered every 20 ms of native simulation time, independently of the preview and wall clock. Each pixel casts its own ray through an analytic approximation of the native bowl, floor and fruit geometry. This is not a MuJoCo render or a calibrated compound eye: silhouettes differ from collision meshes, and self-occlusion, wings and shadows are absent. Native head joints are frozen in this experiment.

A compact, engineered, full-resolution motion front-end retains local horizontal and vertical direction, ON/OFF changes and expansion. Its sole neural injection boundary is 8,810 individually mapped T4/T5 cells. L1–L3 do not receive the same image a second time. Cell identities come from the prepared BANC mapping; representative-point retinotopy, subtype preferred directions, optics and gains remain explicit hypotheses. Adding pixels does not identify those biological unknowns.

The final v2 optics analytically filter surface texture according to each pixel's surface footprint, including grazing-angle stretch. This removes moiré without additional rays. It does not antialias geometric silhouettes. The initial v1 pilot omitted this filtering and retained a legacy `config.vision:false` declaration despite actually delivering retinal input; v2 corrects both and has a new fingerprint. V1 and v2 evidence must remain labeled separately.

The antenna path replaces the former unsigned speed/tilt proxy with passive virtual bending driven by signed air velocity in the native root frame. Configured wind must match native physical wind. It drives 562 supported JO-C/D/E/F family cells and gives zero added input to 17 ambiguous/mixed cells. Each episode owns its bending state. Native antenna joints and actuators stay frozen; this observer has no physical force feedback, active antennal control, local rotational airflow or auditory vibration model. Its mechanical constants and per-cell preferred directions are uncalibrated priors, not anatomy recovered from BANC.

Existing leg proprioception and tegula feedback remain unchanged. This patch does not restore the directional information lost by absolute-valued leg angle/velocity encoders. The live mechanical haltere prior is investigated separately rather than silently replacing the existing flight feedback.

## Fixed-parameter flight evaluation

- Reset seed: 2590888, matching the previously failed fitted-v2 trial.
- 500 ms root-held warm-up, then up to five scored seconds of unrestrained maintained flight.
- Actual browser neural and MuJoCo WASM modules, hosted locally in Node.
- Neural steps 0.5 ms; body feedback blocks 2 ms; event-driven decoder/muscle updates 1 ms. Physics scheduling is unchanged.
- Record native motion, contact, individual motor-event behavior, rendered eyes, requested sensory rates, applied current and actual sensory neural activity.
- Disabled feedback must reproduce the historical return, failure time/reason and complete final observation exactly. The off-v1 run passed this check.
- Both-feedback trials are paired against that control. A single successful seed is not evidence of generalization, and cannot attribute improvement separately to vision and antenna feedback. Separate component bundles are available for subsequent ablations.

The unchanged control scored −2.4232 and failed at 1.830 scored seconds, with 0.412 s of best qualifying flight. All parameters remained unchanged. Final sensory-on results belong in the accompanying README after completion.

## Directional haltere assay

Capture the real neural state at native times 500 and 600 ms, with no environment contact or applied external force. Copy both neural buffers, parameters, synaptic history/kinetics, events, exact motor-event adapter state and full causal decoder history. Each branch starts from the same retained state; source disposal is allowed.

At each state, test current caps of 100, 200, 400 and 800 pA, and sensory phase offsets of 0 and π. Each combination has six 40 ms branches: duplicate shams; +5 and −5 rad/s pitch-rate pulses for 10 ms followed by 30 ms recovery; and a pair of deliberately sign-folded controls. Other angular components stay fixed. The sham recomputes the mechanical haltere prior at captured native angular velocity; it is not an unmodified live continuation.

Only the 328 annotated haltere inputs change. Other sensory inputs repeat the last delivered 2 ms current schedule; internal state and body are frozen, while the prescribed wing phase advances. Preserve raw afferent and individual motor-neuron events at 0.5 ms resolution, 2 ms event packets, and 1 ms decoder features/outputs. Verify requested current against what the native WASM actually consumes. Duplicate shams and sign-folded controls must match exactly.

Interpretation proceeds through current → afferent voltage/release/events → individual motor events → decoder features → decoder outputs. Spike timing changes count even when aggregate event counts match. A graded neuron need not spike to transmit a signal. A null finite-window contrast does not prove information-theoretic absence.

The pulse replaces native pitch rate with ±5 rad/s rather than perturbing the native value symmetrically. Therefore the reported even response also includes the shift of operating point; it is not a pure measure of nonlinearity. Current approaching a transducer cap is not equivalent to neural saturation. More afferent spikes alone cannot select a calibrated gain. These branches do not integrate the body and cannot establish that an output would correct pitch or sustain flight.

## Step 4: fit recovery behavior

1. Freeze and version the sensory model first. Select transducer gains using stimulus-to-neural-response evidence, with anatomy, assumed preferred directions and fitted values kept distinct. Resolve missing or one-sided responses before increasing decoder size.
2. Generate varied, physically recoverable trajectories: both pitch and roll signs, upward/downward velocity, different initial heights, and bounded disturbances. Split whole trajectories and initial-condition families into training, validation and untouched tests before fitting.
3. A state-aware teacher may produce recovery demonstrations or label states visited by the student. BANC still receives only the sensory loop. Record aligned individual-MN events and the teacher's physical control targets. Teacher actions are one feasible set of labels, not measured muscle physiology or the unique solution.
4. Fit the existing bounded decoder on its causal MN features with regularization. Choose regularization on validation trajectories; compare against nominal constant/phase controls and shuffled-MN diagnostics. Never give body pose, reward, teacher actions or future neural samples to the deployed decoder.
5. Evaluate the fitted decoder alone in closed loop on held-out resets. Measure recovery time, attitude error, altitude loss, contact, termination and qualifying airtime. Offline imitation error cannot establish control. Feedback ablations test whether the fly actually uses the new sensory information.
6. Collect student failures and relabel those states if distribution shift dominates. If opposite disturbances still do not produce usable motor information, investigate sensory-to-neural gains or dynamics rather than expecting decoder optimization to invent that information.
7. Once recovery and maintained flight pass reliably, progress to takeoff and landing. Food localization, approach and feeding remain the larger target, beyond these airborne tests.

FlyBody's published results establish that this physical model can support learned locomotion and visual control, not that the present BANC inputs or decoder already suffice. See the [primary FlyBody paper](https://www.nature.com/articles/s41586-025-09029-4).

## Reproduction and provenance

`v1-source-archive/` retains exact runtime JavaScript plus the original native body XML/metadata. `v2/` owns the corrected runtime sources; binaries and prepared graph/data remain external hash-verified dependencies. The `scripts/flight-feedback-runtime.mjs` loader uses these owned sources, so future working-tree edits cannot silently alter archived experiments. Full source/config/binary identities accompany each result.

```sh
node scripts/evaluate-flight-feedback.mjs --variant=vision-airflow --name=repeat-v2 --bundle=reports/sensory-feedback-20260915/v2/vision-airflow.bundle.json --capture=true
node scripts/summarize-flight-feedback.mjs reports/sensory-feedback-20260915/repeat-v2.assay-0.json reports/sensory-feedback-20260915/repeat-v2.assay-1.json
```

Use a fresh result name: the runner refuses to overwrite completed evidence. The final implementation is local; no cloud checkpoint, production page or coordinator evaluation is modified by this protocol.
