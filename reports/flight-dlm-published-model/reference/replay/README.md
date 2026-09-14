# Frozen DLM conductance replay

The original strict replay failed on voltage differences of only 1–2 ULP; every other state and event was bit exact. That failed [run-001](run-001/result.json), original runner and original plan are preserved. A separately pinned [numerical protocol](plan-numerical.json) then completed once, accepting at most four ULP **and** 0.0001 mV voltage error while keeping all other fields and events bit exact. Its legacy trace exactly equals the first attempt. See the [result report](run-numerical-001/README.md) and [precision audit](failure-analysis-001.md); compiler cause remains unproven.

This isolates the effect of replacing the generic motor membrane model while holding its observed synaptic input fixed. The [capture](../../../flight-sensory-families/dlm-conductance/result.json) passed all 600 native half-millisecond steps, 150 comparisons with the historical 2 ms records, and the initial/final full forward-state hash checks. The [ionic reference](../comparison-001.json) separately passed comparison with the actual archived author Brian2 implementation on the declared modern environment.

The input is a 300 ms **open-loop BANC neural experiment with fixed external currents**. It is not the current closed-loop flight trajectory. Receptor conductances were calculated by the existing BANC model; they are not physiological measurements. The ten captured neurons are the two annotated five-cell DLM groups. Their prepared graph has no incoming electrical edges, so this comparison adds none. The published five-cell coupling result does not authorize inventing BANC pair identities or silently adding gap junctions.

## Gates and execution

The completed numerical run used the command below. Choose a new directory if explicitly repeating it; all source, input and prior-failure hashes remain pinned in `plan-numerical.json`.

```sh
node reports/flight-dlm-published-model/reference/replay/replay-numerical.mjs reports/flight-dlm-published-model/reference/replay/plan-numerical.json reports/flight-dlm-published-model/reference/replay/run-numerical-001
```

The following commands describe the retained original **strict** protocol. They are not the acceptance rule for the completed numerical comparison.

Run from the repository root. Verification reads files and computes descriptive statistics only; it does not initialize Metal, execute a neuron model, or install packages:

```sh
node reports/flight-dlm-published-model/reference/replay/replay.mjs --verify-only reports/flight-dlm-published-model/reference/replay/plan.json
```

To execute after operator review, choose a new output directory:

```sh
node reports/flight-dlm-published-model/reference/replay/replay.mjs reports/flight-dlm-published-model/reference/replay/plan.json reports/flight-dlm-published-model/reference/replay/run-001
```

The runner checks pinned capture bytes, archived source, completed reference comparison, parameters, receptor definitions, capture continuity, and event/count consistency. It requires precisely the recorded native backend provenance: `webgpu@0.6.0`, Node 23.7.0, the same Metal adapter/driver and native artifacts. There is no backend fallback.

It then replays the original generic membrane recurrence on Dawn using captured post-kinetics conductances, raw external inputs and internal-state coefficients. The shader preserves the original membrane arithmetic and float32 ordering. State propagates from the initial condition; it is never reset to the captured state between ticks. Every state8 field and last-spike time must match **bit for bit** for all ten neurons at all 600 steps. Cumulative counts, instantaneous release and last-spike fields jointly gate events. Any mismatch saves the mismatches and replay trace, marks the run incomplete, and prevents ionic execution. Passing verifies this isolated membrane replay; it does not independently regenerate receptor gates or the full recurrent graph.

Only after that gate does the runner execute two 300 ms ionic arms: all captured chemical conductances, and a zero-input control. Each uses the already compared `reference.mjs` derivative, RK4 at 0.1 ms, captured initial V, and the declared author-reference initial gates h=b=0.146. This gate initialization is a modeling assumption, not a measured BANC state. The five effective conductances are held over each original 0.5 ms interval, with each current recomputed as g(E−V) at every RK4 substage. No interpolation, fitted scaling, extra adaptation, voltage reset, phase reset, tonic current, or gap current is introduced. Source 108.75 pA is **not** added.

The native state7 is never used as the ionic stimulus: it includes negative legacy adaptation and excludes chemical driving current. The captured external, tonic, and hormonal currents are independently verified zero before adaptation. The recorded monoamine gates are retained in the input record, but their existing model only scales that zero external drive for these cells; no new ionic neuromodulator mechanism is invented.

## Interpreting output

`result.json` retains the plan, hashes, exact-gate results, source/backend provenance, per-neuron conductance statistics and both ionic summaries. `legacy.f32` contains all half-ms replay states and spike times; each ionic `.f64le` contains all ten V,h,b trajectories at 0.1 ms resolution. Output paths must be new, and failed runs remain intact.

Each ionic summary reports both the source's threshold detector and raw upward crossings of −10 mV. The source has a 10 ms event-detection guard without a membrane reset. That detector can suppress counted events during strong input, so its apparent 100 Hz ceiling is not a biological prediction. Raw crossings, voltage/gate ranges and the fraction of time above threshold help distinguish repeated spikes, sustained depolarization and numerical failure. Brian's pre-step clock label is retained beside the post-step time. No sub-tick crossing is inferred. The fixed reporting window is (100,280] ms, only a short transient.

Comparing the two membrane implementations under identical frozen g(t) can reveal whether the generic membrane profile materially changes recruitment. It cannot establish physiological synaptic calibration, prove a firing-rate fit, or predict the response of the reconnected recurrent circuit: an altered DLM would change its network output and potentially its future input. Strong drive and intrinsic excitability may both matter. The published ionic model is a reference hypothesis for these DLM cells, not an established profile for DVM or synchronous steering neurons.

The derivative retains the [Hürkey et al. 2023](https://doi.org/10.1038/s41586-023-06099-0) / [Zenodo 7740678](https://zenodo.org/records/7740678) attribution and the author code's CC-BY-NC-4.0 license, documented in the [parent README](../README.md). This isolated extension and its held-conductance discretization are new diagnostic code; numerical agreement of the original reference does not by itself validate the extension.
