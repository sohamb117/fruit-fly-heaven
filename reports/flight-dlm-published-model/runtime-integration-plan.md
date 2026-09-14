# Proposed opt-in DLM ionic runtime

Plan only. No production source, native binary, model metadata or simulation was changed for this review. Implementation must wait until the current phase/effector experiments finish and their source pins are recorded.

## Smallest explicit model boundary

Add an optional, versioned manifest entry, for example `intrinsic_models: {schema:1, profile:"dlm-snl-2023-v1", cells:[{index,root_id},...], ionic_step_ms:0.1, initial_gates:{h:0.146,b:0.146}, event_policy:"threshold-10ms-guard"}`. The name denotes an experimental model prior, not measured physiology of these BANC cells. Include the entry in model/config fingerprints and reject unrecognized options, profiles, duplicate cells, identity mismatches or unsupported timesteps. Its absence selects the existing runtime exactly. Keep the 16-float parameter order and its `graded` field unchanged; do not overload `graded` with an ionic model ID.

For the real BANC model, resolve and validate exactly the ten entries below against the checksummed `io.motor_neurons` and the union of the `asynchronous_wing` / `dorsal_longitudinal_muscle` mappings. Fixture models can use an explicitly declared test-only selection. Do not select all motor neurons, DVM, steering muscles or similarly named cells. Both DLM5 cells are included without inventing MN1–4 pair identities or changing muscle-side assignments.

| Index | Root ID |
| ---: | --- |
| 12322 | 720575941432491145 |
| 31143 | 720575941463144336 |
| 41465 | 720575941479034563 |
| 74497 | 720575941519783640 |
| 84317 | 720575941533190337 |
| 127563 | 720575941583644382 |
| 135445 | 720575941596050112 |
| 157417 | 720575941640465439 |
| 160138 | 720575941645403000 |
| 173140 | 720575941692400603 |

Use the existing graph, edge weights, transmitter/receptor assignment and delay history. The ten cells have no incoming electrical edges in the current prepared graph; assert this for the first opt-in version. Do not add the paper's 43.5 pS coupling or its 108.75 pA operating-point current. Real input currents remain real inputs; no generic adaptation current is supplied to the ionic derivative.

## Allocation and API changes

`model.js` currently builds packed parameters of length `17*N+27`, state8 arrays, a delay-history buffer and kinetics of length `19*N`. Preserve those offsets. Only for the opt-in model:

- Append a Float32 slot table of length N after the existing parameter/receptor data: zero means generic, 1–10 selects a sparse DLM state slot. At N=175,401 this costs 701,604 bytes. It is immutable and shared with the graph/parameters on the GPU; WASM retains it in each instance's existing packed-parameter allocation. No ninth storage binding is needed.
- Append 30 private floats after the existing kinetics: `(h,b,eventGuardSubsteps)` per selected cell. The guard is an integer in 0–100, not a physiological refractory clamp. Only h/b are new ionic dynamical variables. Each instance owns its tail; reset initializes h=b=0.146 and guard=0. Full-state capture/restore/hash code must include the tail.
- Keep both native state arrays at stride 8 and `readState(...,{includeSpikeTime:true})` at stride 9. Do not place h/b in the adaptation/refractory fields. For ionic cells those two fields remain zero; voltage, counts, filtered rate and release retain their meanings. Expose optional diagnostic h/b readout separately, sharing the existing readback when requested.
- Define state6 for this profile explicitly as effective membrane conductance at the final state, including ionic leak, sodium and Shab channel conductances plus chemical inputs. State7 remains actual external/tonic/internal current, with no legacy adaptation subtraction. These are observations, not additional stimuli.

A small new `cell-models.js` helper should perform selection/validation, sparse layout construction and event-contract construction once per model. Existing `initialBuffers`, WASM allocation and GPU limit calculations should use its optional lengths. `WebGPUBrain` currently shares immutable `params` and pipeline while keeping states/history/kinetics private; that ownership must remain unchanged. `WasmBrain` shares only graph wiring, already keeping params and dynamics per instance.

## Preserve the default kernels

Keep the present `neural.wgsl` source and `_br_step` implementation as the feature-off paths. Add an opt-in shader/source and `_br_step_dlm` export with the same native ABI. Both use the existing full incoming-CSR/receptor/history update; only the ten flagged neurons enter the ionic membrane branch. `WasmBrain` chooses its native function once, and `WebGPUBrain` chooses its pipeline once, from the validated manifest. Neither choice depends on activity or reward.

`build.sh` currently exports `_br_step`, `_muscle_step` and `_joint_step`; add the new export and copy the new shader. Do not replace the muscle exports or current core while a pinned body experiment is running. If sharing generic code internally, verify default-path output parity before accepting the refactor. A shared generated source fragment is preferable to letting two generic kernels drift, but exact feature-off preservation takes precedence over that cleanup.

The existing WGSL event ring uses workgroup barriers reached by every invocation, including padding. The ionic branch must rejoin the common event/history epilogue; do not early-return around those barriers. Sparse h/b state can update in place because only its owning neuron accesses that slot and this version has no DLM electrical-state reads.

Within an ionic cell, update receptor gates once per existing 0.5 ms neural tick, then hold the five chemical conductances for five 0.1 ms RK4 substeps. Evaluate each `g*(E−V)` at each RK4 substage. Integrate V,h,b without the generic membrane clamp, reset or adaptation increment. The paper's fixed ionic constants replace the generic C/leak values only in this branch. Reuse the existing monoamine/external/hormonal expression separately; for the current ten cells that drive is zero before adaptation. Do not invent a direct monoamine effect on h or b.

Retain the validated reference's event detector as an explicit policy: updated V>−10 mV and at least 100 ionic substeps since the previous detection, with no V freeze/reset. The integer guard avoids floating-point interval drift. Its 10 ms limit is a detection convention, not a biological firing-rate ceiling; diagnostic raw upward crossings must remain observable. At most one detection can occur in a 0.5 ms graph tick. Increment count once, write `release=count_this_tick/0.5`, and update the existing rate EMA once per graph tick. Delivery through graph history remains on the existing 0.5 ms/delay grid; that discretization is unchanged and must be stated.

Nonfinite V/h/b or invalid gate states must invalidate the run visibly. Do not repair them by clipping, resetting or silently falling back to the generic cell. For GPU execution, retain an instance-local failure marker/readout checked at the next mandatory motor read; native execution can return an explicit status from the new export. This failure plumbing must not affect the default kernel.

## Event-time dependency: two validators must change together

The current `packages/banc-runtime/src/motor-events.js` rejects timestamps outside the exact 0.5 ms grid and assumes generic refractory≥2 ms. The new `web/flybody-wing-event-excitation.js` independently enforces the same grid, a 2.5 ms minimum interval, and corresponding snapshot count bounds. Updating only one would break the end-to-end route.

Recommended: pass one immutable per-unit event contract from the validated model to both constructors. Leave the current generic branch unchanged. For the ten ionic cells declare a 0.1 ms event grid and a 10 ms detector interval. Preserve the true detected substep time in last-spike readout and canonicalize its Float32 representation to a unique 0.1 ms grid point; do not use `Number.isInteger(raw/.1)` or require decimal 0.1 to be exactly representable. Require the raw value to lie within a small declared float-rounding bound of that unique grid point and reject times where Float32 precision makes adjacent grid points ambiguous. Compare interval limits using integer substep indices. Packet/read clocks stay on the exact 2 ms grid, and the count delta must still be 0 or 1.

The effector must apply that same contract to event validation, last-event state, snapshot history bounds and restored pending packets. Snapshot/contract versions must distinguish the opt-in timing. Its existing continuous integration already splits at event times, so no force kernel or biomechanics change is required. Preserve the old half-ms positive-control fixtures and add a separate mixed-clock fixture; do not rewrite their historical source pins.

If this timing change is deferred, explicitly publish DLM events at the enclosing 0.5 ms tick and retain substep crossings in diagnostics. That is an acknowledged timing approximation and a different profile version. It is not equivalent to preserving substep motor information.

## Validation gates before a whole-fly experiment

These are proposed fixed engineering gates, not completed tests. Pin them before running implementations. A failure calls for diagnosis, not progressively widening thresholds.

| Check | Proposed gate |
| --- | --- |
| Feature-off regression | Existing WASM tests and saved baseline states/events byte exact; GPU uses unchanged shader/pipeline and reproduces its saved baseline. Existing muscle/effector positive controls remain unchanged. |
| Identity/configuration | Exact ten index/root-ID pairs; no unknown model values; no hidden tonic/gap input; params/CSR/delays unchanged. Reject invalid config before allocations. |
| Frozen-state RK4 step against the validated float64 reference | On explicit V,h,b/current/conductance samples from the reference and captured trajectory, V absolute error ≤0.0002 mV and h/b ≤0.000002 for one 0.1 ms step. No state is propagated in this local-error assay. Check both GPU and WASM. |
| GPU/WASM local numeric parity | Same frozen input suite: V error ≤0.0001 mV, h/b ≤0.000001, exact detector/guard/count decisions. Record all maxima and locations. Do not demand bitwise transcendental equality across compilers. |
| Free trajectories against actual author/reference output | Quiet 1 s and near-onset 2 s, then frozen-g 300 ms: finite states, h/b within [0,1], exact event counts and identities; event times within one ionic substep (0.1 ms). Retain complete unaligned V/h/b errors and waveforms, not just rates. A local-step pass alone is insufficient. |
| Backend free-trajectory parity | Same deterministic conditions and fixed conductances: event counts/identities exact, times within 0.1 ms; report unaligned V/h/b errors. Do not treat phase-shifted traces as byte-identical or suppress a count mismatch. |
| Graph integration | Tiny fixture with one opted-in DLM and generic presynaptic/postsynaptic cells: show receptor delay, observed DLM release and downstream current without changing edge weights. Trace odd/even buffer swaps and 1/3/4/128-step chunking. |
| Ownership/restart | Two brains sharing wiring have independent h/b/guard/events; disposing one preserves the other. Reset and capture/restore include the sparse state. |
| Timing/effector | Mixed 0.5 ms generic and 0.1 ms DLM events, exact packet coverage, no duplicate/gap/overrun, boundary events, Float32 timestamp rounding, snapshots and paused/resumed integration. Half-ms legacy fixtures retain their current outputs. |

The 43.5 pS five-cell reference remains a numerical equation test, not a reason to add coupling to the BANC model. The first real integration should use a newly pinned diagnostic bundle, unchanged graph, no training and paired backend runs with the same seed/inputs. Closed-loop changes in non-DLM neurons can be real consequences of the modified DLM output; do not incorrectly require those neurons to remain identical after the network is reconnected.

## Source and distribution boundary

The paper publishes the current-balance/channel equations, Q=39.2/V and the deterministic 100 µs RK4 scheme; supplementary Tables 2–4 provide the constants needed for the HOM/SNL profile. They are sufficient to derive a new implementation without copying the author program. The paper itself is CC BY 4.0. [Paper, Methods and license notice](https://pmc.ncbi.nlm.nih.gov/articles/PMC10232364/).

The separately archived program is CC BY-NC 4.0. That license restricts reproduction/adaptation of protected material to noncommercial purposes, so do not import the diagnostic `reference.mjs`, translate the author loader, or copy the archived code into this GPL-2.0-only runtime and label the combination GPL-only. Attribution alone does not remove the NC restriction. Keep author code and the NC diagnostic reference outside deployed assets. [Author archive](https://zenodo.org/records/7740678), [CC BY-NC legal terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).

The implementation route is new code derived from published mathematical methods and numeric constants, with equation/table attribution, and external comparison against the isolated author reference. The US Copyright Office distinguishes methods/systems from their protected expression; this is not a blanket clearance for adapting existing code. Since we have inspected that code, do not claim a formal clean-room process. The exact adaptation/distribution boundary remains a legal ambiguity if source expressions or substantial table/document formatting are copied; permission or a license review would resolve that case. Do not relicense the existing diagnostic files as a shortcut. [Copyright Office Circular 31](https://www.copyright.gov/circs/circ31.pdf).
