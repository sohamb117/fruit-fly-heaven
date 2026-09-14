# Five-cell DLM reference — numerical comparison passed

The JavaScript port was compared with the actual archived author's Brian2 implementation under the pinned modern environment. All three conditions passed the fixed tolerances in `plan-modern.json`; see [comparison-001.json](comparison-001.json), [author manifest](author-run-001/manifest.json), and [JavaScript manifest](js-run-001/manifest.json). No production BANC files were changed.

| Condition | Maximum voltage difference | Maximum h / b difference | Spikes, JS / author |
| --- | ---: | ---: | ---: |
| quiet |5.68e−14mV |1.20e−16 /3.89e−16 |0 /0 |
| near-onset |1.11e−8mV |1.60e−10 /2.15e−11 |60 /60 |
| weak-coupled |1.82e−7mV |2.60e−9 /3.51e−10 |70 /70 |

Spike counts and neuron identities matched. The maximum event-time difference was2.28e−13ms, at floating-point roundoff scale. This establishes numerical agreement for the stated conditions on Brian2 2.9.0; it does not validate incoming BANC synaptic conductances, all neuron types, muscle conversion, or flight.

## Source and license

Hürkey S, Niemeyer N, Schleimer JH, Ryglewski S, Schreiber S, Duch C (2023), [*Gap junctions desynchronize a neural circuit to stabilize insect flight*](https://doi.org/10.1038/s41586-023-06099-0).

The [author code archive, Zenodo7740678 v1.0.0](https://zenodo.org/records/7740678), is labeled **CC-BY-NC-4.0** by its [record API](https://zenodo.org/api/records/7740678). The paper is separately CC-BY4.0. The archived code/configuration attribution and noncommercial license notice are retained here; this diagnostic has not been added to the deployed runtime. [Code license](https://creativecommons.org/licenses/by-nc/4.0/).

`reference.mjs` is a new JavaScript expression of the published deterministic V,h,b equations. `run-author.py` imports the archived, unchanged `utils.sim.run_sim` and `cfg/Berger_SNL.json`: it does not contain a second transcription of those equations. The plan pins both the reference files and all retained author source/configuration files by SHA-256. The source archive's published MD5 was independently verified during extraction; `../source-provenance.json` preserves that evidence.

## Numerical contract

- Five identical HOM/SNL neurons; V in mV, conductances in nS, currents in pA, capacitance in pF, time in ms; h and b are dimensionless. Constants are those in the author SNL configuration, including C130pF, gNa431.2nS and gShab137.68216nS.
- Deterministic RK4 at0.1ms (100µs), matching the author's Fig.3B/C step. No white noise, resets, voltage clamps, generic adaptation current, or external BANC inputs.
- Homogeneous bidirectional coupling only: archived scalar0.043499999686087025nS, approximately43.5pS. No physiological MN1–4 identities or stronger-pair assignments are invented.
- Match the author's **Brian2 scheduling**: summed gap current is computed from the beginning-of-step voltages once, then held during the neuronal RK4 stages. Fully recomputing electrical currents at every RK4 stage would be a different discretization. See [Brian2 2.4.2 summed-variable scheduling](https://raw.githubusercontent.com/brian-team/brian2/2.4.2/brian2/synapses/synapses.py), `SummedVariableUpdater`, and [RK4 implementation](https://raw.githubusercontent.com/brian-team/brian2/2.4.2/brian2/stateupdaters/explicit.py).
- Traces use StateMonitor's beginning-of-tick convention. Spike detection examines the updated V>−10mV, with the author's10ms event guard and Brian's time label t; that label refers to threshold evaluation after the update to t+dt. V is never reset or frozen by the guard. No sub-tick spike time is inferred.

## Compared conditions

| Condition | Duration | Constant input | Gap conductance | Initial state |
| --- | ---: | ---: | ---: | --- |
| quiet |1s |0pA |0 |All five at V=−60mV,h=b=0.146 |
| near-onset |2s |108.75pA |0 |Explicit small voltage spread, same gates |
| weak-coupled |2s |108.75pA |43.5pS |Exactly the same spread as near-onset |

The spread is V=[−60,−59.9,−60.1,−59.8,−60.2]mV with h=b=0.146. These are declared diagnostic initial conditions, not measured voltages, desired firing phases, or the hand-selected Fig.3B example. The paired implementations receive precisely the same values. Identical initial conditions with identical deterministic inputs would preserve symmetry, so such a run must not be represented as a test of spontaneous symmetry breaking.

The comparison checks every sampled voltage and gate, spike counts/identities/times, zero spikes in the quiet condition, and actual spiking in active conditions. Fixed engineering tolerances in the plan are0.0001mV for V,0.000001 for gates, and0.10000001ms for event times. They are implementation acceptance thresholds, not physiological uncertainty estimates. The output retains actual maximum errors and locations; a failed comparison is not silently accepted or retuned.

The observed agreement establishes numerical implementation agreement for these conditions. Assessing PRCs, stable splay/strong-coupling behavior, f–I curves and initialization robustness remains separate. No108.75pA tonic current is added to canonical BANC.

## Dependency status and run protocol

Inspected available environments without importing or running Brian2:

- System Python3.14.6: NumPy2.4.2 available; Brian2,SymPy,SciPy andMatplotlib absent.
- Existing repo `.venv`, Python3.13.15: NumPy2.5.3 available; Brian2,SymPy,SciPy andMatplotlib absent.
- Node23.7.0 available; the JavaScript port uses only Node built-ins.

The original author environment declares Python3.9.7, Brian2 2.4.2, NumPy1.20.3, SymPy1.8, SciPy1.7.1, Matplotlib3.4.3 and pip21.2.4 in `../source/drosophila_wing_cpg/drosophila_mini.yml`. `plan-legacy.json` preserves the author-era comparison option, with a5s weak-coupling arm. That environment is not required for the prepared modern comparison.

`plan-modern.json` instead pins the installed Python3.13.15 and **Brian2 2.9.0, NumPy2.2.6 and SymPy1.14.0**, plus their actual dependencies. `requirements-modern.txt` contains exact versions and verified PyPI wheel hashes. Matching macOS arm64 CPython3.13 wheels exist; `modern-environment-wheel-metadata.json` records their download URLs, sizes and hashes. This is the actual unchanged author model/function running on a **different simulator/dependency environment**, not a reproduction of the paper's original binary environment. The difference is present in the plan, runtime manifest and comparison result.

Direct imports of the retained author `utils.sim` path are NumPy,Brian2,SymPy and Python's standard library. SciPy is absent and the unchanged author module imports successfully. Matplotlib, plotting scripts, conda and the old Python environment are unnecessary for this path. Actual remaining packages are Cython3.3.0,PyParsing3.3.2,Jinja2 3.1.6,setuptools84.0.0,packaging26.3,mpmath1.3.0 andMarkupSafe3.0.3. `installed-environment.json` records distribution versions and the import-only check. This is not a neuron-simulation or numerical-validation result.

`run-author.py` checks the exact selected environment and source pins, then explicitly selects NumPy code generation; it never downloads, installs, builds or substitutes packages. The parent used an isolated wheel-only/no-cache install without modifying the repository's existing environment. To reproduce the environment in a fresh checkout where the target is absent:

```sh
uv --no-cache venv --python .venv/bin/python reports/flight-dlm-published-model/reference/.venv
```

```sh
uv --no-cache pip install --python reports/flight-dlm-published-model/reference/.venv/bin/python --only-binary=:all: --require-hashes -r reports/flight-dlm-published-model/reference/requirements-modern.txt
```

The completed run used the commands below from the repository root. Choose new output names when repeating it:

```sh
node reports/flight-dlm-published-model/reference/reference.mjs reports/flight-dlm-published-model/reference/plan-modern.json reports/flight-dlm-published-model/reference/js-run-001
```

```sh
reports/flight-dlm-published-model/reference/.venv/bin/python reports/flight-dlm-published-model/reference/run-author.py reports/flight-dlm-published-model/reference/plan-modern.json reports/flight-dlm-published-model/reference/author-run-001
```

```sh
node reports/flight-dlm-published-model/reference/compare.mjs reports/flight-dlm-published-model/reference/js-run-001 reports/flight-dlm-published-model/reference/author-run-001 reports/flight-dlm-published-model/reference/comparison-001.json
```

Output directories/files must be new. Each backend writes finite, little-endian float64 state traces and event lists, plus a completion manifest. The modern paired binary traces require12.8MB (20.5MB for the longer legacy plan); there is no neural graph, body model, optimizer, coordinator, or network activity in these runs. A pending or failed comparison must remain labeled unvalidated.
