This is a **fast, numerically stable reconstruction, not an exact reproduction** of Bartussek & Lehmann's 2018 MN.b1 model. It does not yet justify changing the live BANC cell model. The paper models the blowfly *Calliphora* at 100 Hz; the 236 Hz cases here are explicit extrapolations. Everything in this folder is isolated from the application, neural graph, muscles and training.

The important uncertainty is in the input equation. Equations 2.2 and 2.4–2.6 sum currents from absolute afferent voltages, then rectify the aggregate. The text instead describes every electrical junction as rectifying. These operations differ: quiet afferents at −65 mV can cancel an active afferent before aggregate rectification. The adapted Matlab code is available only on request. This reconstruction retains the printed operation as its default and tests the per-junction interpretation separately, without fitting conductances to obtain a desired result. Neither interpretation reproduces all the paper's reported firing patterns with the presently recoverable input waveform and stated completion choices.

Sources are the [2018 article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6127168/), its [official supplementary PDF](https://doi.org/10.6084/m9.figshare.6940820.v1), the cited textbook author's [HH code](https://hallab.cs.dal.ca/images/7/73/hh.m), and the [Dryad figure-trace deposit](https://datadryad.org/dataset/doi:10.5061/dryad.fp3qb51). Downloaded source bytes and source URLs are preserved in `sources/` and `source-ledger.json`; `result.json` pins every numerical input/source. The supplementary PDF contains two pages of fitting and parameter-sensitivity results; it does **not** contain a complete parameter table or gating equations. The unrelated `atlas-*` files were fetched by the parent for a separate anatomical audit and are not numerical inputs here.

Reproduce from the repository root (Node built-ins and Python standard library; no package installation):

```sh
magick reports/flight-b1-reference/sources/figure-2.jpg -depth 8 reports/flight-b1-reference/sources/figure-2.ppm
.venv/bin/python reports/flight-b1-reference/digitize_waveform.py
node --test reports/flight-b1-reference/reference.test.mjs
node reports/flight-b1-reference/run.mjs
```

The last command regenerates `result.json`, compressed `traces.json.gz`, and prints a summary. The saved console output is `run.log`; test output is `tests.log`. The scalar work includes 32 fixed cases, five timestep comparisons, and three initialization checks. It is a bounded numerical characterization, not a parameter search. Main periodic runs last 1 s after a 100 ms quiescent warm-up; rates below use the final 500 ms. Per-case and total wall times are recorded, including stimulus-table construction. No implication about native/GPU throughput follows from this Node benchmark.

| Check | Result with the declared reconstruction |
|---|---|
| Isolated resting voltage | −55.687445 mV; no spikes; same settled state from initial −75, −65 and −55 mV |
| Visual-only input at −40 / −30 mV | Rest at −54.218822 / −53.336537 mV; no spikes |
| Single haltere EPSP, visual input disabled | 0.593709 mV above warm voltage, near the paper's approximate 0.5 mV electrical peak but **not** a fitted/validated trace match |
| Same EPSP, visual conductance retained | 0.714301 mV; the fitting experiment's visual operating point is unspecified |
| Printed aggregate rectification, 100 Hz periodic cases | No sustained spikes; even the stated 50-haltere volley fails to cross −40 mV |
| Per-junction interpretation, 100 Hz, 55 / 66 / 77 sensilla per organ, σ=1 ms | Final-window rates 50 / 50 / 80 Hz; paper reports approximately 50 / 66.7 / 100 Hz |
| Per-junction interpretation, 100 Hz, 46 per organ, σ=0.2 ms | 74 Hz at visual −40 mV; 100 Hz at −30 mV; paper reports 1:1 locking and a phase shift at both settings |
| Per-junction interpretation, 236 Hz, same 46/46 timings in milliseconds | One initial spike, zero spikes in final 500 ms at both visual settings |
| Heun 0.01 ms versus RK4 0.0025 ms | Same spike counts in all five comparison cases; maximum paired spike-time error 0.001238 ms, maximum sampled-voltage error 0.04064 mV; smaller step reduces error |

The tests cover the cited gate equations and removable singularities, current signs/densities, the two different rectification operations, waveform support/amplitude and population moments, initialization independence, actual spiking/transient-only timestep convergence, forcing-grid refinement, and rejection of unbounded inputs. Passing numerical tests establishes the reconstruction's internal behavior; it does not resolve the paper's missing source/input details.

The reported model uses

\[
C\dot V=I_{ext}-\bar g_{Na}m^3h(V-E_{Na})-\bar g_K n^4(V-E_K)-g_L(V-E_L),
\qquad \dot x=\alpha_x(V)(1-x)-\beta_x(V)x.
\]

Time is milliseconds, voltage millivolts, capacitance µF/cm², conductance mS/cm², and current µA/cm². Thus no additional `1000`, timestep multiplier or membrane area enters the differential equation. No absolute membrane area is selected. Converting these densities to the application's pF/nS/pA units would require an independently declared area; the paper does not identify one for the BANC cell.

| Quantity | Value | Status |
|---|---|---|
| \(g_L,E_L,\bar g_{Na}\) | 5.84 mS/cm², −52.26 mV, 165 mS/cm² | Fitted/reported in article §2.4 |
| \(\bar g_K\) | 36 mS/cm² | Explicit in supplement p.2 |
| \(g_H,g_W,g_{INs}\) | 0.16 per H, 0.08 per W, 1.0 total visual mS/cm² | Article §§2.3–2.4; wing value assigned from half-amplitude response |
| Sensory resting potential / peak excursion | −65 mV / +50 mV | Authors' assumed/rescaled input, §2.2 |
| Wing delay | 0.625 ms after haltere | Fitted in article, not a universal measured latency |
| Maximum population | 110 H + 110 W | H anatomically motivated; W count explicitly assumed by authors |
| Heun step | 0.01 ms; checked by authors over 0.005–0.02 ms | Explicit in article §2.4 |
| \(C,E_{Na},E_K\) | 1 µF/cm², +50 mV, −77 mV | **Inherited completion** from the cited textbook code, using a −65 mV voltage offset; not explicitly specified in B1 paper/supplement |
| Gate-rate functions, kinetic multiplier | Equations below; multiplier 1 | **Inherited completion** from cited code, not measured B1 channel kinetics |

With absolute voltage \(V\) in mV, the inherited gates are

\[
\alpha_m=0.1\frac{V+40}{1-e^{-(V+40)/10}},\quad
\beta_m=4e^{-(V+65)/18},\quad
\alpha_h=0.07e^{-(V+65)/20},\quad
\beta_h=\frac1{1+e^{-(V+35)/10}},
\]
\[
\alpha_n=0.01\frac{V+55}{1-e^{-(V+55)/10}},\qquad
\beta_n=0.125e^{-(V+65)/80}.
\]

The two singularities use their analytic limits. The original textbook code expresses voltage relative to its resting reference; `reference.mjs` explicitly shifts by +65 mV when evaluating those functions. No temperature correction, active adaptation, A-current, stochastic noise, voltage reset or imposed refractory timer is added.

For participating sensilla, the default input is exactly the *form* of the printed equations:

\[
\widetilde I = g_H\sum_i(U_{H,i}-V)+g_W\sum_i(U_{W,i}-V)+g_{INs}(U_{IN}-V),
\qquad I_{ext}=\max(0,\widetilde I).
\]

The separately labeled alternative sums `max(0,g*(U−V))` over each junction, including the visual junction. The meaning of `#CS` between spikes is not fully specified by the paper. Default connected counts equal participating population counts; a fixed 110+110 connected population is a separate sensitivity case. Neither silently assumes that a resting afferent has zero voltage or that only positive pulse increments should be summed.

Remaining local choices and limits:

- **Waveform:** Dryad metadata/file listings were accessible; the public file-stream URL returned 403 and the API download endpoints returned 401. No authentication was attempted. `waveform.json` is reproducibly digitized from the red `U_CS` line in Figure 2b. The baseline is −65 mV and peak is normalized to the article's 50 mV excursion. Horizontal resolution is 0.004535 ms/pixel and normalized vertical resolution about 1.01 mV/pixel. Interpolation, crop and endpoint choices are serialized. It is not the authors' ASCII waveform. The ±20% width cases are deliberately labeled sensitivity choices, not confidence bounds.
- **Population timing:** individual phases are deterministic normal quantiles, renormalized to the requested population SD and repeated each cycle. The paper does not supply the exact per-cell phases, random seed, discretization or cycle-to-cycle jitter procedure. We define volley time by the waveform **peak**, with haltere mean at 25% cycle and wing delayed by τ. This convention and upward −40 mV threshold crossing define reported phases; exact agreement with the paper's plotted spike phase cannot be claimed. No BANC endpoint or cell-specific phase map is assigned.
- **Initialization:** voltage starts at −65 mV, gates at their equilibrium there, followed by 100 ms of the scenario's quiet electrical connections/visual input. This is a local equilibration choice, not the authors' initial condition. For periodic drives, the prescribed stimulus is mathematically periodic across time zero; onset transients are excluded from final-half rate summaries.
- **236 Hz:** default σ and τ remain fixed in milliseconds. Two additional aggregate-input cases instead preserve their fractions of a wing cycle. Neither rescaling is a measured Drosophila adaptation. The unchanged HH recovery kinetics and input waveform need not support 236 Hz firing. These results cannot establish that biological B1, or a differently parameterized cell, cannot do so.
- **Numerics:** periodic forcing is tabulated at ≤0.0025 ms and interpolated. Heun evaluates the complete coupled voltage/gate derivative at both stages. An independently implemented RK4 integration schedule at 0.0025 ms supplies the comparison; it shares the physical right-hand side. Gate/current identities and stimulus semantics are tested separately. No step size was chosen to rescue an unstable trajectory.

An implementable numerical reference therefore exists, and it is inexpensive for isolated cells. A **faithful paper reproduction is still unestablished**. The concrete missing items are the adapted author Matlab code (especially rectification and `#CS` semantics), the original digitized afferent waveform and volley phases, and confirmation of inherited membrane constants/voltage reference. The current evidence supports keeping this folder as a reference and ambiguity test; it does not support transferring fitted *Calliphora* densities, adding BANC electrical edges, or substituting this model into live training.
