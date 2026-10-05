# Isolated mechanical-current transfer

A direct mechanical-current prior preserves opposite-rotation information in the tested sensory cells. The neurons, native kernel and geometry are unchanged from the [failed DC-rate bridge](../flight-haltere-current-check/README.md). This is a usable isolated input path, **not validated BANC flight or measured receptor physiology**.

Current is `Imax × c / (1+c)`, with `c` the positive signed bending projection divided by an explicit mechanical scale. The four tested asymptotic current caps are sensitivity choices. No neural or behavioral reward was used to fit them; the original DC rate-to-current conversion is bypassed only in these disconnected test cells.

| Current cap | Output at zero rotation,0.5ms inputs | Output with2ms input holds |
|---:|---:|---:|
|100pA|0Hz|4–6Hz|
|200pA|47Hz|48Hz|
|400pA|118Hz|96Hz|
|800pA|236Hz|188–204Hz|

At the800pA cap, all eight cells fire once per236Hz cycle in the zero-rotation case. Opposite±20rad/s rotation changes spike timing in8/8 cells for each of the three axes, while their mean rates remain236Hz. Thus averaging only output rate would again hide useful timing information. This is not evidence that these eight assumed orientations correspond to particular BANC cells. Current amplitude, virtual oscillation, beam geometry, and the eventual cell assignments remain model priors.

The56 fixed cases completed in about2.15s, with the unchanged0.5ms neuron step. Coarse2ms held inputs change both rate and phase, so the proposed runtime must deliver per-neural-tick currents rather than hold one waveform sample across the body block. The two-second tests use the final second for measurement and retain all spike times.

`plan.json` and `result.json` record exact sources, parameters and conditions. Run `node reports/flight-haltere-current-direct/analyze.mjs` to recompute the signed-pair comparison from both saved experiments; `run.mjs` refuses to overwrite the frozen experiment. No recurrent BANC network, body, muscle, coordinator or training run was executed.
