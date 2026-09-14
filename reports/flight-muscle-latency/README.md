The unchanged classical controller passed with direct force and with the actual native muscle dynamics. Adding the modeled 50 ms rate smoother caused it to fail over the same 1.5 s release.

| Output path | Maximum tilt | Final filtered angular speed | COM height change | Maximum horizontal drift | Criterion |
|---|---:|---:|---:|---:|---|
| Direct normalized force | 8.03° | 2.84 rad/s | −0.118 cm | 5.53 cm | Pass |
| 28 native muscles, 15/40 ms activation/deactivation | 6.68° | 2.29 rad/s | +0.007 cm | 2.76 cm | Pass |
| 50 ms rate smoother, then the same native muscles | 112.62° | 36.27 rad/s | −9.867 cm | 5.79 cm | Fail |

All arms completed 30,000 native steps, with no contacts or instability warnings. The predeclared criterion required tilt below 20° throughout, final filtered angular speed below 3 rad/s, absolute COM height change below 10 cm, and no contacts or warnings. There was no horizontal controller.

The direct arm exactly reproduced all 750 historical samples, final state and metrics from [the earlier positive control](../flight-classical-control/result.json). All three initial plant/wing hashes equal `a29ad3a08caa5a2fa54b4dc22d4eb587034dd110f6f76d4f77345f79393948aa`. `historicalDirectGate.passed`, `pairedInitialStateIdentical`, `sourceUnchanged` and `completed` are true. No new calibration or gain search was performed.

Each arm used the same reduced free-root plus six-wing-joint plant, trim, measured response Jacobian, controller gains, and 100 ms restrained plant warm-up driven by direct trim. Before release, native muscle activation and the rate-filter state were initialized at the trim-force inverse; fatigue started at zero. The prepared muscle/filter arrays matched exactly across arms, with at most `2.27e-8` normalized-force rounding error. This is a prepared operating point, not a simulated neural or fatigue equilibrium. Direct-force records contain an unused copy of that prepared state; no muscle kernel drives that arm.

The native arms used real `WasmMuscles` at 1 ms with length 1, shortening velocity 0, maximum force 1 and energy 1. The controller requested excitation `clamp(desiredForce / (1 − fatigue), 0, 1)` every 2 ms, using actual evolving fatigue. The full force-multiplier formula is recorded in the result. Excitation was mapped through the canonical legacy rate/80 decoder. The additional filter used synthetic held rates and the nominal 50 ms EMA with four 0.5 ms updates per command; it did not execute BANC or generate spikes. Controller access to fatigue is an explicit diagnostic assumption.

Native-muscle force tracking RMS error was 0.115 without the rate smoother and 0.342 with it. Inverse excitation clipping affected 12/21,000 muscle commands (0.057%) versus 4,525/21,000 (21.55%); maximum fatigue was 0.0854 versus 0.0898. The failed trajectory therefore includes nonlinear saturation and changed force availability after divergence. It shows that this fixed controller tolerates the modeled muscle dynamics but is not robust to the added rate-smoothing stage. It does not establish that every controller with that delay must fail, or validate biological timing, BANC control, takeoff or landing.

Artifacts: [result and trajectories](result.json), [exact executed script](diagnostic-source.used.mjs), [reduced XML](fixed-nonwing.xml). The executed script SHA256 is `90480a9171a3b71b080161a96dee5110affb326365be243aacd2abf27fc0b8fa`. The historical report and its source snapshot were preserved.

To repeat into a **new** directory:

```sh
node scripts/diagnose-flight-muscle-latency.mjs --run --output=reports/flight-muscle-latency-repeat
```

The diagnostic writes only existing wing actuator controls after release; it applies no external force and performs no root reset. Production model/controller code was not changed for this assay.
