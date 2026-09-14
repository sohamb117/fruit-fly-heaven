# Fixed scalar-power diagnostics

`prepare-power-sweep.mjs` prepares four separate one-job plans using the existing `plans-v1/maintained-baseline` config and original wing/metadata bytes. Only job parameter 0, `flight_power_log_gain`, changes: 0, 0.1, 0.2 or 0.3, corresponding to physical gains 1, 1.105170, 1.221403 and 1.349859. All other 26 coordinates remain byte-for-byte numerically equal to the saved baseline vector. This is a predeclared diagnostic sweep, not optimizer updates, checkpoint changes or promotion.

Each plan uses seed 1290888, the staged maintained-flight environment, full native body, 0.5 s of live neural/event/muscle warm-up with the root held at `[0,0,3.5,1,0,0,0]`, then 5 s scored flight or physical failure. The reused config retains its original initial parameters; the job supplies the complete legal diagnostic vector. No affine operating-point profile is enabled.

```sh
node reports/flight-maintained-native-staging/prepare-power-sweep.mjs reports/flight-maintained-native-staging/power-sweep-v1
```

Preparation verifies the exact original baseline plan SHA, unchanged runner, all inherited sources, model overrides and asset manifest. Every derived plan adds source pins for the parent plan, reused config and builder. The output archives those three inputs, writes a provenance ledger and verifies that the only runtime change from the parent plan is parameter 0 and the descriptive job name. Existing output directories are refused. This command does not run evaluations, start a server, communicate with a coordinator or change live sources.

The sweep tests sensitivity below the previous approximately 2× power gain, where saturation may obscure parameter effects. That is a hypothesis to examine in subsequent saved evaluation records. Preparation alone says nothing about resulting flight performance.
