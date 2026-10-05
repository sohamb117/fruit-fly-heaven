# Structural proprioception implementation

The new local configuration keeps the BANC graph, neural parameters, 672-coefficient motor decoder, native articulated body, and existing flight score fixed. Changes occur in the sensory observers and input routing. Inputs still travel through BANC sensory cells, brain/VNC dynamics, individual motor neurons, and the existing muscle/body interface. No additional learned adapter, neural-network policy, decoder fitting or cloud deployment was performed.

## Implemented paths

| Path | Structural change | Remaining assumptions |
|---|---|---|
| Legs | Verified BANC catalog separates 160 claw position, 137 hook directional-motion and 335 club bidirectional-motion cells. Native signed tibia angle/velocity replace the generic self-motion channel. All 1,066 indices are accounted for; 434 unsupported cells receive zero added drive. | Individual claw/hook polarity is unknown. A declared native-positive coordinate prior and root-verified per-cell overrides are available; strict mode abstains on all 297 unknown directional cells. Preferred angles, gains and adaptation remain unidentified. |
| Antennae | Actual frozen head/antenna transforms and receiver position determine local airflow, including rotational velocity at each receiver. Family-level C/E responses have opposing polarity; D has a separate mixed response. No root-ID-random preferred axes. | The reduced body lacks the movable a2/a3 joint and true arista geometry. The observer is virtual; family transfer, gains and mechanics are unfit priors. F and ambiguous families have no added drive: 388 driven, 191 abstained. |
| Halteres | A damped virtual oscillator receives its own motor-derived drive and separate ipsilateral native-wing kinematic coupling. Zero left muscle drive no longer forces zero left virtual motion. Angle, velocity and phase persist across blocks and warm-up release. | Selected wing-roll coordinate, transmission, damping, resonance and field directions are unmeasured priors. Mechanical coupling evidence in soldier flies does not calibrate Drosophila coefficients. The observer applies no force to the native body. |
| Wing/tegula strain | Signed aerodynamic moments are reconstructed in the thorax frame from the matching native force cache. A virtual compliant hinge retains all three components before explicit per-cell receptive projection and rectification. | This is aerodynamic-load deformation, not measured cuticle strain or total inertial/contact/actuator reaction. Default fields share a mirrored thorax-X projection; their polarity is an engineering prior. Per-cell projection or explicit abstention can replace it. |

The leg catalog preserves raw organ, family, root-ID and function evidence rather than inferring physiological subtype from a reused SNpp name. Unsupported hair plates, unknown strain axes and ambiguous peripheral targets do not borrow tibia, speed or tilt feedback. Club movement is modeled separately from high-frequency vibration; the old collision-times-joint-speed proxy is not treated as a vibration recording.

The haltere directional candidate uses one explicit beam-axis prior per annotated population; it does not fabricate individual receptive-field assignments. A second, orientation-marginalized profile is supplied as a control. It intentionally removes assigned direction and is not the default candidate. Original version-1 haltere arithmetic remains exact.

## Clocks and failure handling

All observers use simulation time. Leg sampling is stateless. Antenna and wing-deformation dynamics evolve under the previous held measurement. Haltere dynamics use the 0.5 ms neural grid with 50 microsecond internal RK4 steps and actual same-side native joint position/velocity; within-block harmonic continuation is explicitly an approximation.

Wing moments and their thorax rotation come from the same final Euler force cache, 50 microseconds before the reported body time. They are sampled without another native forward evaluation. Native joints, forces and contacts are not changed by these observers.

Per-episode observers start fresh. Haltere state is reset once per episode and retained through airborne warm-up/release. Invalid structural inputs cannot leave legacy rates cached: corrected same-time samples are retryable, and rejected combined samples restore other observers. Snapshot contracts reject mismatched profiles, clocks and held-source data.

## Validation

The focused runtime suite passes **142 tests, zero failures or skips**. It covers signed and mirrored input response, family and per-cell identities, current-vector replacement, native frame covariance, finite-difference receiver velocities, causal timing, snapshot/reset behavior, malformed-sample rollback, disabled-feature parity, and existing visual/motor/episode contracts. Test log: [final-tests.tap](final-tests.tap). A further **14 packaging tests pass**, covering existing bundle behavior and strict validation of owned runtime sources and the leg catalog. All three real v2 bundles pass the browser packager's bundle validation.

The combined **real BANC WASM + native MuJoCo WASM** evaluation completed successfully with seed `2590888`, 500 ms of airborne root-held warm-up and five seconds of scored free flight. Its 672 decoder coefficients stayed exactly equal to the source configuration throughout. No simulation/input/clock error or coordinator write occurred.

| Recorded outcome | Value |
|---|---:|
| Existing maintained-flight criterion | Passed (`stage_success`) |
| Score | 5.3408 |
| Total qualifying flight | 4.682 s / 5 s |
| Longest continuous qualifying bout | 1.896 s |
| Qualifying bout at the final horizon | 1.672 s |
| Environment contact time | 0 s |
| Minimum root-up cosine during scored flight | 0.9651 |
| Wall time | 590.0 s |

The existing criterion requires at least one continuous qualifying second at the five-second horizon. This episode started airborne: it supplies no takeoff or landing evidence. It is one seeded evaluation of several sensory changes together, with unfit priors; it does not establish physiological accuracy, generalization, recovery learning, or which individual change caused the outcome.

The recorded signals confirm the structural path is active. Left own-haltere power was exactly zero in all 250 sampled observations, while its coupled virtual amplitude was 0.196–0.240 rad and 199 observations had nonzero left haltere current (maximum 2.423 pA). This verifies that zero own power no longer erases wing-driven sensory motion; it does not restore missing left hDVM activity. The sampled leg, antenna, visual and tegula populations had nonzero actual neural rates and applied currents. Antenna and leg mean applied currents varied only slightly in this flight, so these traces alone cannot identify their dynamic gains.

Evidence: [evaluation](structural-v2.evaluation.json), [summary](structural-v2.summary.json), [post-run audit](structural-v2.audit.json), [sensory trace](structural-v2.sensory.json), and [body observations](structural-v2.observations.json). The [140 retained poses](structural-v2.frames.json) span native time 0–5.5 s and include the final state; 2,500 body observations cover every scored 2 ms step. These are recorded state data, not a claim of visual inspection of every frame.

## Reproduction

The runtime modules and three `models/banc-*.json` catalogs/profiles are source-controlled candidates. The local bundle owns all 52 executable JavaScript sources plus its native XML/metadata and leg catalog. Prepared graph data and WASM binaries remain external SHA-256-verified dependencies.

```sh
node scripts/prepare-structural-feedback.mjs \
  --base-bundle=reports/sensory-feedback-20260915/v2/vision-airflow.bundle.json \
  --output-dir=reports/structural-proprioception-20260915/new-bundles

node scripts/evaluate-flight-feedback.mjs \
  --variant=vision-airflow --name=new-structural-evaluation \
  --bundle=reports/structural-proprioception-20260915/new-bundles/structural.bundle.json \
  --output-dir=reports/structural-proprioception-20260915 --capture=false
```

Use a fresh directory/name; completed evidence is not overwritten. The builder also creates `marginalized.bundle.json` and `prior-feedback.bundle.json` for controlled comparisons. It accepts an explicit existing base bundle, retains its decoder coefficients and body assets, and does not modify `web/training/config.json` or a coordinator checkpoint.

The browser packager now accepts these frozen runtime sources and their pinned leg catalog. See the [browser packaging handoff](browser-packaging.md) for the command and validation scope. The existing training page still requires a coordinator with the same configuration identity; packaging does not redirect the public trainer or bypass synchronization. No large browser package was built or browser training started during this change.

The evaluated candidate is [v2/structural.bundle.json](v2/structural.bundle.json), configuration hash `ecce268f712ade63f5f839b4a949426b81f1d4e3cdbb83e813660a9a827def65`, source fingerprint `c71fcc1a2e25bbd0192e14b9f790fc703a1a19334ac8c93c1c6872b02d0683a4`. The initial v1 packaging attempt was rejected before any simulation because its outer configuration retained a redundant `enabled` field; the v2 builder removes that field without changing model parameters.

## Calibration boundary

These structures now expose the missing quantities rather than asking the existing motor weights to compensate for them. Next identification work requires stimulus-response evidence for mechanical coefficients, receptor polarity and tuning, followed by recovery demonstrations and fitting the existing decoder. A higher reward under one collection of unfit sensory priors would not identify physiological truth.

Family and mechanical sources are documented alongside [antenna evidence](../antenna-structural-v2-20260915/README.md), [haltere evidence](../haltere-structural-20260915/README.md), and the leg catalog's explicit provenance. Primary examples: [leg proprioceptor physiology](https://doi.org/10.1016/j.neuron.2018.09.009), [antennal wind responses](https://doi.org/10.1038/nature07843), and [wing sensory anatomy](https://doi.org/10.7554/eLife.107867).
