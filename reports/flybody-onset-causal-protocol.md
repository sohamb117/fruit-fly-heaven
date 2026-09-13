# Causal onset capture and replay

The actual original-UI capture and gated replay have completed. Both full-control and MN-rate/muscle baseline replays match exactly over 400 ms, and every required source hash matches. [Results and limits](flybody-onset-causal/README.md). The earlier 40 ms synthetic fixture remains separate plumbing evidence.

```sh
node scripts/capture-flybody-onset.mjs --output=reports/flybody-onset-causal
node scripts/replay-flybody-onset.mjs --output=reports/flybody-onset-causal
```

The capture uses its own original-UI Chromium instance, one real BANC brain, direct movement and the neural body clock. It installs response-route instrumentation before the body module executes, then observes **200 actual 2 ms steps, starting at physical time zero**. It does not advance the body manually or substitute neural outputs. After the final step, it clicks the normal UI Pause button and closes its own browser. Use `--fast=true` only when deliberately matching Fast mode; default is the 0.5 ms physiology mode. `--url=...`, `--timeout=...` and `--headless=false` are optional.

`capture.json` contains complete MN-rate maps, before/after qpos/qvel/act/ctrl/time/warm-start state, muscle activation/fatigue/force, internal state, each 1 ms muscle input and output, and each 200 µs wing command, opening, deployment and effective power. It also preserves the exact scene XML, heightfield samples, initial pose, fruit state, metadata, IO map and served-source hashes. Original module copies are saved under `sources/`, together with an end screenshot. The observer adds no physical state writes or forces.

## Replay gates

Replay runs sequential native MuJoCo WASM bodies in Node. It checks current mechanics/runtime source hashes against the capture and refuses a mismatched version. Keep the corresponding checkout, or deliberately restore the captured source version before interpreting a later replay.

1. Replay every recorded full-control interval continuously from the initial state. Require qpos error below `1e-9` and qvel below `1e-7`.
2. Replay the complete recorded MN-rate sequence through the actual production muscle and body code. Also require control error below `1e-9` and muscle-state error below `1e-6`.
3. Only if both pass, replay with DLM/DVM rates zero and every other recorded MN rate unchanged. Keep native wing gain 18, steering, passive springs and the resting target servo.
4. Repeat that intervention with the six active wing controls zero. This leaves native passive wing springs/damping, and distinguishes residual servo activity from DLM/DVM-driven beating.

There are no state corrections after the initial restore. Root forces stay zero. `replay.json` reports baseline errors, first airborne/overturned times, maximum height rise, angular speed, minimum uprightness and dense state traces. Failed baseline parity suppresses the counterfactual runs and interpretation. `--passive-wings=false` omits the final variant.

This is a counterfactual mechanical replay: changed body motion does not feed a newly simulated brain. It tests whether the recorded power-muscle recruitment was necessary for that onset, and whether the remaining active wing servo contributes after power removal. It does not identify a physiological gain or demonstrate recovered autonomous behavior.

## Completed fixture check

```sh
node scripts/replay-flybody-onset.mjs --fixture-only=true --output=reports/flybody-onset-fixture
```

The completed fixture used synthetic constant 80 Hz power-muscle stimulation and one extensor family on a freely moving native body for 40 ms. Full-control replay and MN-rate/muscle replay both had **zero qpos, qvel, control and muscle-state error**. All variants had zero externally applied force. [Fixture report](flybody-onset-fixture/replay.json). This validates the recorder, restoration and replay mechanics; it is not evidence about the actual BANC launch. Browser routing and the actual 400 ms capture subsequently passed in the result linked above.
