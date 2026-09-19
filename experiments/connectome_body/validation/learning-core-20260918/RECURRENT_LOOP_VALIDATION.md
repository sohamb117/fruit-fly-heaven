# Recurrent PPO loop validation — September 18, 2026

The production optimizer **does perform truncated backpropagation through time**.
Rollouts are collected without autograd, then ordered sequences are recomputed
with autograd during PPO optimization. The deployed gradient window is 16 body
decisions, not zero. No production policy/trainer source was changed by this audit.

## Local verification

54 tests passed (one native-FlyBody test excluded) across:

- `tests/test_recurrent_ppo_integrity.py`
- `tests/test_compatibility_training.py`
- `tests/test_compatibility_imitation.py`
- `tests/test_compatibility_controller.py`

The integrity tests isolate a final-decision-only actor advantage and verify
nonzero gradients at the first encoder output. Deliberately detaching hidden
state removes all earlier encoder gradients while retaining the same forward
values. This negative control establishes that the test detects severed BPTT.
An inserted episode reset cuts gradients into the preceding episode.

Additional checks cover log-probability replay with burn-in and episode resets,
unrolled sparse gradients against a dense reference and finite differences,
terminal/time-limit GAE semantics, policy update direction, BC-to-PPO checkpoint
handoff, and exact optimizer/state/RNG resume.

A delayed-cue task uses the unchanged production collect/optimize/run methods.
The cue is visible only at reset; the rewarded response occurs at decision 8.
After 16,384 training interactions, final policies on 200 fresh held-out episodes:

| Controller | Mean score | Success |
| --- | ---: | ---: |
| GRU | 0.9999634 | 100% |
| Stateless adapter | 0.7508070 | 0% |

This demonstrates temporal learning in the loop; it is a software diagnostic,
not evidence of successful embodied control or a biological topology advantage.
Log: `recurrent-loop-tests.log`.

The first broader local invocation was blocked before training by sandbox denial
of the read-only macOS CPU identity query. The complete rerun passed with that
access enabled. Initial test-fixture path reuse was corrected in the test itself.

## Full-size CUDA verification

`scripts/audit_recurrent_ppo.py` ran on isolated copies of live policies using
their deployed source, actual graph caches, and expert observation sequences.
It invoked the production `Trainer.optimize()` with a zero-rate optimizer and
terminal-only advantage. The active training workers were not modified.

All three audits passed on the RTX 4090 in 25.36 seconds:

| Policy | Copied checkpoint step | First encoder-output gradient L1 | Max replay log-probability error |
| --- | ---: | ---: | ---: |
| Full BANC / hover | 672,288 | 0.0000596712 | 0 |
| Full MaleCNS / hover | 522,560 | 0.0000450397 | 0 |
| C. elegans / worm | 659,968 | 0.0692210 | 0.000002861 |

For every policy the detached-state negative control eliminated earlier
gradients, and an episode reset eliminated all gradients across that boundary.
Gradient magnitudes across these different policies are not comparable performance
metrics. Machine-readable evidence: `recurrent-loop-cuda-audit.json`.

## What remains unresolved

Passing these checks rules out missing temporal gradients in the exercised paths.
It does not establish that the current horizon, gain, exploration scale, adapter
capacity, or PPO update sizes are suitable for FlyBody. At the deployed 0.2 ms
decision interval, 16 steps cover only 3.2 ms while the neural time constant is
20 ms. The poor BC constant-action comparison and worm PPO regression in
`hhvas8ssr8wtz1/TRAINING_DIAGNOSIS.md` remain real concerns.

The MLP and matched-GRU hover conditions from the existing frozen plan are being
prioritized for a bounded comparison on the existing pod. Their source, teacher
data, observations/actions, scientific training budgets, and original 24-hour
deadline remain those of the main plan. This changes scheduling only.
