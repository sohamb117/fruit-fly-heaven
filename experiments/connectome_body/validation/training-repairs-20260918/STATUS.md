# Training repair qualification — September 18, 2026

This is a development qualification, separate from the frozen scientific run.
The old source, datasets, checkpoints and six active workers remain intact.
New repair runs use the existing pod and original 24-hour deadline.

Changes under test:

- Fixed observation normalization fitted only to valid training demonstrations;
  buffers are saved with the policy and used by both actor and critic.
- Per-action standardized BC loss and training-action mean initialization.
- Fly PPO sequence/burn-in of 128 decisions (25.6 ms), rollout of 512 decisions
  (102.4 ms), discount decay of 200 ms and GAE coefficient decay of 50 ms.
- Exact conditional Gaussian KL stopping at 1.5 times target KL 0.01;
  learning rate 0.0001 and fixed exploration log standard deviation -3.
- Same stateless adapter family, ports, parameter budgets, body and public data.

119 local regression tests passed, including causal BPTT, normalization/resume,
BC-to-PPO handoff, KL stopping, parameter accounting and plan construction.
This validates software behavior, not scientific performance.

Remote isolated runtime:
`/workspace/training-repairs-20260918/connectome_body`

Supervisor PID 14064; MLP PID 14065; GRU PID 14066.
Both development controls use 20 BC epochs, batch 4, sequence 128, followed by
up to 40k PPO interactions. They use only four episodes per evaluation. These
settings and the increased expert exposures are development choices, not a
budget-matched comparison with the original five-epoch run.

The supervisor limits this first check to 35 minutes inside the original paid
window and restores the old admission limit on exit. It does not stop the pod.
The user selected manual provider shutdown at September 19, 12:08:34 PM EDT.

Source upload SHA256:
`ecde304253c9439ba44e6b2f509a5ca8a3220d94b48fd12afb7a19c79e387017`

Closed-loop improvement is not yet established. Do not launch a revised full
scientific matrix solely because these software checks pass.

## Second candidate and continuation

The first normalization-only candidate was retired with checkpoints preserved
because its random output initialization produced excessive action error. The
second candidate adds fixed training-data action coordinates before tanh and
zero-initializes the final trainable decoder, starting at the constant training
action mean. It also tests the same 2 ms neural time constant for every substrate.
The action interface, observations, reward and parameter budgets do not change.

Remote source: `/workspace/training-repairs-v2-20260918/connectome_body`.
Initial supervisor PID 14957, workers 14958–14960; that window is finished.
The three candidates paused cleanly during BC at the 35-minute window limit:

| Candidate | Completed BC exposures | Latest validation action MSE | Hover success |
| --- | ---: | ---: | ---: |
| Adapter-only MLP | 1,440,000 | 0.00013613 | 0% |
| Matched GRU | 1,360,000 | 0.00016525 | 0% |
| BANC, 2 ms | 624,608 (latest evaluation at 560,000) | 0.00178309 | 0% |

These are development validation scores; no PPO stage had started at that point.
The original BANC five-epoch BC error was 0.00313313, with 0% hover success.
Different settings and exposure counts preclude a paired scientific effect claim.

The second candidate passed 24 native regression tests on the pod and 80 relevant
tests after restoration into the local checkout. Its downloaded source archive
SHA256 is `7c38b2289f6ad28c5e32a2b68da4b1bb1794ac8130bc13b8b947cc4c91ab2348`.

The continuation uses `runs/continuation/` for scheduling and resumes
`runs/controls/{adapter_only,gru,real}/` without changing request identities.
It is bounded to 90 minutes and the original paid deadline, whichever is sooner.

### Mechanistic measurements

An isolated small-signal probe of BANC's ports measured RMS gains at 200 Hz:
0.0001301 for tau=20 ms, 0.005260 for 2 ms and 0.05013 for 0.2 ms. At 20 ms,
the gain at 10 Hz was 0.01589, about 122 times larger than the gain at 200 Hz.
This suggests a bandwidth bottleneck; it does not establish a sufficient remedy.

The common expert's wing-command peaks occur at 220 or 440 Hz. Fractions of
within-trajectory wing-command power above 50 Hz range from about 70% to 97%.

A private BANC linear-readout ridge probe, fitted only to training trajectories,
reached validation MSE 0.0014191 but 0% closed-loop success. It was not installed
into the production controller or counted as a scientific trial.
