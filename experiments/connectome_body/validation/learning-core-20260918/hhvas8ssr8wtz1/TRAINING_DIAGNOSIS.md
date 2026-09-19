# Training assessment, September 18, 2026

Read-only audit of the active BC-to-PPO window on `hhvas8ssr8wtz1`.
The active run and frozen protocol were not changed. Source and deployed
configuration were inspected; these findings do not establish a specific code bug.

## Observed failures

- Latest inspected hover evaluations: BANC survived 381.55/5000 decisions on
  average; MaleCNS 362.1/5000. Every evaluated fly fell. Success failure is not
  merely a strict final dwell threshold. These are seed-0 results.
- Real C. elegans on worm locomotion achieved 8/20 successful validation episodes
  at both 120k and 140k PPO interactions, then 4/20 at 160k and subsequently zero.
  The best checkpoint remains retained. Rewired worm validation has remained zero.
  This is a single-seed, validation-selected observation, not a replicated effect.

## Imitation compared with a constant predictor

The constant predictor uses each action channel's mean over the common training
demonstrations, evaluated against the held-out demonstration actions.

| Task / predictor | Validation action MSE |
| --- | ---: |
| Hover constant training-action mean | 0.0031846659 |
| Hover BANC after BC | 0.0031331277 |
| Hover MaleCNS after BC | 0.0031431356 |
| Worm constant training-action mean | 0.3808038831 |
| Worm real C. elegans after BC | 0.0726050088 |

BANC hover improves only about 1.6% over the constant predictor, MaleCNS about
1.3%. The decrease from initial BC loss mostly fails to demonstrate useful
observation-dependent prediction. Worm BC shows a much larger improvement.
Action MSE and closed-loop success must both be inspected.

## Deployed time and optimization settings

The live source/configuration, rather than default values, supplied these values:

| Quantity | Fly | Worm |
| --- | ---: | ---: |
| Decision interval | 0.0002 s | 0.01 s |
| Neural time constant | 0.02 s | 0.02 s |
| PPO differentiable sequence, 16 decisions | 0.0032 s | 0.16 s |
| BC differentiable sequence, 32 decisions | 0.0064 s | 0.32 s |
| Discount e-folding time, gamma=0.99 | 0.01990 s | 0.99499 s |
| GAE coefficient e-folding time, gamma*lambda=0.9405 | 0.003260 s | 0.163016 s |

These e-folding times describe coefficients, not hard cutoffs: value
bootstrapping and carried recurrent state still exist. The short fly gradient
window relative to the neural time constant is a plausible learning bottleneck,
not an established sole cause. Identical step counts imply very different
physical-time learning windows across bodies.

Recent 100-update summaries at the audit snapshot:

- BANC hover approximate KL: median 0.000484, maximum 0.005618.
- Real C. elegans worm approximate KL: median 0.10423, maximum 1.07138.

The trainer records approximate KL but has no KL-triggered early stop. It uses
four PPO epochs, sequence minibatches of 16 x 4 decisions, fixed pre-tanh
exploration standard deviation exp(-1)=0.367879, and gradient clipping. These
are candidates for controlled stability checks; correlation with regression
does not prove causation.

Source inspection found the expected BC checkpoint identity checks and policy
load, fresh PPO optimizer/critic, recurrent reset at the handoff, stored raw
actions with tanh-corrected log probabilities, episode reset masks, and
termination-aware GAE. This inspection is not a comprehensive correctness proof.

## Most informative next checks

1. Complete short adapter-only MLP and matched-GRU controls using the same
   observation/action interface, datasets and evaluation scenarios. No such
   full-training control result was available at this audit.
2. Require BC to beat the constant-action predictor materially, and inspect
   closed-loop survival before treating BC initialization as useful scaffolding.
3. Test longer physical-time gradient windows for fly control and bounded PPO
   policy updates as separately versioned diagnostics. Preserve current results.
4. Evaluate the retained successful worm policy on held-out episodes; compare
   controlled continuations to determine whether PPO can retain that behavior.

PPO clipping does not guarantee a small policy change; OpenAI's reference
implementation discusses KL-based early stopping:
https://spinningup.openai.com/en/latest/algorithms/ppo.html
