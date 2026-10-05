"""Causal and numerical checks of the production recurrent PPO training loop.

These are software diagnostics, not scientific connectome performance results.
In particular, a final-decision-only advantage isolates temporal actor gradients
from the critic and from losses on earlier decisions.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_compatibility_training import short_spec

from connectome_body.compatibility.training import Trainer
from connectome_body.policy import Actor
from connectome_body.train import advantages


def replay(actor, observations, resets, initial):
    state, means = initial, []
    context = actor.context()
    for obs, reset in zip(observations, resets, strict=True):
        mean, state = actor(obs, state, context, reset=reset, squash=False)
        means.append(mean)
    return torch.stack(means), state


def temporal_probe(trainer, batch, snapshots, *, detach=False):
    """Use the actual optimize() method; only the last action receives credit."""
    batch = {k: v.detach().clone() for k, v in batch.items()}
    batch["advantage"].zero_()
    batch["advantage"][-1] = 1
    before = trainer.actor.forward
    if detach:
        # Deliberate defect: unchanged forward values, but sever temporal credit.
        def severed(obs, state, *args, **kwargs):
            return before(obs, state.detach(), *args, **kwargs)

        trainer.actor.forward = severed
    retained = []

    def remember(module, inputs, output):
        if torch.is_grad_enabled():
            output.retain_grad()
            retained.append(output)

    handle = trainer.actor.encoder.register_forward_hook(remember)
    try:
        trainer.optimize(batch, snapshots)
    finally:
        handle.remove()
        trainer.actor.forward = before
    return [0.0 if z.grad is None else float(z.grad.abs().sum()) for z in retained]


@pytest.mark.parametrize(
    "kind,regime", [("connectome", "adapters"), ("connectome", "joint"), ("gru", "joint")]
)
def test_real_optimizer_temporal_credit_reset_boundary_and_detach_negative_control(
    tmp_path, compatibility_manifest_factory, kind, regime
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, kind, regime)
    spec = replace(
        spec,
        body=replace(spec.body, horizon=32),
        training=replace(
            spec.training,
            interactions=64,
            eval_every=64,
            rollout_steps=16,
            sequence_length=16,
            burn_in=0,
            epochs=1,
        ),
    )
    trainer = Trainer(spec, tmp_path / "run")
    try:
        batch, snapshots = trainer.collect()
        assert not batch["logp"].requires_grad  # Rollout storage is intentionally detached.
        trainer.optimizer = torch.optim.SGD(
            [*trainer.actor_parameters, *trainer.critic.parameters()], lr=0
        )
        intact = temporal_probe(trainer, batch, snapshots)
        severed = temporal_probe(trainer, batch, snapshots, detach=True)
        assert len(intact) == len(severed) == 16
        assert intact[0] > 1e-12, intact  # Only the final decision had nonzero advantage.
        assert max(severed[:-1]) == 0, severed
        assert severed[-1] > 0 or kind == "connectome"

        # A new episode must cut every gradient from that episode into its predecessor.
        batch["reset"][8] = True
        with torch.no_grad():
            mean, _ = replay(trainer.actor, batch["obs"], batch["reset"], snapshots[0])
            batch["raw"] = mean + 0.3
            batch["logp"] = Actor.log_probability(mean, trainer.log_std, batch["raw"])
        bounded = temporal_probe(trainer, batch, snapshots)
        assert max(bounded[:8]) == 0, bounded
        assert bounded[8] > 1e-12, bounded
    finally:
        trainer.close()


@pytest.mark.parametrize("kind", ["connectome", "gru"])
def test_collected_probabilities_replay_exactly_with_burnin_and_episode_resets(
    tmp_path, compatibility_manifest_factory, kind
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, kind)
    spec = replace(
        spec,
        training=replace(
            spec.training,
            interactions=64,
            eval_every=64,
            rollout_steps=16,
            sequence_length=4,
            burn_in=3,
        ),
    )
    trainer = Trainer(spec, tmp_path / "run")
    try:
        batch, snapshots = trainer.collect()
        assert batch["reset"][8].all(), "Native fixture must cross an episode boundary"
        for start in range(0, 16, 4):
            burn = max(0, start - 3)
            with torch.no_grad():
                _, state = (
                    replay(
                        trainer.actor,
                        batch["obs"][burn:start],
                        batch["reset"][burn:start],
                        snapshots[burn],
                    )
                    if start > burn
                    else (None, snapshots[burn])
                )
            mean, _ = replay(
                trainer.actor,
                batch["obs"][start : start + 4],
                batch["reset"][start : start + 4],
                state,
            )
            logp = Actor.log_probability(mean, trainer.log_std, batch["raw"][start : start + 4])
            torch.testing.assert_close(logp, batch["logp"][start : start + 4], rtol=0, atol=2e-6)
            torch.testing.assert_close(
                (logp - batch["logp"][start : start + 4]).exp(),
                torch.ones_like(logp),
                rtol=0,
                atol=2e-6,
            )
    finally:
        trainer.close()


@pytest.mark.parametrize("regime", ["adapters", "joint"])
def test_unrolled_sparse_policy_gradients_match_dense_and_finite_difference(
    tmp_path, compatibility_manifest_factory, regime
):
    trainer = Trainer(
        short_spec(tmp_path, compatibility_manifest_factory, regime=regime), tmp_path / "run"
    )
    try:
        sparse = trainer.actor.double()
        dense = copy.deepcopy(sparse)

        def dense_multiply(state, values=None):
            core = dense.core
            weights = core.edge_values() if values is None else values
            matrix = weights.new_zeros(core.count, core.count).index_put(
                (core.dst, core.src), weights
            )
            return state @ matrix.T

        dense.core.multiply = dense_multiply
        x = torch.randn(16, 2, sparse.obs_dim, dtype=torch.float64) * 0.2
        x.requires_grad_()
        resets = torch.zeros(16, 2, dtype=torch.bool)
        initial = sparse.reset(2)

        def objective(actor, observations):
            mean, _ = replay(actor, observations, resets, initial)
            return (mean[-1] - 0.3).square().sum()

        a, b = objective(sparse, x), objective(dense, x)
        ga = torch.autograd.grad(a, [x, *[p for p in sparse.parameters() if p.requires_grad]])
        gb = torch.autograd.grad(b, [x, *[p for p in dense.parameters() if p.requires_grad]])
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        for actual, expected in zip(ga, gb, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-11)
        assert ga[0][0].abs().max() > 1e-12
        # Numerical perturbation of an EARLY input, observed only through the final loss.
        channel = int(ga[0][0, 0].abs().argmax())
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[0, 0, channel] += 1e-5
        minus[0, 0, channel] -= 1e-5
        with torch.no_grad():
            numerical = (objective(sparse, plus) - objective(sparse, minus)) / 2e-5
        torch.testing.assert_close(numerical, ga[0][0, 0, channel], rtol=2e-5, atol=1e-9)
    finally:
        trainer.close()


def test_gae_terminal_truncation_and_recurrent_policy_update_direction(
    tmp_path, compatibility_manifest_factory
):
    # Hand-computed two-transition return: truncation bootstraps, true termination does not;
    # neither may propagate rewards from the reset episode back across its boundary.
    reward = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    value = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    next_value = torch.tensor([[10.0, 10.0], [4.0, 4.0]])
    terminated = torch.tensor([[True, False], [True, True]])
    done = torch.ones(2, 2, dtype=torch.bool)
    advantage, returns = advantages(reward, value, next_value, terminated, done, 0.9, 0.95)
    torch.testing.assert_close(advantage, torch.tensor([[-1.0, 8.0], [97.0, 97.0]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 10.0], [100.0, 100.0]]))

    spec = short_spec(tmp_path, compatibility_manifest_factory, "gru")
    spec = replace(spec, training=replace(spec.training, epochs=1, sequence_length=4, burn_in=0))
    trainer = Trainer(spec, tmp_path / "run")
    try:
        batch, snapshots = trainer.collect()
        with torch.no_grad():
            mean, _ = replay(trainer.actor, batch["obs"], batch["reset"], snapshots[0])
            batch["raw"] = mean + 0.2
            batch["logp"] = Actor.log_probability(mean, trainer.log_std, batch["raw"])
        batch["advantage"].fill_(1)
        # Small SGD step permits a clean directional test of the actual PPO objective.
        trainer.optimizer = torch.optim.SGD(
            [*trainer.actor_parameters, *trainer.critic.parameters()], lr=1e-4
        )
        old = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
        trainer.optimize(batch, snapshots)
        mean_after, _ = replay(trainer.actor, batch["obs"], batch["reset"], snapshots[0])
        logp_after = Actor.log_probability(mean_after, trainer.log_std, batch["raw"])
        assert float((logp_after - batch["logp"]).mean().detach()) > 0
        assert any(
            not torch.equal(p, old[n])
            for n, p in trainer.actor.core.named_parameters(prefix="core")
        )
    finally:
        trainer.close()


class DelayedCueBody:
    """Eight decisions; the sign is observed only at reset and rewarded only at the end."""

    obs_dim, action_dim, control_dt = 2, 1, 0.01
    fingerprint, evidence = "delayed-cue-audit-v1", "software_fixture_only"
    obs_schema = [{"name": "cue_then_response_flag", "shape": [2], "scale": 1.0}]
    action_names = ["recall"]

    def __init__(self, spec):
        self.config = spec

    def reset(self, seed, split):
        self.case = {"seed": seed, "split": split}
        self.target = 1.0 if np.random.default_rng(seed).integers(2) else -1.0
        self.steps = 0
        return np.array([self.target, 0], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        done = self.steps == 8
        reward = 1 - 0.25 * (float(action[0]) - self.target) ** 2 if done else 0.0
        info = {
            "steps": self.steps,
            "success": done and abs(float(action[0]) - self.target) < 0.25,
            "score": reward,
            "episode_return": reward,
            "numerical_failure": False,
        }
        # At the response decision, both target signs have IDENTICAL observations.
        return np.array([0, self.steps == 7], dtype=np.float32), reward, done, False, info

    def state_dict(self):
        return {"case": self.case, "target": self.target, "steps": self.steps}

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def close(self):
        pass


def test_production_ppo_learns_delayed_cue_while_stateless_control_cannot(
    tmp_path, compatibility_manifest_factory, monkeypatch
):
    from connectome_body.compatibility import training
    from connectome_body.compatibility.config import AdapterConfig
    from connectome_body.compatibility.evaluation import evaluate

    monkeypatch.setattr(training, "make_body", DelayedCueBody)
    scores = {}
    for kind in ["gru", "adapter_only"]:
        spec = short_spec(tmp_path / kind / "inputs", compatibility_manifest_factory, kind)
        spec = replace(
            spec,
            controller=replace(
                spec.controller,
                adapter=AdapterConfig(family="mlp_linear", budget=128, channels=4),
                substrate=replace(
                    spec.controller.substrate, total_budget=2000 if kind == "gru" else None
                ),
            ),
            training=replace(
                spec.training,
                interactions=16384,
                num_envs=4,
                rollout_steps=64,
                sequence_length=8,
                burn_in=8,
                epochs=2,
                learning_rate=0.003,
                eval_every=16384,
                eval_episodes=20,
                test_episodes=100,
            ),
        )
        trainer = Trainer(spec, tmp_path / kind)
        try:
            result = trainer.run()
            assert result["status"] == "complete"
            # Fresh held-out episodes, final policy: no cherry-picking a learning-curve peak.
            scores[kind] = evaluate(trainer.actor, trainer.eval_env, 98173, 200, "test")
        finally:
            trainer.close()
    print({kind: {k: r[k] for k in ["mean_score", "success_rate"]} for kind, r in scores.items()})
    assert scores["gru"]["mean_score"] > 0.90
    assert scores["gru"]["success_rate"] > 0.80
    assert scores["adapter_only"]["mean_score"] < 0.80
