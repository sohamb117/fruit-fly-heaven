"""Bounded, read-only CUDA audit of checkpoint copies through the real PPO optimizer.

An artificial terminal-only advantage isolates backpropagation through time.
SGD has zero learning rate: no policy, checkpoint, or active run is modified.
This is a gradient diagnostic, not an embodied performance measurement.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch

    from connectome_body.compatibility.controller import Controller
    from connectome_body.compatibility.learning_study import read_plan
    from connectome_body.compatibility.run_config import RunSpec
    from connectome_body.compatibility.training import Trainer, source_identity
    from connectome_body.policy import Actor, Critic
    from connectome_body.util import atomic_json, digest_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--executions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(73021)
    plan = read_plan(args.plan)
    rows = {r["execution_id"]: r for r in plan["conditions"]}
    records = []
    started = time.monotonic()
    for eid in args.executions:
        row = rows[eid]
        run = Path(args.plan).resolve().parent / "runs" / eid
        spec = RunSpec.from_dict(json.loads((run / "config.json").read_text()))
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["code_fingerprint"] != source_identity():
            raise ValueError("Audit must use the deployed training implementation")
        trainer = Trainer.__new__(Trainer)  # Isolated actor/critic; no running Trainer touched.
        trainer.device = torch.device("cuda")
        trainer.config = replace(
            spec,
            training=replace(
                spec.training,
                sequence_length=16,
                burn_in=0,
                epochs=1,
                num_envs=2,
            ),
        )
        trainer.actor = Controller(
            manifest["observation_dim"], manifest["action_dim"], spec.controller
        ).to(trainer.device)
        # Read through one file descriptor so atomic replacement cannot mix checkpoints.
        with (run / "latest.pt").open("rb") as f:
            saved = torch.load(f, map_location="cpu", weights_only=False)
        trainer.actor.load_state_dict(saved["actor"])
        checkpoint_step = saved["interactions"]
        del saved
        trainer.critic = Critic(manifest["observation_dim"], spec.training.critic_width).to(
            trainer.device
        )
        trainer.actor_parameters = [p for p in trainer.actor.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.SGD(
            [*trainer.actor_parameters, *trainer.critic.parameters()],
            lr=0,
        )
        trainer.log_std = torch.full(
            (manifest["action_dim"],), spec.training.exploration_log_std, device=trainer.device
        )
        trainer.optimizer_steps = trainer.optimized_decisions = trainer.burnin_decisions = 0
        with np.load(Path(plan["datasets"][row["task"]]["path"]) / "trajectories.npz") as data:
            obs = torch.from_numpy(data["train_obs"][:2, :16].copy()).transpose(0, 1).contiguous()

        def batch_for(reset_step=None):
            resets = torch.zeros(16, 2, dtype=torch.bool)
            resets[0] = True
            if reset_step is not None:
                resets[reset_step] = True
            state = trainer.actor.reset(2)
            snapshots = {0: state.cpu().clone()}
            means = []
            with torch.no_grad():
                context = trainer.actor.context()
                for x, reset in zip(obs, resets, strict=True):
                    mean, state = trainer.actor(
                        x.cuda(), state, context, reset=reset.cuda(), squash=False
                    )
                    means.append(mean)
            means = torch.stack(means)
            raw = means + 0.3
            advantage = torch.zeros(16, 2)
            advantage[-1] = 1
            batch = {
                "obs": obs,
                "reset": resets,
                "raw": raw.cpu(),
                "logp": Actor.log_probability(means, trainer.log_std, raw).cpu(),
                "advantage": advantage,
                "return": torch.zeros(16, 2),
            }
            return batch, snapshots

        def probe(reset_step=None, detach=False):
            batch, snapshots = batch_for(reset_step)
            retained, actual_logp = [], []
            original = trainer.actor.forward

            def wrapped(x, state, *a, **kw):
                if detach:
                    state = state.detach()
                mean, state = original(x, state, *a, **kw)
                actual_logp.append(
                    Actor.log_probability(
                        mean, trainer.log_std, batch["raw"][len(actual_logp)].cuda()
                    )
                    .detach()
                    .cpu()
                )
                return mean, state

            def remember(module, inputs, output):
                output.retain_grad()
                retained.append(output)

            trainer.actor.forward = wrapped
            handle = trainer.actor.encoder.register_forward_hook(remember)
            try:
                trainer.optimize(batch, snapshots)
            finally:
                handle.remove()
                trainer.actor.forward = original
            grads = [0.0 if z.grad is None else float(z.grad.abs().sum()) for z in retained]
            error = float((torch.stack(actual_logp) - batch["logp"]).abs().max())
            assert error < 2e-5, error
            return {"encoder_output_gradient_l1_by_time": grads, "max_replay_logp_error": error}

        intact, cut, reset = probe(), probe(detach=True), probe(reset_step=8)
        assert intact["encoder_output_gradient_l1_by_time"][0] > 0
        assert max(cut["encoder_output_gradient_l1_by_time"][:-1]) == 0
        assert max(reset["encoder_output_gradient_l1_by_time"][:8]) == 0
        assert reset["encoder_output_gradient_l1_by_time"][8] > 0
        record = {
            "execution": eid,
            "source": row["source"],
            "task": row["task"],
            "checkpoint_interactions": checkpoint_step,
            "source_identity": source_identity(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "sequence_length": 16,
            "control_dt": spec.controller.substrate.dynamics.control_dt,
            "intact": intact,
            "detached_negative_control": cut,
            "episode_reset": reset,
            "passed": True,
        }
        records.append(record)
        atomic_json(args.output, {"complete": False, "records": records})
        print(json.dumps(record), flush=True)
        trainer = None
        torch.cuda.empty_cache()
    atomic_json(
        args.output,
        {
            "complete": True,
            "records": records,
            "wall_seconds": time.monotonic() - started,
            "script_sha256": digest_file(__file__),
        },
    )


if __name__ == "__main__":
    main()
