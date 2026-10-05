"""Capacity-controlled adapters with no observation-to-decoder bypass.

The actor sees observations only through E. Fixed disjoint input/output ports
force graph actors to transmit information across at least one actual edge.
The critic is separate and never contributes features to the actor.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import RunConfig
from .graphs import Graph, neuron_signs, normalized_matrix, transform_graph
from .util import seed_for

BASELINES = ("adapter_only", "adapter_gru", "trainable_gru")
INTERVENTIONS = ("none", "silence", "reset", "no_recurrence", "permute_readout")


def parameter_count(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class HashPorts(nn.Module):
    """O(N) fixed storage and O(K) learned interface, independent of taxonomy."""

    def __init__(self, n: int, channels: int, seed: int):
        super().__init__()
        if n < 2 * channels:
            raise ValueError(f"Need >= {2 * channels} neurons for disjoint, covered ports; got {n}")
        rng = np.random.default_rng(seed_for(seed, "ports"))
        order = rng.permutation(n)
        ni = n // 2
        input_nodes, output_nodes = order[:ni], order[ni:]
        input_bucket = rng.permutation(np.arange(ni) % channels)
        output_bucket = rng.permutation(np.arange(n - ni) % channels)
        input_sign = rng.choice([-1.0, 1.0], ni).astype(np.float32)
        output_sign = rng.choice([-1.0, 1.0], n - ni).astype(np.float32)
        output_sign /= np.sqrt(np.bincount(output_bucket, minlength=channels)[output_bucket])
        arrays = dict(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            input_sign=input_sign,
            output_sign=output_sign,
            output_permutation=rng.permutation(output_nodes),
        )
        digest = hashlib.sha256()
        for name, array in arrays.items():
            self.register_buffer(name, torch.from_numpy(array), persistent=False)
            digest.update(array.tobytes())
        self.fingerprint = digest.hexdigest()
        self.n, self.channels = n, channels

    def inject(self, z):
        drive = z.new_zeros((len(z), self.n))
        return drive.index_copy(1, self.input_nodes, z[:, self.input_bucket] * self.input_sign)

    def read(self, state, permute=False):
        nodes = self.output_permutation if permute else self.output_nodes
        values = state[:, nodes] * self.output_sign
        return values.new_zeros((len(state), self.channels)).index_add(
            1, self.output_bucket, values
        )


class _FixedSparseMultiply(torch.autograd.Function):
    """Cache both CSR orientations instead of rebuilding W.T at every BPTT step."""

    @staticmethod
    def forward(ctx, state, matrix, transpose):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(matrix, state.T.contiguous()).T

    @staticmethod
    def backward(ctx, gradient):
        (transpose,) = ctx.saved_tensors
        state_gradient = torch.sparse.mm(transpose, gradient.T.contiguous()).T
        return state_gradient, None, None


class FrozenConnectome(nn.Module):
    def __init__(self, graph: Graph, config: RunConfig):
        super().__init__()
        d = config.dynamics
        signs = neuron_signs(graph, d.sign_mode, d.inhibitory_fraction, config.substrate_seed)
        src, dst, weights, self.null_report = transform_graph(
            graph, config.substrate, signs, config.substrate_seed
        )
        matrix = normalized_matrix(graph.n, src, dst, weights, signs, d.weight_transform)
        for name, orientation in (("matrix", matrix), ("transpose", matrix.T.tocsr())):
            self.register_buffer(
                name,
                torch.sparse_csr_tensor(
                    torch.from_numpy(orientation.indptr.astype(np.int64)),
                    torch.from_numpy(orientation.indices.astype(np.int64)),
                    torch.from_numpy(orientation.data),
                    size=orientation.shape,
                ),
                persistent=False,
            )
        self.ports = HashPorts(graph.n, d.channels, config.substrate_seed)
        self.n, self.substeps = graph.n, d.substeps
        self.alpha = -math.expm1(-config.body.control_dt / (d.substeps * d.tau_seconds))
        self.gain, self.input_gain = d.gain, d.input_gain
        self.graph_fingerprint = graph.fingerprint

    def forward(self, z, state, intervention="none"):
        if intervention == "reset":
            state = torch.zeros_like(state)
        drive = self.ports.inject(z) * self.input_gain
        for _ in range(self.substeps):
            recurrence = (
                torch.zeros_like(state)
                if intervention == "no_recurrence"
                else _FixedSparseMultiply.apply(state, self.matrix, self.transpose)
            )
            state = (1 - self.alpha) * state + self.alpha * torch.tanh(
                self.gain * recurrence + drive
            )
        latent = self.ports.read(state, permute=intervention == "permute_readout")
        return latent, state


class Actor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, config: RunConfig, graph: Graph | None = None
    ):
        super().__init__()
        self.kind = config.substrate
        self.channels = k = config.dynamics.channels
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.brain = None
        self.gru = None
        if self.kind == "trainable_gru":
            # A conventional observation -> GRU -> action policy. The recurrent
            # width, not a port bottleneck, uses the entire actor budget.
            h = 1
            while 3 * (h + 1) * (obs_dim + h + 3) + action_dim * (h + 3) <= config.adapter_budget:
                h += 1
            self.encoder = nn.Identity()
            self.gru = nn.GRUCell(obs_dim, h)
            self.decoder = nn.Linear(h, action_dim)
            self.hidden_width = self.state_dim = h
        else:
            # Two one-hidden-layer maps. All biases and action log-std count.
            overhead = k + 2 * action_dim
            if self.kind == "adapter_gru":
                overhead += 3 * k * (2 * k + 2)
            width = (config.adapter_budget - overhead) // (obs_dim + 2 * k + action_dim + 2)
            if width < 1:
                raise ValueError("Actor budget is too small for this interface and action space")
            self.hidden_width = int(width)
            self.encoder = nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(), nn.Linear(width, k))
            self.decoder = nn.Sequential(
                nn.Linear(k, width), nn.Tanh(), nn.Linear(width, action_dim)
            )
            if self.kind == "adapter_gru":
                self.gru = nn.GRUCell(k, k)
                self.state_dim = k
            elif self.kind == "adapter_only":
                self.state_dim = 0
            else:
                if graph is None:
                    raise ValueError("No graph; refusing a synthetic substitute")
                self.brain = FrozenConnectome(graph, config)
                self.state_dim = graph.n
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.reset_parameters()
        self.num_parameters = parameter_count(self)
        if self.num_parameters > config.adapter_budget:
            raise ValueError(
                f"Minimum actor exceeds budget: {self.num_parameters} > {config.adapter_budget}"
            )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        last = self.decoder if isinstance(self.decoder, nn.Linear) else self.decoder[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def initial_state(self, batch: int):
        return self.log_std.new_zeros((batch, self.state_dim))

    def forward(self, obs, state, reset=None, intervention="none"):
        if intervention not in INTERVENTIONS:
            raise ValueError("Unknown intervention")
        if reset is not None:
            state = state * (~reset.bool()).to(state.dtype).unsqueeze(-1)
        z = self.encoder(obs)
        if self.brain is not None:
            latent, state = self.brain(z, state, intervention)
        elif self.gru is not None:
            if intervention in ("reset", "no_recurrence"):
                state = torch.zeros_like(state)
            state = self.gru(z, state)
            latent = state
        else:
            latent = z
        if intervention == "silence":
            latent = torch.zeros_like(latent)
        return self.decoder(latent), self.log_std.clamp(-5, 1).expand(len(obs), -1), state

    @staticmethod
    def log_probability(mean, log_std, raw_action):
        normal = torch.distributions.Normal(mean, log_std.exp())
        # Stable log |d tanh(u)/du|; raw_action is stored, no atanh(+-1).
        log_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        return (normal.log_prob(raw_action) - log_jacobian).sum(-1)

    def sample(self, obs, state, reset=None):
        mean, log_std, state = self(obs, state, reset)
        raw = mean + log_std.exp() * torch.randn_like(mean)
        return torch.tanh(raw), raw, self.log_probability(mean, log_std, raw), state


class Critic(nn.Module):
    def __init__(self, obs_dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
