"""Learned sparse structural queries with neuron-count-independent parameters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..util import seed_for


@dataclass
class PortMap:
    input_nodes: torch.Tensor
    input_weights: torch.Tensor
    output_nodes: torch.Tensor
    output_weights: torch.Tensor


class StructuralPorts(nn.Module):
    def __init__(self, features, channels=16, support=256, seed=0, temperature=1.0):
        super().__init__()
        n, f = features.shape
        if n < 2 * channels or support < 1 or temperature <= 0:
            raise ValueError("Need >=2K neurons, positive support and temperature")
        self.n, self.channels = n, channels
        self.support, self.temperature = min(support, n // 2), temperature
        rng = np.random.default_rng(seed_for(seed, "structural-port-partition"))
        order = rng.permutation(n)
        self.register_buffer("input_pool", torch.from_numpy(order[: n // 2]), persistent=False)
        self.register_buffer("output_pool", torch.from_numpy(order[n // 2 :]), persistent=False)
        self.register_buffer("features", torch.as_tensor(features.copy()), persistent=False)
        # A disruption stays within output neurons, so it cannot create a bypass.
        perm = np.arange(n)
        perm[order[n // 2 :]] = rng.permutation(order[n // 2 :])
        self.register_buffer("output_permutation", torch.from_numpy(perm), persistent=False)
        self.input_queries = nn.Parameter(torch.randn(channels, f) / math.sqrt(f))
        self.output_queries = nn.Parameter(torch.randn(channels, f) / math.sqrt(f))
        self.fingerprint = hashlib.sha256(features.tobytes() + order.tobytes()).hexdigest()

    def _allocate(self, queries, pool):
        scores = queries @ self.features[pool].T / math.sqrt(self.features.shape[1])
        scores, local = scores.topk(self.support, dim=-1, sorted=True)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        # Unit L2 columns control gain independently of population size.
        weights = weights / weights.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        return pool[local], weights

    def compile(self):
        i, iw = self._allocate(self.input_queries, self.input_pool)
        o, ow = self._allocate(self.output_queries, self.output_pool)
        return PortMap(i, iw, o, ow)

    def inject(self, latent, ports):
        values = (latent.unsqueeze(-1) * ports.input_weights).flatten(1)
        return latent.new_zeros((len(latent), self.n)).index_add(
            1, ports.input_nodes.flatten(), values
        )

    def read(self, state, ports, disrupted=False):
        nodes = ports.output_nodes
        if disrupted:
            nodes = self.output_permutation[nodes]
        return (state[:, nodes] * ports.output_weights).sum(-1)

    def overlap(self, ports):
        # Both pools partition neurons, independent of learned query values.
        return int(torch.isin(ports.input_nodes, ports.output_nodes).sum())
