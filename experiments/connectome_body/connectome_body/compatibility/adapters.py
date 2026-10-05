"""Seven stateless interface families with exact, non-padding parameter accounting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..adaptation.ports import PortMap
from ..util import digest_file, seed_for
from .config import AdapterConfig


def count_parameters(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


class LowRankLinear(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        if not 1 <= rank <= min(inputs, outputs):
            raise ValueError("Rank must not exceed either map dimension")
        self.left = nn.Linear(inputs, rank, bias=False)
        self.right = nn.Linear(rank, outputs)

    def forward(self, values):
        return self.right(self.left(values))


class SparseLinear(nn.Module):
    """Fixed fan-in per output; train only existing coefficients, never a dense mask."""

    def __init__(self, inputs, outputs, density, seed):
        super().__init__()
        self.inputs, self.outputs = inputs, outputs
        fan_in = max(1, math.ceil(inputs * density))
        rng = np.random.default_rng(seed)
        columns = np.stack([rng.choice(inputs, fan_in, replace=False) for _ in range(outputs)])
        self.register_buffer("columns", torch.from_numpy(columns), persistent=False)
        self.weight = nn.Parameter(torch.empty(outputs, fan_in))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.uniform_(self.weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, values):
        return (values[..., self.columns] * self.weight).sum(-1) + self.bias


def _dimensions(inputs, outputs, hidden, width, depth):
    return [inputs] + ([width] * depth if hidden else []) + [outputs]


def _map_cost(dimensions, density=None):
    return sum(
        b * ((a if density is None else max(1, math.ceil(a * density))) + 1)
        for a, b in zip(dimensions[:-1], dimensions[1:], strict=True)
    )


def _map(dimensions, seed, density=None):
    modules = []
    for index, (a, b) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        if index:
            modules.append(nn.Tanh())
        modules.append(
            nn.Linear(a, b)
            if density is None
            else SparseLinear(a, b, density, seed_for(seed, f"sparse-layer-{index}"))
        )
    return nn.Sequential(*modules)


def allocate_maps(obs_dim, action_dim, config: AdapterConfig, port_parameters=0, seed=0):
    config.validate()
    k = config.channels
    allowance = config.budget - port_parameters
    if allowance < 1 or min(obs_dim, action_dim) < 1:
        raise ValueError("Insufficient adapter budget or invalid observation/action dimensions")
    family, rank, width = config.family, None, None
    density = config.density if family == "sparse_structured" else None
    encoder_hidden = family in ("mlp_linear", "mlp_mlp", "sparse_structured", "anatomical")
    decoder_hidden = family in ("linear_mlp", "mlp_mlp", "sparse_structured", "anatomical")
    if family == "low_rank":
        maximum = min(obs_dim, action_dim, k)
        affordable = (allowance - k - action_dim) // (obs_dim + 2 * k + action_dim)
        rank = config.rank if config.rank is not None else min(maximum, affordable)
        if rank < 1 or rank > maximum:
            raise ValueError("No admissible low-rank adapter at this budget")
        encoder = LowRankLinear(obs_dim, k, rank)
        decoder = LowRankLinear(k, action_dim, rank)
    else:

        def dimensions(w):
            return (
                _dimensions(obs_dim, k, encoder_hidden, w, config.depth),
                _dimensions(k, action_dim, decoder_hidden, w, config.depth),
            )

        def cost(w):
            return sum(_map_cost(d, density) for d in dimensions(w))

        if encoder_hidden or decoder_hidden:
            if config.width is not None:
                width = config.width
            else:
                if cost(1) > allowance:
                    raise ValueError("No hidden width fits the declared adapter budget")
                low, high = 1, 2
                while cost(high) <= allowance:
                    low, high = high, 2 * high
                while high - low > 1:
                    middle = (low + high) // 2
                    if cost(middle) <= allowance:
                        low = middle
                    else:
                        high = middle
                width = low
        if cost(width) > allowance:
            raise ValueError("Requested adapter dimensions exceed the parameter ceiling")
        e_dims, d_dims = dimensions(width)
        encoder = _map(e_dims, seed_for(seed, "encoder"), density)
        decoder = _map(d_dims, seed_for(seed, "decoder"), density)
    for module in (encoder, decoder):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    actual = count_parameters(encoder) + count_parameters(decoder) + port_parameters
    if actual > config.budget:
        raise ValueError("Requested adapter rank exceeds the parameter ceiling")
    report = {
        "family": family,
        "budget": config.budget,
        "allocated": actual,
        "unused_budget": config.budget - actual,
        "width": width,
        "rank": rank,
        "encoder_depth": config.depth if encoder_hidden else 0,
        "decoder_depth": config.depth if decoder_hidden else 0,
        "channels": k,
        "density": density,
        "port_parameters": port_parameters,
        "encoder_parameters": count_parameters(encoder),
        "decoder_parameters": count_parameters(decoder),
        "stateless": True,
        "oracle": family == "anatomical",
    }
    return encoder, decoder, report


class AnatomicalPorts(nn.Module):
    """Explicitly privileged, fixed neuron groups; never silently infer homologies."""

    def __init__(self, graph, path, channels):
        super().__init__()
        value = json.loads(Path(path).read_text())
        if value.get("schema") != "anatomical-ports-v1":
            raise ValueError("Anatomical port file needs schema anatomical-ports-v1")
        if value.get("graph_fingerprint") != graph.fingerprint or not value.get("provenance"):
            raise ValueError(
                "Anatomical mapping needs matching graph identity and annotation provenance"
            )
        lookup = {str(node): index for index, node in enumerate(graph.node_ids)}
        supports = []
        for side in ("input", "output"):
            groups = value[f"{side}_groups"]
            if len(groups) != channels or any(not group for group in groups):
                raise ValueError("Every anatomical port requires a nonempty, declared neuron group")
            nodes = np.zeros((channels, max(map(len, groups))), dtype=np.int64)
            weights = np.zeros_like(nodes, dtype=np.float32)
            used = set()
            for index, group in enumerate(groups):
                if len(set(map(str, group))) != len(group):
                    raise ValueError("Duplicate neurons within one anatomical port")
                try:
                    indices = [lookup[str(node)] for node in group]
                except KeyError as error:
                    raise ValueError("Unknown neuron in anatomical mapping") from error
                nodes[index, : len(indices)] = indices
                weights[index, : len(indices)] = 1 / math.sqrt(len(indices))
                used.update(indices)
            supports.append(used)
            self.register_buffer(f"{side}_nodes", torch.from_numpy(nodes), persistent=False)
            self.register_buffer(f"{side}_weights", torch.from_numpy(weights), persistent=False)
        if supports[0] & supports[1]:
            raise ValueError("Anatomical input/output populations must be disjoint")
        self.n, self.fingerprint, self.provenance = graph.n, digest_file(path), value["provenance"]

    def compile(self):
        return PortMap(self.input_nodes, self.input_weights, self.output_nodes, self.output_weights)

    def inject(self, latent, ports):
        values = (latent.unsqueeze(-1) * ports.input_weights).flatten(1)
        return latent.new_zeros((len(latent), self.n)).index_add(
            1, ports.input_nodes.flatten(), values
        )

    def read(self, state, ports):
        return (state[:, ports.output_nodes] * ports.output_weights).sum(-1)
