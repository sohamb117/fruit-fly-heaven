"""One reset/step interface for canonical real and randomized frozen graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..graphs import Graph, neuron_signs, normalized_matrix, transform_graph
from ..policy import _FixedSparseMultiply
from .features import structural_features
from .nulls import weighted_degree_null


@dataclass(frozen=True)
class RateConfig:
    gain: float = 0.9
    tau_seconds: float = 0.02
    control_dt: float = 0.001
    substeps: int = 2
    input_gain: float = 1.0
    sign_mode: str = "random_dale"
    inhibitory_fraction: float = 0.2
    weight_transform: str = "log1p"

    def validate(self):
        if (
            min(self.tau_seconds, self.control_dt, self.input_gain, self.substeps) <= 0
            or self.gain < 0
        ):
            raise ValueError("Invalid rate dynamics")
        if self.sign_mode not in ("random_dale", "annotated", "available"):
            raise ValueError("Unknown sign rule")
        if not 0 <= self.inhibitory_fraction <= 1 or self.weight_transform not in (
            "log1p",
            "count",
        ):
            raise ValueError("Invalid sign fraction or weight transform")


class RateSubstrate(nn.Module):
    def __init__(self, graph: Graph, variant="real", config=RateConfig(), seed=0, backend="cpu"):
        super().__init__()
        config.validate()
        mode = config.sign_mode
        if mode == "available":
            mode = "annotated" if np.any(graph.signs) else "random_dale"
        signs = neuron_signs(graph, mode, config.inhibitory_fraction, seed)
        if variant == "degree_shuffled":
            src, dst, weight, self.null_report = weighted_degree_null(graph, signs, seed)
        else:
            src, dst, weight, self.null_report = transform_graph(graph, variant, signs, seed)
        self.features, self.feature_report = structural_features(graph.n, src, dst, weight)
        matrix = normalized_matrix(graph.n, src, dst, weight, signs, config.weight_transform)
        self.metal = None
        if backend == "mps":
            from ..metal_sparse import FrozenMetalCSR

            self.metal = FrozenMetalCSR(matrix)
        else:
            for name, value in (("matrix", matrix), ("transpose", matrix.T.tocsr())):
                self.register_buffer(
                    name,
                    torch.sparse_csr_tensor(
                        torch.from_numpy(value.indptr.astype(np.int64)),
                        torch.from_numpy(value.indices.astype(np.int64)),
                        torch.from_numpy(value.data),
                        size=value.shape,
                    ),
                    persistent=False,
                )
        self.n, self.config = graph.n, config
        self.alpha = -math.expm1(-config.control_dt / (config.substeps * config.tau_seconds))
        self.report = {
            "graph_fingerprint": graph.fingerprint,
            "neurons": graph.n,
            "edges": len(src),
            "variant": variant,
            "null": self.null_report,
            "features": self.feature_report,
            "actual_inhibitory_fraction": float(np.mean(signs < 0)),
            "resolved_sign_mode": mode,
            "normalization": "D_in^-1/2 T(A)^T D_out^-1/2 S; operator norm at most 1",
            "sparse_backend": "metal_csr" if self.metal is not None else "pytorch_csr",
        }

    def reset(self, batch_size):
        device = self.metal.row.device if self.metal is not None else self.matrix.device
        return torch.zeros(batch_size, self.n, device=device)

    def step(self, state, neural_input, lesion=False):
        for _ in range(self.config.substeps):
            if lesion:
                recurrent = torch.zeros_like(state)
            elif self.metal is not None:
                recurrent = self.metal(state)
            else:
                recurrent = _FixedSparseMultiply.apply(state, self.matrix, self.transpose)
            state = (1 - self.alpha) * state + self.alpha * torch.tanh(
                self.config.gain * recurrent + self.config.input_gain * neural_input
            )
        return state


class NullSubstrate(nn.Module):
    """Stateless identity bottleneck: it has neither recurrence nor trainable weights."""

    def reset(self, batch_size):
        return torch.empty(batch_size, 0)

    def step(self, state, neural_input):
        return neural_input
