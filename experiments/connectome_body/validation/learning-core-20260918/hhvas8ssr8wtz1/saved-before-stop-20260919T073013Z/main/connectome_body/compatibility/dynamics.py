"""Sparse recurrent dynamics with optional learning on existing edges only."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DynamicsConfig


class EdgeMultiply(torch.autograd.Function):
    """CSR forward/backward without an N x N dense matrix or B x M edge activation."""

    @staticmethod
    def forward(ctx, state, values, row_ptr, col, rows, t_ptr, t_col, t_order, chunk):
        count = state.shape[1]
        matrix = torch.sparse_csr_tensor(row_ptr, col, values, size=(count, count))
        ctx.save_for_backward(state, values, col, rows, t_ptr, t_col, t_order)
        ctx.chunk = chunk
        return torch.sparse.mm(matrix, state.T).T.contiguous()

    @staticmethod
    def backward(ctx, gradient):
        state, values, col, rows, t_ptr, t_col, t_order = ctx.saved_tensors
        d_state = d_values = None
        if ctx.needs_input_grad[0]:
            matrix = torch.sparse_csr_tensor(
                t_ptr, t_col, values[t_order], size=(state.shape[1], state.shape[1])
            )
            d_state = torch.sparse.mm(matrix, gradient.T).T.contiguous()
        if ctx.needs_input_grad[1]:
            d_values = torch.empty_like(values)
            for start in range(0, len(values), ctx.chunk):
                end = min(start + ctx.chunk, len(values))
                d_values[start:end] = (gradient[:, rows[start:end]] * state[:, col[start:end]]).sum(
                    0
                )
        return d_state, d_values, None, None, None, None, None, None, None


class SparseRateCore(nn.Module):
    """One learnable magnitude per permitted edge; source signs remain fixed.

    Both frozen and plastic conditions use incoming/outgoing strength
    normalization. For plastic weights it is differentiable and recomputed,
    preserving the operator-norm bound throughout training.
    """

    def __init__(self, count, src, dst, magnitude, signs, config=DynamicsConfig(), plastic=False):
        super().__init__()
        config.validate()
        self.count, self.config = count, config
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        if count < 1 or src.ndim != 1 or not len(src) or src.shape != dst.shape:
            raise ValueError("Invalid sparse topology")
        if min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= count:
            raise ValueError("Edge endpoint outside the neural state")
        if np.any(src == dst) or len(np.unique(src * count + dst)) != len(src):
            raise ValueError("Canonical dynamics require unique non-autapse edges")
        order = np.lexsort((src, dst))
        src, dst = src[order], dst[order]
        magnitude = np.asarray(magnitude, dtype=np.float32)[order]
        signs = np.asarray(signs)
        if signs.shape != (count,) or not np.isin(signs, [-1, 1]).all():
            raise ValueError("Dynamical source signs must be known +/-1")
        if (
            magnitude.shape != src.shape
            or not np.isfinite(magnitude).all()
            or np.any(magnitude <= 0)
        ):
            raise ValueError("Initial edge magnitudes must be finite and positive")
        t_order = np.lexsort((dst, src))
        arrays = {
            "src": src,
            "dst": dst,
            "row_ptr": np.r_[0, np.cumsum(np.bincount(dst, minlength=count))],
            "transpose_ptr": np.r_[0, np.cumsum(np.bincount(src, minlength=count))],
            "transpose_col": dst[t_order],
            "transpose_order": t_order,
        }
        for name, array in arrays.items():
            self.register_buffer(name, torch.from_numpy(array.copy()).long(), persistent=False)
        self.register_buffer(
            "source_signs", torch.as_tensor(signs.copy(), dtype=torch.float32), persistent=False
        )
        initial = torch.from_numpy(magnitude.copy())
        self.register_buffer("initial_magnitudes", initial, persistent=False)
        if plastic:
            # Stable inverse softplus, including counts too large for exp(x).
            self.raw_magnitudes = nn.Parameter(initial + torch.log(-torch.expm1(-initial)))
        else:
            self.register_parameter("raw_magnitudes", None)
        self.register_buffer("frozen_values", self._normalize(initial), persistent=False)
        self.alpha = -math.expm1(-config.control_dt / (config.substeps * config.tau_seconds))

    def _normalize(self, magnitude):
        incoming = magnitude.new_zeros(self.count).index_add(0, self.dst, magnitude)
        outgoing = magnitude.new_zeros(self.count).index_add(0, self.src, magnitude)
        # Positive magnitudes ensure the indexed endpoint strengths are positive.
        denominator = (incoming[self.dst] * outgoing[self.src]).sqrt().clamp_min(1e-20)
        return magnitude / denominator * self.source_signs[self.src]

    def edge_values(self):
        if self.raw_magnitudes is None:
            return self.frozen_values
        return self._normalize(F.softplus(self.raw_magnitudes).clamp_min(1e-12))

    def multiply(self, state, values=None):
        if state.device.type == "mps":
            raise RuntimeError(
                "Plastic CSR dynamics require CPU or CUDA; no silent device fallback"
            )
        return EdgeMultiply.apply(
            state,
            self.edge_values() if values is None else values,
            self.row_ptr,
            self.src,
            self.dst,
            self.transpose_ptr,
            self.transpose_col,
            self.transpose_order,
            self.config.gradient_edge_chunk,
        )

    def reset(self, batch_size):
        return self.initial_magnitudes.new_zeros((batch_size, self.count))

    def forward(self, state, neural_input, *, values=None, lesion_mask=None, remove_edges=False):
        if state.shape != neural_input.shape or state.shape[-1] != self.count:
            raise ValueError("State and neural input must share [batch, neurons] shape")
        values = self.edge_values() if values is None else values
        if lesion_mask is not None:
            if lesion_mask.shape != (self.count,) or lesion_mask.dtype != torch.bool:
                raise ValueError("Lesion mask must be one boolean per neuron")
            keep = (~lesion_mask).to(state)
            state, neural_input = state * keep, neural_input * keep
        for _ in range(self.config.substeps):
            recurrent = torch.zeros_like(state) if remove_edges else self.multiply(state, values)
            state = (1 - self.alpha) * state + self.alpha * torch.tanh(
                self.config.gain * recurrent + self.config.input_gain * neural_input
            )
            if lesion_mask is not None:
                state = state * keep
        return state
