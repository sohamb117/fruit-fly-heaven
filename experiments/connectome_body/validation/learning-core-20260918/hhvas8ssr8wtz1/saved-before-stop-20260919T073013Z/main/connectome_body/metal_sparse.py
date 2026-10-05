"""Frozen CSR multiplication on Apple GPUs, without densification or CPU fallback.

Uses the public PyTorch 2.8 torch.mps.compile_shader interface. The analytic
backward multiplies by the cached transpose; no synapse weights are learned.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import nn

SHADER = """
#include <metal_stdlib>
using namespace metal;
kernel void csr_spmm(
    device float* output,
    device const int* row,
    device const int* column,
    device const float* weight,
    device const float* input,
    constant uint& n,
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint unit = index / 32;
    uint r = unit % n;
    uint batch = unit / n;
    float sum = 0.0f;
    for (int edge = row[r] + int(lane); edge < row[r + 1]; edge += 32) {
        sum += weight[edge] * input[batch * n + column[edge]];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[unit] = sum;
}
"""


@lru_cache(maxsize=1)
def shader():
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Apple MPS shader execution is unavailable; no CPU fallback")
    library = torch.mps.compile_shader(SHADER)
    if library.csr_spmm.thread_execution_width != 32:
        raise RuntimeError("This CSR kernel requires a 32-lane Apple GPU SIMD group")
    return library


def multiply(x, row, column, weight, n):
    if x.device.type != "mps" or x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != n:
        raise ValueError("Metal CSR expects a float32 [batch, neurons] MPS tensor")
    if x.numel() * 32 >= 2**32:
        raise ValueError("Metal dispatch exceeds the supported 32-bit grid")
    x = x.contiguous()
    output = torch.empty_like(x)
    shader().csr_spmm(output, row, column, weight, x, n, threads=x.numel() * 32, group_size=256)
    return output


class _MetalMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, operator):
        ctx.save_for_backward(
            operator.transpose_row, operator.transpose_column, operator.transpose_weight
        )
        ctx.n = operator.n
        return multiply(x, operator.row, operator.column, operator.weight, operator.n)

    @staticmethod
    def backward(ctx, gradient):
        row, column, weight = ctx.saved_tensors
        return multiply(gradient, row, column, weight, ctx.n), None


class FrozenMetalCSR(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        matrix = matrix.tocsr()
        if matrix.shape[0] != matrix.shape[1] or max(matrix.shape[0], matrix.nnz) >= 2**31:
            raise ValueError("Metal CSR requires a square graph fitting signed 32-bit indices")
        self.n = matrix.shape[0]
        for prefix, value in (("", matrix), ("transpose_", matrix.T.tocsr())):
            for name, array in (
                ("row", value.indptr.astype(np.int32)),
                ("column", value.indices.astype(np.int32)),
                ("weight", value.data.astype(np.float32)),
            ):
                self.register_buffer(
                    prefix + name, torch.from_numpy(array.copy()), persistent=False
                )

    def forward(self, x):
        return _MetalMultiply.apply(x, self)
