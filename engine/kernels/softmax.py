"""Row log-softmax, one CTA per row."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry


@triton.jit
def _log_softmax_kernel(
    X,
    Y,
    stride_xm,
    stride_ym,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X + row * stride_xm
    y_row = Y + row * stride_ym

    num_tiles = tl.cdiv(N, BLOCK_SIZE)

    row_max = tl.full((), float("-inf"), dtype=tl.float32)
    for t in range(0, num_tiles):
        offs = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    total = tl.zeros((), dtype=tl.float32)
    for t in range(0, num_tiles):
        offs = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        total += tl.sum(tl.exp(x - row_max), axis=0)

    log_denom = row_max + tl.log(total)

    for t in range(0, num_tiles):
        offs = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + offs, mask=offs < N, other=0.0).to(tl.float32)
        tl.store(y_row + offs, (x - log_denom).to(Y.dtype.element_ty), mask=offs < N)


def log_softmax(x: torch.Tensor, config: str = "softmax.vocab") -> torch.Tensor:
    """Row-wise log-softmax over the last dimension, returned in fp32."""
    cfg = registry.get(config)
    n = x.shape[-1]
    x2d = x.reshape(-1, n)
    out = torch.empty(x2d.shape, dtype=torch.float32, device=x.device)

    _log_softmax_kernel[(x2d.shape[0],)](
        x2d,
        out,
        x2d.stride(0),
        out.stride(0),
        n,
        BLOCK_SIZE=cfg.constants["BLOCK_SIZE"],
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out.reshape(*x.shape[:-1], n)
