"""RMSNorm, one CTA per row."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """y[row] = x[row] * rsqrt(mean(x^2) + eps) * w, in fp32 throughout."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / N
    scale = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale * w

    tl.store(Y + row * stride_ym + offs, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    config: str = "rmsnorm.hidden",
) -> torch.Tensor:
    """RMSNorm over the last dimension."""
    cfg = registry.get(config)
    n = x.shape[-1]
    assert n <= cfg.constants["BLOCK_SIZE"], (
        f"{config} pins BLOCK_SIZE={cfg.constants['BLOCK_SIZE']} but the row is "
        f"{n} wide; one CTA per row means the row must fit one tile"
    )
    assert weight.shape == (n,)

    x2d = x.reshape(-1, n)
    out = torch.empty_like(x2d)

    _rmsnorm_kernel[(x2d.shape[0],)](
        x2d,
        weight,
        out,
        x2d.stride(0),
        out.stride(0),
        n,
        eps,
        BLOCK_SIZE=cfg.constants["BLOCK_SIZE"],
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out.reshape(x.shape)
