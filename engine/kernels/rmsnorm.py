"""RMSNorm, one CTA per row.

docs/02-technical-architecture.md section 4.2: "RMSNorm and softmax as one CTA
per row."

One CTA per row is the whole design. The sum of squares is an in-CTA tree
reduction over a fixed-width tile, so it never crosses a block boundary, never
touches an atomic, and never depends on how many other rows are resident. Two
rows in flight and two thousand rows in flight produce the same bits for each.

Qwen3 also applies RMSNorm per attention head to Q and K (`q_norm`, `k_norm`)
over head_dim rather than hidden_size. That is the same kernel at a smaller
pinned tile, which is why the registry carries two entries.
"""

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

    # Masked lanes loaded 0.0, which contributes exactly zero to the sum, so the
    # result does not depend on BLOCK_SIZE exceeding N.
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
    """RMSNorm over the last dimension.

    Deviation from HF, deliberate and recorded: `Qwen3RMSNorm` rounds to the
    input dtype *before* multiplying by the weight, so it takes an extra fp16
    rounding step in the middle. This keeps fp32 through the weight multiply and
    rounds once at the end. That is strictly closer to the fp64 reference, which
    is what F1 measures; HF is a sanity check and not ground truth
    (architecture doc section 7).
    """
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
