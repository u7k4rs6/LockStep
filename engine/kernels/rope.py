"""Rotary position embedding, elementwise."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry


@triton.jit
def _rope_kernel(
    X,
    Cos,
    Sin,
    Out,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_ct,
    HALF,
    BLOCK_PAIRS: tl.constexpr,
):
    """out[i] = x[i]*c - x[i+HALF]*s;  out[i+HALF] = x[i+HALF]*c + x[i]*s."""
    token = tl.program_id(0)
    head = tl.program_id(1)

    offs = tl.arange(0, BLOCK_PAIRS)
    mask = offs < HALF

    base_in = X + token * stride_xt + head * stride_xh
    base_out = Out + token * stride_ot + head * stride_oh

    lo = tl.load(base_in + offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
    hi = tl.load(base_in + (offs + HALF) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(Cos + token * stride_ct + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(Sin + token * stride_ct + offs, mask=mask, other=0.0).to(tl.float32)

    out_lo = lo * c - hi * s
    out_hi = hi * c + lo * s

    tl.store(
        base_out + offs * stride_od,
        out_lo.to(Out.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        base_out + (offs + HALF) * stride_od,
        out_hi.to(Out.dtype.element_ty),
        mask=mask,
    )


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: str = "rope",
) -> torch.Tensor:
    """Rotate `x` of shape [tokens, heads, head_dim] in place-equivalent form."""
    cfg = registry.get(config)
    tokens, heads, head_dim = x.shape
    half = head_dim // 2
    assert head_dim % 2 == 0, "head_dim must be even to have rotation pairs"
    assert half <= cfg.constants["BLOCK_PAIRS"], (
        f"rope pins BLOCK_PAIRS={cfg.constants['BLOCK_PAIRS']} but head_dim/2 is {half}"
    )
    assert cos.shape == (tokens, half) and sin.shape == (tokens, half)

    out = torch.empty_like(x)
    _rope_kernel[(tokens, heads)](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cos.stride(0),
        half,
        BLOCK_PAIRS=cfg.constants["BLOCK_PAIRS"],
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out
