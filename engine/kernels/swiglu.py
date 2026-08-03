"""SwiGLU activation, elementwise.

Not named in the architecture doc's kernel list, because it carries no reduction
and therefore no invariance risk: out[i] depends on gate[i] and up[i] and nothing
else, so it is invariant under any batching by construction.

It is written in Triton anyway so that the fused activation is one launch from
the pinned registry rather than three torch launches, and so that the static grep
over engine/ covers the whole activation path rather than most of it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry


@triton.jit
def _swiglu_kernel(
    Gate,
    Up,
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = silu(gate) * up = gate * sigmoid(gate) * up, computed in fp32."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)

    out = gate * tl.sigmoid(gate) * up
    tl.store(Out + offs, out.to(Out.dtype.element_ty), mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    cfg = registry.get("swiglu")
    assert gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()

    out = torch.empty_like(gate)
    n = gate.numel()
    block = cfg.constants["BLOCK_SIZE"]
    _swiglu_kernel[(triton.cdiv(n, block),)](
        gate,
        up,
        out,
        n,
        BLOCK_SIZE=block,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out
