"""Fixed-BLOCK_K GEMM. No split-K, including on the 151936-wide lm_head.

docs/02-technical-architecture.md section 4.2: "Triton GEMMs with fixed BLOCK_K
and no split-K, including the roughly 151k-vocab lm_head, which is the most
tempting place to reach for split-K and the worst place to do it."

Section 5 explains why split-K is not a thing to fix but a thing to not do:
cuBLAS split-K combines partials via atomics or a heuristically chosen reduce
pass, and either choice is made from the runtime shape. Here, one program owns
one output tile and walks the entire K dimension itself, in ascending order, in
fixed BLOCK_K steps. Nothing accumulates across programs, so there is nothing to
combine and no order to get wrong.

The grid is a function of M and N, and M is batch-derived. That is a launch
dimension, not a config: it changes how many independent output tiles exist, not
the reduction any one of them performs. Every output element is the same
ascending fp32 sum over the same fixed K blocking regardless of how many rows are
in flight.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    Bias,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M, N] = A[M, K] @ B[N, K].T, accumulated in fp32.

    B is indexed as [N, K] because that is how torch.nn.Linear stores weights;
    no transpose materializes.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # The K loop. Ascending, BLOCK_K at a time, trip count ceil(K / BLOCK_K).
    # K is a weight-shape constant, so the trip count and the summation order are
    # identical for every call at this call site, whatever M happens to be.
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k0 * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        acc += tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)[None, :].to(tl.float32)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(C.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    config: str = "gemm.default",
) -> torch.Tensor:
    """y = x @ weight.T + bias, with the pinned config named by `config`.

    `config` is a literal at every call site. It is not derived from `x.shape`,
    and there is no code path here that would let it be.
    """
    cfg = registry.get(config)
    assert x.is_cuda and weight.is_cuda, "kernels under claim run on the pinned GPU"
    assert x.dtype == weight.dtype == torch.float16, "engine dtype is fp16"

    leading = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    m, k = x2d.shape
    n, k_w = weight.shape
    assert k == k_w, f"inner dimension mismatch: {k} vs {k_w}"

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    grid = (
        triton.cdiv(m, cfg.constants["BLOCK_M"]),
        triton.cdiv(n, cfg.constants["BLOCK_N"]),
    )
    _gemm_kernel[grid](
        x2d,
        weight,
        out,
        bias if bias is not None else x2d,  # unused when HAS_BIAS is false
        m,
        n,
        k,
        x2d.stride(0),
        x2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=cfg.constants["BLOCK_M"],
        BLOCK_N=cfg.constants["BLOCK_N"],
        BLOCK_K=cfg.constants["BLOCK_K"],
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out.reshape(*leading, n)
