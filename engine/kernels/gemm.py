"""Fixed-BLOCK_K GEMM. No split-K, including on the 151936-wide lm_head."""

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
    """C[M, N] = A[M, K] @ B[N, K].T, accumulated in fp32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
    """y = x @ weight.T + bias, with the pinned config named by `config`."""
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
        bias if bias is not None else x2d,
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
