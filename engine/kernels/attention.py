"""Fixed-split attention. One kernel family for prefill and decode.

docs/02-technical-architecture.md section 4.2, the minimal design that preserves
chunk invariance:

    One attention kernel family. Constant KV tile (128). Fixed split size (512),
    never a fixed split count. Partials combined by a single CTA in ascending
    split order, accumulated in fp32.

Why one family and not a fast prefill kernel plus a split decode kernel: chunk
invariance (I2) requires that a token's output be identical whether it was
produced by a prefill pass or by a decode step. Two kernels with two reduction
structures cannot satisfy that, whatever their individual merits. The split
structure is therefore paid on prefill as well, and its memory cost for the
partials is the price of the invariant rather than an oversight.

Why split *size* and never split *count*: a fixed count divides the KV length by
a constant, and any implementation that picks the count from a batch-max sequence
length has smuggled a batch-derived quantity into a reduction (section 5, "Fixed
split count"). Here the number of splits is ceil(kv_len / 512) where kv_len is
this request's own KV length, which section 3 explicitly permits.

Two things about the launch grid, because they are what a skeptical reader checks
first:

  * The grid's split dimension is sized by the largest split count in the launch.
    Programs whose split lies past their own sequence's KV write an empty partial
    and return. This changes how many programs exist, not what any surviving
    program sums, and the combine pass loops only to its own sequence's split
    count.
  * BLOCK_M tiles query rows. Rows are independent (section 4.2, "Rows are
    independent, so M-tiling may vary freely"), and it is pinned anyway.

Empty partials carry m = -1e30 rather than -inf. With -inf, an all-empty fold
computes exp(-inf - -inf) = exp(nan); with a finite sentinel the same fold
computes exp(0) = 1 against l = 0 and acc = 0, contributing exactly zero. Real
scores cannot approach -1e30: fp32 q.k over fp16 inputs at head_dim 128 is
bounded by roughly 5.5e11.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import registry

# Wrapped in tl.constexpr because Triton refuses plain module globals inside a
# jitted function. The bare annotation form does not work: an annotation does not
# change the assigned value, and it is the value the JIT inspects.
NEG_SENTINEL = tl.constexpr(-1e30)


@triton.jit
def _attn_split_kernel(
    Q,
    K,
    V,
    Acc,
    MPart,
    LPart,
    q_len,
    kv_len,
    q_start,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_at,
    stride_ah,
    stride_as,
    stride_ad,
    stride_mt,
    stride_mh,
    stride_ms,
    GQA_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    """One program owns one (query tile, KV split, query head).

    Writes an unnormalized fp32 accumulator plus the running max and sum that
    the combine pass needs. Normalization deliberately does not happen here: a
    partial that had already divided by its own l could not be refolded.
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < q_len
    q_pos = q_start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), NEG_SENTINEL, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    split_start = pid_s * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, kv_len)

    # Causal early exit: if this split begins past the last query position in the
    # tile, every score in it is masked. Skipping writes the same empty partial
    # the loop would have produced, so the combine result is unchanged.
    last_q_pos = q_start + tl.minimum(pid_m * BLOCK_M + BLOCK_M - 1, q_len - 1)
    active = (split_start < split_end) and (split_start <= last_q_pos)

    if active:
        kv_head = pid_h // GQA_GROUP
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None],
            other=0.0,
        )

        # Ascending 128-token tiles within this 512-token span. Both bounds come
        # from this request's own kv_len; nothing here has seen another request.
        for n0 in range(split_start, split_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_valid = offs_n < split_end

            k = tl.load(
                K + offs_n[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
                mask=n_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            keep = n_valid[None, :] & m_valid[:, None] & (offs_n[None, :] <= q_pos[:, None])
            qk = tl.where(keep, qk, NEG_SENTINEL)

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            v = tl.load(
                V + offs_n[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
                mask=n_valid[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    tl.store(
        Acc
        + offs_m[:, None] * stride_at
        + pid_h * stride_ah
        + pid_s * stride_as
        + offs_d[None, :] * stride_ad,
        acc,
        mask=m_valid[:, None],
    )
    part_offset = offs_m * stride_mt + pid_h * stride_mh + pid_s * stride_ms
    tl.store(MPart + part_offset, m_i, mask=m_valid)
    tl.store(LPart + part_offset, l_i, mask=m_valid)


@triton.jit
def _attn_combine_kernel(
    Acc,
    MPart,
    LPart,
    Out,
    q_len,
    num_splits,
    stride_at,
    stride_ah,
    stride_as,
    stride_ad,
    stride_mt,
    stride_mh,
    stride_ms,
    stride_ot,
    stride_oh,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fold the per-split partials, one CTA per (query tile, head).

    Each query row's fold happens entirely inside this one CTA, in ascending
    split index, in fp32. That is the architecture doc's requirement stated three
    ways: single CTA, ascending order, fp32.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < q_len

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_run = tl.full((BLOCK_M,), NEG_SENTINEL, dtype=tl.float32)
    l_run = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for s in range(0, num_splits):
        part_offset = offs_m * stride_mt + pid_h * stride_mh + s * stride_ms
        m_s = tl.load(MPart + part_offset, mask=m_valid, other=NEG_SENTINEL)
        l_s = tl.load(LPart + part_offset, mask=m_valid, other=0.0)
        a_s = tl.load(
            Acc
            + offs_m[:, None] * stride_at
            + pid_h * stride_ah
            + s * stride_as
            + offs_d[None, :] * stride_ad,
            mask=m_valid[:, None],
            other=0.0,
        )

        m_new = tl.maximum(m_run, m_s)
        alpha = tl.exp(m_run - m_new)
        beta = tl.exp(m_s - m_new)

        acc = acc * alpha[:, None] + a_s * beta[:, None]
        l_run = l_run * alpha + l_s * beta
        m_run = m_new

    # l_run is zero only for a row with no unmasked key, which causal attention
    # never produces since a query always attends to its own position.
    out = tl.where(l_run[:, None] > 0.0, acc / l_run[:, None], 0.0)

    tl.store(
        Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :] * stride_od,
        out.to(Out.dtype.element_ty),
        mask=m_valid[:, None],
    )


def _num_splits(kv_len: int) -> int:
    """ceil(kv_len / SPLIT_SIZE), from this request's own KV length alone."""
    return triton.cdiv(kv_len, registry.SPLIT_SIZE)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_start: int,
    sm_scale: float,
) -> torch.Tensor:
    """Causal attention for one sequence.

    q is [q_len, n_heads, head_dim]; k and v are [kv_len, n_kv_heads, head_dim]
    holding the whole sequence's keys and values. `q_start` is the position of
    q[0] in the sequence, so prefill passes 0 and a decode step passes
    kv_len - 1. Chunked prefill will pass an arbitrary interior offset, and the
    invariant is that doing so changes nothing.

    Multi-sequence dispatch is week 2's paged KV work and is deliberately absent.
    """
    split_cfg = registry.get("attention.split")
    combine_cfg = registry.get("attention.combine")

    q_len, n_heads, head_dim = q.shape
    kv_len, n_kv_heads, head_dim_k = k.shape
    assert head_dim == head_dim_k == split_cfg.constants["BLOCK_D"]
    assert k.shape == v.shape
    assert n_heads % n_kv_heads == 0, "GQA needs the head count to divide evenly"
    assert q_start + q_len == kv_len, (
        f"q_start={q_start} q_len={q_len} does not end at kv_len={kv_len}; "
        "queries must be the tail of the keys they attend to"
    )

    gqa_group = n_heads // n_kv_heads
    splits = _num_splits(kv_len)
    max_splits = combine_cfg.constants["MAX_SPLITS"]
    assert splits <= max_splits, (
        f"kv_len={kv_len} needs {splits} splits of {registry.SPLIT_SIZE}, past the "
        f"pinned MAX_SPLITS={max_splits}. Raising it is a registry change and "
        "therefore a claims-affecting change."
    )

    block_m = split_cfg.constants["BLOCK_M"]
    acc = torch.empty((q_len, n_heads, splits, head_dim), dtype=torch.float32, device=q.device)
    m_part = torch.empty((q_len, n_heads, splits), dtype=torch.float32, device=q.device)
    l_part = torch.empty((q_len, n_heads, splits), dtype=torch.float32, device=q.device)

    _attn_split_kernel[(triton.cdiv(q_len, block_m), splits, n_heads)](
        q, k, v, acc, m_part, l_part,
        q_len, kv_len, q_start, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        GQA_GROUP=gqa_group,
        BLOCK_M=block_m,
        BLOCK_N=split_cfg.constants["BLOCK_N"],
        BLOCK_D=head_dim,
        SPLIT_SIZE=registry.SPLIT_SIZE,
        num_warps=split_cfg.num_warps,
        num_stages=split_cfg.num_stages,
    )

    out = torch.empty((q_len, n_heads, head_dim), dtype=q.dtype, device=q.device)
    _attn_combine_kernel[(triton.cdiv(q_len, block_m), n_heads)](
        acc, m_part, l_part, out,
        q_len, splits,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=block_m,
        BLOCK_D=head_dim,
        num_warps=combine_cfg.num_warps,
        num_stages=combine_cfg.num_stages,
    )
    return out
