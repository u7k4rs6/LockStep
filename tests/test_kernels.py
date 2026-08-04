"""Kernel correctness, plus the row-count invariance the registry claims.

Two kinds of test here. Correctness against a torch fp64 reference is the
ordinary kind. The other kind checks that a row's output does not depend on how
many other rows were in the launch, which is I1 in miniature: the full
metamorphic suite is week 2 and later, but a kernel that fails this now would
fail it then, and finding that out here costs one launch instead of a scheduler.

These are not the invariance claim. They cover single kernels with no scheduler,
no KV cache, and no cohabitant requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernels under claim run on the pinned GPU"
)

from engine.kernels import registry  # noqa: E402
from engine.kernels.attention import attention  # noqa: E402
from engine.kernels.gemm import linear  # noqa: E402
from engine.kernels.rmsnorm import rmsnorm  # noqa: E402
from engine.kernels.rope import apply_rope  # noqa: E402
from engine.kernels.softmax import log_softmax  # noqa: E402
from engine.kernels.swiglu import swiglu  # noqa: E402

SEED = 20260803


def randn(*shape, dtype=torch.float16):
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)


def assert_within_fp16_rounding(got: torch.Tensor, want: torch.Tensor, ulps: float = 1.0):
    """Assert `got` is `want` to within `ulps` fp16 units in the last place.

    An absolute tolerance would be either meaningless at magnitude 1e-3 or vacuous
    at magnitude 1e3. fp16 carries 10 explicit mantissa bits, so one ULP is
    x * 2**-10; the floor covers values in the subnormal range where that formula
    goes to zero.
    """
    smallest = torch.finfo(torch.float16).smallest_normal
    tolerance = ulps * (want.abs().clamp_min(smallest) * 2**-10)
    error = (got - want).abs()
    worst = int((error / tolerance).argmax())
    assert (error <= tolerance).all(), (
        f"worst element: got {got.flatten()[worst]:.6e} want {want.flatten()[worst]:.6e} "
        f"= {float((error / tolerance).max()):.2f} ULP"
    )


# ---- GEMM -------------------------------------------------------------------


def test_gemm_matches_fp64():
    x = randn(37, 1024)
    w = randn(3072, 1024)
    got = linear(x, w).to(torch.float64)
    want = x.to(torch.float64) @ w.to(torch.float64).T
    assert (got - want).abs().max() < 0.5


def test_gemm_bias_is_added():
    """Checked against fp64, not against the no-bias result minus the bias.

    Those differ: the accumulator rounds to fp16 once with the bias folded in and
    once without, and at the ~1e3 magnitudes a 1024-deep dot product reaches, the
    fp16 spacing is 0.5, so a unit-scale bias is not recoverable by subtraction.
    """
    x = randn(16, 1024)
    w = randn(128, 1024)
    b = randn(128)

    got = linear(x, w, bias=b).to(torch.float64)
    want = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    assert (got - want).abs().max() < 0.5

    unbiased = linear(x, w, bias=None).to(torch.float64)
    assert not torch.equal(got, unbiased), "bias flag had no effect at all"


@pytest.mark.parametrize("config", ["gemm.default", "gemm.lm_head"])
def test_gemm_row_output_is_independent_of_row_count(config):
    """The same row, alone and inside a 64-row launch, must be bit-identical.

    This is what "no batch-derived quantity reaches a kernel config" buys. A
    split-K implementation would fail here as soon as the heuristic switched.
    """
    w = randn(256, 1024)
    rows = randn(64, 1024)

    batched = linear(rows, w, config=config)
    for index in (0, 1, 17, 63):
        alone = linear(rows[index : index + 1], w, config=config)
        assert torch.equal(alone[0], batched[index]), f"row {index} moved"


def test_gemm_is_bitwise_repeatable():
    x, w = randn(64, 1024), randn(512, 1024)
    assert torch.equal(linear(x, w), linear(x, w))


# ---- RMSNorm ----------------------------------------------------------------


@pytest.mark.parametrize(
    "config,width", [("rmsnorm.hidden", 1024), ("rmsnorm.head", 128)]
)
def test_rmsnorm_matches_fp64(config, width):
    x = randn(19, width)
    w = randn(width)
    eps = 1e-6

    got = rmsnorm(x, w, eps, config).to(torch.float64)
    x64 = x.to(torch.float64)
    want = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w.to(torch.float64)
    assert_within_fp16_rounding(got, want)


def test_rmsnorm_row_output_is_independent_of_row_count():
    x = randn(48, 1024)
    w = randn(1024)
    batched = rmsnorm(x, w, 1e-6)
    for index in (0, 5, 47):
        alone = rmsnorm(x[index : index + 1], w, 1e-6)
        assert torch.equal(alone[0], batched[index])


def test_rmsnorm_rejects_a_row_wider_than_its_pinned_tile():
    with pytest.raises(AssertionError, match="one CTA per row"):
        rmsnorm(randn(2, 1024), randn(1024), 1e-6, "rmsnorm.head")


# ---- log-softmax ------------------------------------------------------------


def test_log_softmax_matches_fp64_over_the_full_vocab():
    x = randn(5, 151936)
    got = log_softmax(x).to(torch.float64)
    x64 = x.to(torch.float64)
    want = x64 - torch.logsumexp(x64, dim=-1, keepdim=True)
    assert (got - want).abs().max() < 1e-3


def test_log_softmax_row_output_is_independent_of_row_count():
    x = randn(8, 151936)
    batched = log_softmax(x)
    for index in (0, 3, 7):
        assert torch.equal(log_softmax(x[index : index + 1])[0], batched[index])


def test_log_softmax_sums_to_one():
    """1e-5 is the honest bound: the kernel emits fp32 log-probs, and summing
    151936 of them after exp accumulates roughly |log p| * eps_fp32 relative
    error, which lands near 1e-6 at this vocabulary size."""
    probs = log_softmax(randn(4, 151936)).to(torch.float64).exp().sum(dim=-1)
    assert (probs - 1.0).abs().max() < 1e-5


# ---- RoPE -------------------------------------------------------------------


def test_rope_matches_the_half_rotation_reference():
    tokens, heads, head_dim = 11, 8, 128
    x = randn(tokens, heads, head_dim)
    angles = torch.rand(tokens, head_dim // 2, device="cuda", dtype=torch.float32) * 6.0
    cos, sin = angles.cos(), angles.sin()

    got = apply_rope(x, cos, sin).to(torch.float64)

    x64 = x.to(torch.float64)
    lo, hi = x64[..., :64], x64[..., 64:]
    c = cos.to(torch.float64)[:, None, :]
    s = sin.to(torch.float64)[:, None, :]
    want = torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1)
    assert (got - want).abs().max() < 1e-2


def test_rope_at_position_zero_is_the_identity():
    """cos(0)=1, sin(0)=0, so the rotation must not perturb a single bit."""
    x = randn(3, 4, 128)
    cos = torch.ones(3, 64, device="cuda", dtype=torch.float32)
    sin = torch.zeros(3, 64, device="cuda", dtype=torch.float32)
    assert torch.equal(apply_rope(x, cos, sin), x)


# ---- SwiGLU -----------------------------------------------------------------


def test_swiglu_matches_fp64():
    gate, up = randn(7, 3072), randn(7, 3072)
    got = swiglu(gate, up).to(torch.float64)
    g64, u64 = gate.to(torch.float64), up.to(torch.float64)
    want = g64 * torch.sigmoid(g64) * u64
    assert_within_fp16_rounding(got, want, ulps=2.0)


# ---- Attention --------------------------------------------------------------


def stable_softmax_fp64(x: torch.Tensor) -> torch.Tensor:
    """Explicit max-subtract softmax, because torch.softmax cannot be trusted here.

    `torch.softmax(dim=-1)` on CUDA in fp64 returns wrong results at row widths
    513 and 769 for two or more rows, off by as much as 0.5 in probability. See
    tests/test_torch_softmax_hazard.py for the pinned repro. This helper is the
    reason the attention tests are trustworthy at kv_len=513, which is precisely
    a split-boundary case this suite has to cover.
    """
    m = x.max(dim=-1, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)



def page(k, v, shuffle_seed=None):
    """Lay contiguous [kv_len, heads, dim] K/V into a paged pool.

    Returns (k_cache, v_cache, block_table, kv_len). With `shuffle_seed`, the
    physical blocks are permuted, which is the realistic state after a pool has
    been allocated and freed a few times and is what the invariance test needs.
    """
    from engine.kv.paged import BLOCK_SIZE

    kv_len, heads, dim = k.shape
    nblocks = -(-kv_len // BLOCK_SIZE)
    order = list(range(nblocks))
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(nblocks, generator=g).tolist()

    k_cache = torch.zeros((nblocks, BLOCK_SIZE, heads, dim), dtype=k.dtype, device=k.device)
    v_cache = torch.zeros_like(k_cache)
    for logical in range(nblocks):
        lo = logical * BLOCK_SIZE
        hi = min(lo + BLOCK_SIZE, kv_len)
        k_cache[order[logical], : hi - lo] = k[lo:hi]
        v_cache[order[logical], : hi - lo] = v[lo:hi]
    table = torch.tensor(order, dtype=torch.int32, device=k.device)
    return k_cache, v_cache, table, kv_len


def naive_attention_fp64(q, k, v, q_start, sm_scale):
    q_len, n_heads, head_dim = q.shape
    kv_len, n_kv_heads, _ = k.shape
    group = n_heads // n_kv_heads

    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64).repeat_interleave(group, dim=1)
    v64 = v.to(torch.float64).repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q64, k64) * sm_scale
    q_pos = torch.arange(q_start, q_start + q_len, device=q.device)[:, None]
    k_pos = torch.arange(kv_len, device=q.device)[None, :]
    scores = scores.masked_fill((k_pos > q_pos)[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", stable_softmax_fp64(scores), v64)


@pytest.mark.parametrize("kv_len", [1, 64, 128, 129, 511, 512, 513, 1024, 1500])
def test_attention_matches_fp64_across_split_boundaries(kv_len):
    """512 is the split size and 128 the KV tile, so kv_len at and around both
    multiples is where a fencepost in the split bounds would show up."""
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    sm_scale = head_dim**-0.5

    k = randn(kv_len, n_kv_heads, head_dim)
    v = randn(kv_len, n_kv_heads, head_dim)
    q = randn(kv_len, n_heads, head_dim)

    kc, vc, table, _ = page(k, v)
    got = attention(q, kc, vc, table, q_start=0, kv_len=kv_len, sm_scale=sm_scale).to(torch.float64)
    want = naive_attention_fp64(q, k, v, 0, sm_scale)
    assert (got - want).abs().max() < 2e-2


def test_attention_decode_step_matches_fp64():
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    kv_len = 700
    sm_scale = head_dim**-0.5

    k = randn(kv_len, n_kv_heads, head_dim)
    v = randn(kv_len, n_kv_heads, head_dim)
    q = randn(1, n_heads, head_dim)

    kc, vc, table, _ = page(k, v)
    got = attention(
        q, kc, vc, table, q_start=kv_len - 1, kv_len=kv_len, sm_scale=sm_scale
    ).to(torch.float64)
    want = naive_attention_fp64(q, k, v, kv_len - 1, sm_scale)
    assert (got - want).abs().max() < 2e-2


def test_attention_query_row_is_independent_of_the_query_tile_it_lands_in():
    """One query at position p, alone, must equal that row of a full prefill.

    Prefill-versus-decode equivalence over the *KV cache* is chunk invariance and
    is week 3's claim; this is the weaker statement that query-row tiling alone
    does not move a row, which the pinned BLOCK_M is supposed to guarantee.
    """
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    kv_len = 600
    sm_scale = head_dim**-0.5

    k = randn(kv_len, n_kv_heads, head_dim)
    v = randn(kv_len, n_kv_heads, head_dim)
    q = randn(kv_len, n_heads, head_dim)

    kc, vc, table, _ = page(k, v)
    full = attention(q, kc, vc, table, q_start=0, kv_len=kv_len, sm_scale=sm_scale)
    for position in (0, 1, 15, 16, 17, 511, 512, 599):
        kc1, vc1, table1, _ = page(k[: position + 1], v[: position + 1])
        alone = attention(
            q[position : position + 1],
            kc1, vc1, table1,
            q_start=position,
            kv_len=position + 1,
            sm_scale=sm_scale,
        )
        assert torch.equal(alone[0], full[position]), f"query row {position} moved"


def test_attention_refuses_a_kv_length_past_the_pinned_ceiling():
    ceiling = registry.get("attention.combine").constants["MAX_SPLITS"]
    kv_len = (ceiling + 1) * registry.SPLIT_SIZE
    q = torch.zeros(1, 16, 128, dtype=torch.float16, device="cuda")
    k = torch.zeros(kv_len, 8, 128, dtype=torch.float16, device="cuda")
    kc, vc, table, _ = page(k, k)
    with pytest.raises(AssertionError, match="MAX_SPLITS"):
        attention(q, kc, vc, table, q_start=kv_len - 1, kv_len=kv_len, sm_scale=1.0)


@pytest.mark.parametrize("kv_len", [64, 512, 600, 1024])
def test_attention_is_bit_identical_under_a_shuffled_block_table(kv_len):
    """Paging must move bytes, never the order they are combined in.

    The same logical sequence laid out in ascending physical blocks and in a
    permuted set must produce identical bits. If this fails, the gather has
    become order-dependent and every downstream invariance claim is void.
    """
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    sm_scale = head_dim**-0.5
    k = randn(kv_len, n_kv_heads, head_dim)
    v = randn(kv_len, n_kv_heads, head_dim)
    q = randn(kv_len, n_heads, head_dim)

    kc_a, vc_a, table_a, _ = page(k, v)
    kc_b, vc_b, table_b, _ = page(k, v, shuffle_seed=99)
    assert not torch.equal(table_a, table_b), "the shuffle did not shuffle anything"

    out_a = attention(q, kc_a, vc_a, table_a, q_start=0, kv_len=kv_len, sm_scale=sm_scale)
    out_b = attention(q, kc_b, vc_b, table_b, q_start=0, kv_len=kv_len, sm_scale=sm_scale)
    assert torch.equal(out_a, out_b)
