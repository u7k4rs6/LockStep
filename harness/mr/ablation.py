"""The ablation table: a positive control for the invariance test.

The kernels here were written invariant from day one. Running the invariance
suite against them and reporting "nothing breaks" proves nothing at all: a test
that has never been shown to go red is unfalsifiable, and an all-green table from
one is decoration.

So for each op under claim this builds a deliberately non-invariant variant, swaps
exactly one in at a time, and checks that I1 goes red while the full invariant set
holds. The variants are the ordinary torch implementations, which is the point:
they are not strawmen, they are what an engine written without this constraint
would use, and the architecture doc section 5 names why each is a hazard.

I1, as tested here: a request's logits must be bit-identical whether it runs
alone or alongside cohabitants. Canonical execution C(r) is batch size 1, so the
batch-1 run is the reference and each larger batch is compared against it.

Cohabitants have deliberately uneven lengths. A batch of equal-length prompts
would keep the packed token count a clean multiple of the GEMM tile and could let
a shape-dependent kernel look invariant by luck.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from engine.kv import paged  # noqa: E402
from engine.model import qwen3  # noqa: E402

BATCH_SIZES = (1, 2, 4, 8, 16, 31, 32)


# ---- non-invariant variants -------------------------------------------------


def torch_linear(x, weight, bias=None, config="gemm.default"):
    """cuBLAS via torch.matmul.

    Architecture doc section 5, "Split-K": cuBLAS split-K combines via atomics or
    a heuristically chosen reduce pass, and the heuristic reads the runtime
    shape. M here is the packed token count, which is batch-derived.
    """
    out = torch.matmul(x, weight.t())
    return out + bias if bias is not None else out


def torch_rmsnorm(x, weight, eps, config="rmsnorm.hidden"):
    """torch's reduction over the last dimension.

    Kept in the table but marked not-a-probe: on a 1024-wide row torch uses a
    single block whatever the batch is, so no batch-derived quantity reaches the
    reduction and this was never going to go red. See split_reduction_rmsnorm
    below for the variant that actually breaks.
    """
    original = x.dtype
    x32 = x.to(torch.float32)
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.to(torch.float32)).to(original)


@triton.jit
def _atomic_sumsq_kernel(X, Partial, n_cols, stride_row, SPLIT: tl.constexpr):
    """Sum of squares accumulated across splits with an atomic add."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * SPLIT + tl.arange(0, SPLIT)
    x = tl.load(X + row * stride_row + offs, mask=offs < n_cols, other=0.0).to(tl.float32)
    tl.atomic_add(Partial + row, tl.sum(x * x, axis=0))


def split_reduction_rmsnorm(x, weight, eps, config="rmsnorm.hidden"):
    """RMSNorm whose split count is keyed on the batch token count, combined
    with atomics.

    This is the failure architecture doc section 5 names under "Split-K" and
    "Fixed split count", applied to a norm rather than a GEMM, and it is the
    variant the RMSNorm row needed: swapping torch in tested nothing because
    torch never saw a batch-derived shape.

    Two things are wrong with it at once, and either alone turns I1 red. The
    split width is read from the batch, so the partition of the reduction
    changes with cohabitants. And the partials combine through tl.atomic_add,
    whose summation order is arrival order, so it is not even stable run to run.
    The static gate in scripts/static_checks.py greps engine/ for tl.atomic_ and
    would reject this file; it lives under harness/ precisely because it is the
    thing the gate exists to keep out.
    """
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    rows, cols = x2d.shape

    split = (128, 256, 512, 1024)[_BATCH_TOKENS % 4]
    partial = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _atomic_sumsq_kernel[(rows, triton.cdiv(cols, split))](
        x2d, partial, cols, x2d.stride(0), SPLIT=split, num_warps=4
    )

    scale = torch.rsqrt(partial / cols + eps)
    out = x2d.to(torch.float32) * scale[:, None] * weight.to(torch.float32)
    return out.to(x.dtype).reshape(shape)


def torch_attention(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale):
    """torch SDPA over a gathered contiguous view.

    SDPA picks a backend (flash, mem-efficient, math) from the shape, and the
    flash path's own split heuristics are shape-dependent.
    """
    kv_block = k_cache.shape[1]
    order = block_table.tolist()
    k = torch.cat([k_cache[b] for b in order], dim=0)[:kv_len]
    v = torch.cat([v_cache[b] for b in order], dim=0)[:kv_len]

    q_len, n_heads, head_dim = q.shape
    group = n_heads // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    mask = (
        torch.arange(kv_len, device=q.device)[None, :]
        <= torch.arange(q_start, q_start + q_len, device=q.device)[:, None]
    )
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=mask[None, None],
        scale=sm_scale,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


# Set by `target_logits` before each packed forward, so the variant below can
# read a genuinely batch-derived quantity. Nothing in engine/ can see it.
_BATCH_TOKENS = 1


def batch_derived_split_attention(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale):
    """Attention whose split size is chosen from the batch's total token count.

    This is the failure the architecture doc names twice: section 5, "Fixed split
    count. Reintroduces batch dependence if derived from batch-max sequence
    length", and mutation operator 10.1, "one code path reads split size from
    batch size". The arithmetic is otherwise a faithful online-softmax fold, so
    the only thing wrong with it is where the blocking constant came from.

    Swapping torch SDPA in instead does not test this: the engine calls attention
    once per sequence, so SDPA never sees a batch-derived shape and stays
    invariant for a structural reason rather than a numerical one. That is worth
    knowing, and it is why this variant exists alongside it.
    """
    kv_block = k_cache.shape[1]
    order = block_table.tolist()
    k = torch.cat([k_cache[b] for b in order], dim=0)[:kv_len]
    v = torch.cat([v_cache[b] for b in order], dim=0)[:kv_len]

    q_len, n_heads, head_dim = q.shape
    group = n_heads // k.shape[1]
    k = k.repeat_interleave(group, dim=1).to(torch.float32)
    v = v.repeat_interleave(group, dim=1).to(torch.float32)
    q32 = q.to(torch.float32)

    # The offending line: a blocking constant read from the batch. Kept small
    # enough that the target request actually spans several chunks; a chunk
    # larger than kv_len would fold in one step and the bug would hide.
    chunk = 16 + 4 * (_BATCH_TOKENS % 8)

    positions = torch.arange(q_start, q_start + q_len, device=q.device)
    acc = torch.zeros((q_len, n_heads, head_dim), dtype=torch.float32, device=q.device)
    running_max = torch.full((q_len, n_heads), -1e30, dtype=torch.float32, device=q.device)
    running_sum = torch.zeros((q_len, n_heads), dtype=torch.float32, device=q.device)

    for start in range(0, kv_len, chunk):
        stop = min(start + chunk, kv_len)
        keys = torch.arange(start, stop, device=q.device)
        scores = torch.einsum("qhd,khd->qhk", q32, k[start:stop]) * sm_scale
        scores = scores.masked_fill(keys[None, None, :] > positions[:, None, None], -1e30)

        chunk_max = scores.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(running_max - new_max)
        probs = torch.exp(scores - new_max[:, :, None])

        acc = acc * alpha[:, :, None] + torch.einsum("qhk,khd->qhd", probs, v[start:stop])
        running_sum = running_sum * alpha + probs.sum(dim=-1)
        running_max = new_max

    return (acc / running_sum.clamp_min(1e-30)[:, :, None]).to(q.dtype)


def torch_swiglu(gate, up):
    return F.silu(gate.to(torch.float32)).mul(up.to(torch.float32)).to(gate.dtype)


@dataclass(frozen=True)
class Variant:
    op: str
    attribute: str
    replacement: object
    note: str
    # A probe tests something only if a batch-derived quantity can reach the op
    # at all. Where none can, the right answer is a structural argument, not a
    # green cell: "held" and "no breaking variant exists" are different claims
    # and a reader cannot tell them apart if both print as held.
    is_probe: bool = True
    structural_reason: str = ""


VARIANTS = (
    Variant("GEMM (all projections)", "linear", torch_linear,
            "torch.matmul / cuBLAS, split-K by shape heuristic"),
    Variant("GEMM (lm_head only)", "linear_lm_head", torch_linear,
            "torch.matmul on the 151936-wide head only"),
    Variant("RMSNorm (split from batch)", "rmsnorm", split_reduction_rmsnorm,
            "split width from batch token count, partials combined by atomic_add"),
    Variant("attention (split from batch)", "attention", batch_derived_split_attention,
            "split size read from the batch token count, arch doc 5 and 10.1"),
    Variant("RMSNorm (torch)", "rmsnorm", torch_rmsnorm,
            "torch mean reduction over the last dim", is_probe=False,
            structural_reason="a 1024-wide row is one block whatever the batch is, "
                              "so no batch-derived quantity reaches the reduction"),
    Variant("attention (SDPA)", "attention", torch_attention,
            "torch scaled_dot_product_attention", is_probe=False,
            structural_reason="the engine calls attention once per sequence, so SDPA "
                              "never sees a batch-derived shape"),
    Variant("SwiGLU (torch)", "swiglu", torch_swiglu,
            "torch silu/mul", is_probe=False,
            structural_reason="elementwise: each output reads one gate and one up "
                              "element, so batching cannot reach it"),
)


# ---- I1 probe ---------------------------------------------------------------


# Two shape profiles, both probed for every variant. One is not enough: cuBLAS's
# non-invariance is intermittent across shapes, and the lm_head variant below is
# detectable under the short-target profile and invariant under the long-target
# one. Reporting whichever profile made the table look best would be tuning the
# test to its subject.
PROFILES = {
    "short-target": (17, 33, 8, 64, 25, 41, 12, 55),
    "long-target": (96, 33, 8, 21, 25, 41, 12, 17),
}


def make_prompts(count: int, vocab: int, profile: str, seed: int = 4242) -> list[list[int]]:
    """Uneven lengths, so the packed token count is never a tidy multiple.

    r00, the request under test, is deliberately the long one. A short target
    would sit inside a single tile of every variant's blocking and could hold
    invariant for want of anything to reorder, which would read as the test
    passing rather than as the probe missing.

    Cohabitants stay short so that batch 32 packs a few hundred tokens rather
    than a few thousand: forward_batch materializes logits for every packed
    token, and 4000 tokens of 151936-wide fp16 is 1.2 GB.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = list(PROFILES[profile])
    prompts = []
    for index in range(count):
        length = lengths[index % len(lengths)] + (index % 5)
        prompts.append(
            torch.randint(0, vocab, (length,), generator=generator).tolist()
        )
    return prompts


def target_logits(model, prompts: list[list[int]], num_blocks: int) -> torch.Tensor:
    """Logits for prompt 0 when run alongside the rest, as raw bytes."""
    pool = paged.PagedKVCache(
        num_blocks=num_blocks,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        device=model.device,
        dtype=torch.float16,
        poison_on_free=False,
    )
    global _BATCH_TOKENS
    _BATCH_TOKENS = sum(len(p) for p in prompts)

    work = []
    for index, prompt in enumerate(prompts):
        uid = f"r{index:02d}"
        pool.create(uid)
        pool.reserve(uid, len(prompt))
        work.append((uid, prompt, 0))

    out = model.forward_batch(pool, work)["r00"].clone()
    del pool
    torch.cuda.empty_cache()
    return out


def check_i1(model, vocab: int, num_blocks: int, profile: str) -> dict:
    """Compare prompt 0's logits at batch 1 against every larger batch."""
    prompts = make_prompts(max(BATCH_SIZES), vocab, profile)
    canonical = target_logits(model, prompts[:1], num_blocks)

    held = True
    first_batch = None
    first_position = None
    magnitude = 0.0

    for size in BATCH_SIZES:
        if size == 1:
            continue
        observed = target_logits(model, prompts[:size], num_blocks)
        if torch.equal(observed, canonical):
            continue
        held = False
        diff = (observed.to(torch.float64) - canonical.to(torch.float64)).abs()
        rows = torch.nonzero(diff.sum(dim=-1) > 0).flatten()
        if first_batch is None:
            first_batch = size
            first_position = int(rows[0]) if rows.numel() else 0
        magnitude = max(magnitude, float(diff.max()))

    return {
        "held": held,
        "first_batch_size": first_batch,
        "first_position": first_position,
        "max_abs_divergence": magnitude,
    }


def run(weights: Path, num_blocks: int = 512) -> list[dict]:
    model = qwen3.Qwen3(weights, max_len=256)
    vocab = model.cfg.vocab_size

    def probe_all(label):
        results = {p: check_i1(model, vocab, num_blocks, p) for p in PROFILES}
        red = [p for p, r in results.items() if not r["held"]]
        worst = next((results[p] for p in red), results["short-target"])
        return {
            "op": label,
            "held": not red,
            "red_profiles": red,
            "profiles_probed": list(PROFILES),
            "first_batch_size": worst["first_batch_size"],
            "first_position": worst["first_position"],
            "max_abs_divergence": max(r["max_abs_divergence"] for r in results.values()),
            "per_profile": results,
        }

    rows = [{**probe_all("none (full invariant set)"), "variant": "-",
             "is_probe": True, "structural_reason": ""}]

    originals = {
        "linear": qwen3.linear,
        "rmsnorm": qwen3.rmsnorm,
        "attention": qwen3.attention,
        "swiglu": qwen3.swiglu,
    }

    for variant in VARIANTS:
        if variant.attribute == "linear_lm_head":
            # Swap only the lm_head call by dispatching on the config name, so
            # the row isolates the 151936-wide GEMM from the rest.
            def only_head(x, weight, bias=None, config="gemm.default", _real=originals["linear"]):
                if config == "gemm.lm_head":
                    return torch_linear(x, weight, bias, config)
                return _real(x, weight, bias, config)

            qwen3.linear = only_head
        else:
            setattr(qwen3, variant.attribute, variant.replacement)

        try:
            result = probe_all(variant.op)
        finally:
            for name, function in originals.items():
                setattr(qwen3, name, function)

        rows.append({**result, "variant": variant.note, "is_probe": variant.is_probe,
                     "structural_reason": variant.structural_reason})

    return rows


def format_table(rows: list[dict]) -> str:
    lines = [
        "ablation: does the I1 test actually go red when an op stops being invariant?",
        f"batch sizes {list(BATCH_SIZES)}, request r00's logits compared bitwise against its",
        "own batch-1 (canonical) run",
        "",
        f"{'op swapped':<28} {'probe?':<8} {'I1':<5} {'red in':>8} {'first B':>8}"
        f" {'first pos':>10} {'max |delta|':>12}",
        "-" * 88,
    ]
    for row in rows:
        if not row.get("is_probe", True):
            lines.append(f"{row['op']:<28} {'N/A':<8} {'-':<5} {'-':>8} {'-':>8} {'-':>10} {'-':>12}")
            continue
        verdict = "held" if row["held"] else "RED"
        red_in = "-" if row["held"] else f"{len(row['red_profiles'])}/{len(row['profiles_probed'])}"
        first_b = "-" if row["first_batch_size"] is None else str(row["first_batch_size"])
        first_p = "-" if row["first_position"] is None else str(row["first_position"])
        magnitude = "-" if row["held"] else f"{row['max_abs_divergence']:.4e}"
        lines.append(
            f"{row['op']:<28} {'probe':<8} {verdict:<5} {red_in:>8} {first_b:>8}"
            f" {first_p:>10} {magnitude:>12}"
        )
    lines.append("")
    lines.append(f"shape profiles probed: {', '.join(PROFILES)} (r00 length differs)")
    lines.append("")
    lines.append("probes, one deliberately non-invariant variant each:")
    for row in rows[1:]:
        if not row.get("is_probe", True):
            continue
        detail = f"  {row['op']:<28} {row['variant']}"
        if row["red_profiles"] and len(row["red_profiles"]) < len(row["profiles_probed"]):
            detail += f"\n  {'':<28} red only under: {', '.join(row['red_profiles'])}"
        lines.append(detail)
    lines.append("")
    lines.append("N/A, no batch-derived quantity can reach the op, so a probe would be theater:")
    for row in rows[1:]:
        if row.get("is_probe", True):
            continue
        lines.append(f"  {row['op']:<28} {row['structural_reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from engine import envlock
    from report.artifact import Artifact, relpath

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "Qwen3-0.6B")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    table = run(args.weights)
    print(format_table(table))

    env = envlock.capture()
    print()
    print(f"env  {env.fingerprint()}")

    probes = [r for r in table[1:] if r.get("is_probe", True)]
    detected = sum(1 for r in probes if not r["held"])
    print(f"\nprobes detected: {detected}/{len(probes)}")
    print(f"not probes (structural argument instead): {len(table) - 1 - len(probes)}")
    print(f"invariant set held: {table[0]['held']}")

    if not args.no_artifact:
        path = Artifact(kind="ablation", env=env, payload={"rows": table}).write()
        print(f"artifact  {relpath(path)}")
