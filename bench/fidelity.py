"""F1: the engine's fp16 logits against the fp64 CPU reference.

docs/02-technical-architecture.md section 7 names the reported metrics: max
absolute error, per-position KL, greedy-token match rate, with near-tie positions
excluded from the match rate and counted separately.

Three KL numbers are reported, not one:

  * **exact KL**, over all 151936 tokens, on the stratified audit subset. This is
    the headline KL, because it is the only one that is exact.
  * **compressed KL**, over every position, from the top-k reference. It
    partitions the vocabulary into the k retained tokens plus one bin holding the
    rest. KL over a coarser partition is never larger than KL over the finer one,
    so this is a *lower bound* on the true full-vocabulary KL, never an estimate.
  * **the gap between them on the audit subset**, which says how loose that bound
    is.

The gap was measured rather than assumed, and the measurement changed the design.
At k=256 the compressed KL was missing 27.4 percent of the true KL, and the
sweep from 256 to 8192 (still available under --k-sweep) closes the gap only
about 1.3x per doubling. There is no practical k at which top-k KL becomes exact
on a 151936-token vocabulary with a tail this heavy; reaching one percent would
take k in the tens of thousands, which is not compression. So k moved to 2048 and
the framing moved with it: the exact number leads, and the wide-coverage number
is presented as the bound it is, with its measured looseness printed beside it.

Near-tie exclusion is exact under compression, because it needs only the fp64
top-1 and top-2, both of which the top-256 reference retains at full fp64
precision. That is stated in the output so a reader does not have to wonder
whether the threshold was applied to approximated values.

HF is a sanity check, never ground truth, because it is not shape-invariant
(architecture doc section 7). It appears at the bottom, after the reference
comparison, in that framing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.corpus import PROMPTS, corpus_sha256  # noqa: E402
from engine import envlock  # noqa: E402
from engine.kernels.softmax import log_softmax  # noqa: E402
from engine.model.qwen3 import KVCache, Qwen3  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
DEFAULT_REFERENCE = REPO_ROOT / "reference"

HISTOGRAM_EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, float("inf")]


def quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.flatten().to(torch.float64)
    if flat.numel() == 0:
        return {"median": float("nan"), "p99": float("nan"), "max": float("nan"),
                "mean": float("nan")}
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p99": float(torch.quantile(flat, 0.99)) if flat.numel() < 16_000_000
        else float(flat.sort().values[int(0.99 * (flat.numel() - 1))]),
        "max": float(flat.max()),
    }


def histogram(values: torch.Tensor) -> list[tuple[str, int]]:
    flat = values.flatten().to(torch.float64)
    rows: list[tuple[str, int]] = []
    for lo, hi in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:]):
        count = int(((flat >= lo) & (flat < hi)).sum())
        label = f"[{lo:g}, {hi:g})" if hi != float("inf") else f"[{lo:g}, inf)"
        rows.append((label, count))
    return rows


def bar(count: int, total: int, width: int = 24) -> str:
    if total == 0:
        return ""
    filled = round(width * count / total)
    return "█" * filled


def compressed_kl(
    ref_top_logits: torch.Tensor,
    ref_full_lse: torch.Tensor,
    ref_tail_lse: torch.Tensor,
    engine_log_probs_at_top: torch.Tensor,
    engine_tail_log_mass: torch.Tensor,
) -> torch.Tensor:
    """KL(P_ref || Q_engine) over the {top-k} + {everything else} partition.

    Every quantity here is exact: the reference's top-k logits and its
    full-vocab and tail logsumexps are stored at fp64, and the engine's tail mass
    is computed from its own full logit row rather than by subtraction, so it
    does not lose precision when the head holds almost all the mass.
    """
    log_p = ref_top_logits - ref_full_lse[:, None]
    p = log_p.exp()
    head = (p * (log_p - engine_log_probs_at_top)).sum(dim=-1)

    log_p_tail = ref_tail_lse - ref_full_lse
    p_tail = log_p_tail.exp()
    tail = torch.where(
        p_tail > 0, p_tail * (log_p_tail - engine_tail_log_mass), torch.zeros_like(p_tail)
    )
    return head + tail


def exact_kl(ref_logits: torch.Tensor, engine_log_probs: torch.Tensor) -> torch.Tensor:
    log_p = ref_logits - torch.logsumexp(ref_logits, dim=-1, keepdim=True)
    p = log_p.exp()
    return (p * (log_p - engine_log_probs)).sum(dim=-1)


def run_engine_prefill(model: Qwen3, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Engine logits (fp16) and log-probs (fp32) for every prompt position.

    The cache is sized to the prompt rather than to model.max_len. A 4096-token
    cache is 470 MB across 28 layers, and allocating one per prompt to hold 30
    tokens is pure churn on a device that also holds the weights and a
    151936-wide logit tensor.
    """
    cache = KVCache(model.cfg, len(token_ids), model.device)
    ids = torch.tensor(token_ids, dtype=torch.long, device=model.device)
    logits = model.forward(ids, 0, cache)
    return logits, log_softmax(logits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=1e-2,
        help="fp64 top1-to-top2 logit gap below which a position is a near tie.",
    )
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="Report compressed-KL loss at several k, to choose the reference k.",
    )
    parser.add_argument("--hf-new-tokens", type=int, default=32)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    compressed_path = args.reference / "compressed.pt"
    if not compressed_path.is_file():
        raise SystemExit(
            f"{relpath(compressed_path)} is missing. Build it with:\n"
            "    OMP_NUM_THREADS=8 python3 scripts/build_fp64_reference.py --exact"
        )

    ref = torch.load(compressed_path, weights_only=False)
    if ref["corpus_sha256"] != corpus_sha256():
        raise SystemExit(
            "reference was built from a different corpus.\n"
            f"  reference  {ref['corpus_sha256'][:16]}\n"
            f"  corpus.py  {corpus_sha256()[:16]}\n"
            "Rebuild the reference; a fidelity number is scoped to its corpus."
        )

    exact_path = args.reference / "exact.pt"
    exact_ref = torch.load(exact_path, weights_only=False) if exact_path.is_file() else None

    model = Qwen3(args.weights)
    env = envlock.capture(corpus_sha256=corpus_sha256())

    print(f"lockstep fidelity  corpus sha256:{corpus_sha256()[:16]}  "
          f"positions={int(ref['entropy'].numel())}")
    print()
    print("reference")
    print("  method            fp64 CPU forward, exact per-row softmax")
    print("  weights           engine fp16 values upcast to fp64, not the bf16 checkpoint")
    print(f"  compression       top-{ref['top_k']} logits + full-vocab logsumexp, max, tail")
    print(f"  omp_num_threads   {ref['omp_num_threads']}  (pinned; reduction order depends on it)")
    entropy = ref["entropy"].to(torch.float64)
    print(f"  entropy (nats)    min {float(entropy.min()):.4f}  "
          f"median {float(entropy.median()):.4f}  max {float(entropy.max()):.4f}")

    cast = model.weight_cast_report()
    print(f"  weight cast       {cast['checkpoint_dtype']} -> {cast['engine_dtype']}, "
          f"{cast['elements_not_exactly_representable']}/{cast['elements']} inexact, "
          f"max abs {cast['max_abs_error']:.3e}")
    print("                    reported separately: F1 measures the forward pass, not the cast")

    # ---- engine pass over every prompt --------------------------------------
    abs_err_parts: list[torch.Tensor] = []
    rel_err_parts: list[torch.Tensor] = []
    kl_parts: list[torch.Tensor] = []
    engine_top1: list[torch.Tensor] = []
    engine_log_prob_rows: dict[int, torch.Tensor] = {}

    offset = 0
    exact_positions = (
        set(exact_ref["positions"].tolist()) if exact_ref is not None else set()
    )

    # A 727-position prompt is 727 x 151936; one fp64 copy of that is 884 MB and
    # the tail-mass computation needs two live at once. Chunking the fp64 work
    # over rows keeps the peak near 150 MB, which matters on 8 GB alongside the
    # engine's own weights and KV cache.
    ROW_CHUNK = 64

    for prompt, token_ids in zip(PROMPTS, ref["prompt_token_ids"]):
        seq_len = len(token_ids)
        logits, log_probs = run_engine_prefill(model, token_ids)
        engine_top1.append(logits.argmax(dim=-1).cpu())

        for start in range(0, seq_len, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, seq_len)
            rows = slice(offset + start, offset + stop)

            top_idx = ref["top_indices"][rows].to(torch.int64).to(model.device)
            ref_top = ref["top_logits"][rows].to(torch.float64)

            chunk64 = logits[start:stop].to(torch.float64)
            engine_at_top = torch.gather(chunk64, 1, top_idx).cpu()
            abs_err = (engine_at_top - ref_top).abs()
            abs_err_parts.append(abs_err)
            rel_err_parts.append(abs_err / ref_top.abs().clamp_min(1e-12))

            lp = log_probs[start:stop].to(torch.float64)
            engine_lp_at_top = torch.gather(lp, 1, top_idx).cpu()

            # Engine tail mass from its own full row, by masking the reference's
            # retained ids. Exact even when the head carries almost everything.
            engine_full_lse = torch.logsumexp(chunk64, dim=-1)
            masked = chunk64.scatter_(1, top_idx, float("-inf"))
            engine_tail_log_mass = (torch.logsumexp(masked, dim=-1) - engine_full_lse).cpu()

            kl_parts.append(
                compressed_kl(
                    ref_top,
                    ref["full_logsumexp"][rows].to(torch.float64),
                    ref["tail_logsumexp"][rows].to(torch.float64),
                    engine_lp_at_top,
                    engine_tail_log_mass,
                )
            )

            for local in range(start, stop):
                if offset + local in exact_positions:
                    engine_log_prob_rows[offset + local] = lp[local - start].cpu().clone()

            del chunk64, lp, masked

        offset += seq_len
        del logits, log_probs
        torch.cuda.empty_cache()

    abs_err = torch.cat(abs_err_parts)
    rel_err = torch.cat(rel_err_parts)
    kl_compressed = torch.cat(kl_parts)
    engine_argmax = torch.cat(engine_top1)

    # ---- logit error --------------------------------------------------------
    abs_stats = quantiles(abs_err)
    rel_stats = quantiles(rel_err)
    print()
    print(f"logit error, engine fp16 vs fp64 reference, over top-{ref['top_k']} "
          f"({abs_err.numel()} values)")
    print(f"  absolute   max {abs_stats['max']:.6e}  p99 {abs_stats['p99']:.6e}  "
          f"median {abs_stats['median']:.6e}")
    print(f"  relative   max {rel_stats['max']:.6e}  p99 {rel_stats['p99']:.6e}  "
          f"median {rel_stats['median']:.6e}")
    print("  ULP is not reported: it is meaningless across dtypes")

    print()
    print("  absolute error histogram")
    rows = histogram(abs_err)
    total = abs_err.numel()
    for label, count in rows:
        print(f"    {label:<16} {bar(count, total):<24} {count:>10}")

    # ---- KL -----------------------------------------------------------------
    print()
    print("per-position KL(fp64 reference || engine), nats")
    stats = quantiles(kl_compressed)

    delta_summary: dict[str, float] | None = None
    if exact_ref is None:
        print(f"  compressed, top-{ref['top_k']}, {kl_compressed.numel()} positions")
        print(f"    mean {stats['mean']:.6e}  median {stats['median']:.6e}  "
              f"max {stats['max']:.6e}")
        print("  exact,      not built     rerun the reference builder with --exact")
        print("    without it the bound's looseness is asserted rather than measured,")
        print("    and it was measured at 27% for k=256, so it is not a small thing")
    else:
        positions = exact_ref["positions"]
        ref_full = exact_ref["logits"].to(torch.float64)
        engine_lp = torch.stack([engine_log_prob_rows[int(p)] for p in positions]).to(
            torch.float64
        )
        kl_exact = exact_kl(ref_full, engine_lp)
        kl_comp_subset = kl_compressed[positions]
        delta = kl_exact - kl_comp_subset

        exact_stats = quantiles(kl_exact)
        delta_stats = quantiles(delta)

        # Per-position ratios are unusable on their own: at a position where the
        # engine nearly matches the reference, exact KL approaches zero and any
        # fixed floor turns the ratio into whatever the floor was. The aggregate
        # ratio is the honest summary, since it weights each position by how much
        # KL it actually carries. The per-position ratio is reported alongside it
        # but only over positions carrying at least the median KL, with the
        # denominator stated.
        aggregate = float(delta.sum() / kl_exact.sum())
        carries = kl_exact >= kl_exact.median()
        rel_delta = (delta[carries] / kl_exact[carries]).abs()
        rel_stats_delta = quantiles(rel_delta)

        print(f"  exact, full vocab, {kl_exact.numel()} positions   <- headline")
        print(f"    mean {exact_stats['mean']:.6e}  median {exact_stats['median']:.6e}  "
              f"max {exact_stats['max']:.6e}")
        print(f"  compressed, top-{ref['top_k']}, {kl_compressed.numel()} positions"
              f"   <- lower bound, wider coverage")
        print(f"    mean {stats['mean']:.6e}  median {stats['median']:.6e}  "
              f"max {stats['max']:.6e}")
        print("    a bound, not an estimate: coarsening the partition to top-k plus")
        print("    one tail bin can only decrease KL (log-sum inequality)")
        print(f"  how loose that bound is, on the same {kl_exact.numel()} positions")
        print(f"    exact minus compressed   mean {delta_stats['mean']:.6e}  "
              f"max {delta_stats['max']:.6e}")
        print(f"    aggregate KL missed      {aggregate:.4%}  "
              f"(sum of delta / sum of exact)")
        print(f"    per position, over the {int(carries.sum())} carrying at least "
              f"the median KL")
        print(f"      fraction of exact      median {rel_stats_delta['median']:.4%}  "
              f"max {rel_stats_delta['max']:.4%}")
        verdict = (
            f"top-{ref['top_k']} compression is defensible on this corpus"
            if aggregate < 0.01
            else f"the bound understates true KL by {aggregate:.1%}; exact KL is the headline"
        )
        print(f"    verdict                  {verdict}")
        delta_summary = {
            "exact_mean": exact_stats["mean"],
            "delta_mean": delta_stats["mean"],
            "delta_max": delta_stats["max"],
            "aggregate_fraction_missed": aggregate,
            "relative_delta_median_over_carrying": rel_stats_delta["median"],
            "relative_delta_max_over_carrying": rel_stats_delta["max"],
        }

        if args.k_sweep:
            print()
            print("  choosing k: compressed KL at each k against the same exact KL")
            print(f"    {'k':>7}  {'aggregate KL missed':>21}  {'max per-position':>17}")
            sweep = []
            ref_lse = torch.logsumexp(ref_full, dim=-1)
            for k in (256, 512, 1024, 2048, 4096, 8192):
                values, indices = torch.topk(ref_full, k, dim=-1)
                masked = ref_full.scatter(-1, indices, float("-inf"))
                tail_lse = torch.logsumexp(masked, dim=-1)
                lp_at = torch.gather(engine_lp, -1, indices)
                eng_masked = engine_lp.scatter(-1, indices, float("-inf"))
                eng_tail = torch.logsumexp(eng_masked, dim=-1)
                kl_k = compressed_kl(values, ref_lse, tail_lse, lp_at, eng_tail)
                miss = float((kl_exact - kl_k).sum() / kl_exact.sum())
                per = (kl_exact - kl_k)[carries] / kl_exact[carries]
                sweep.append({"k": k, "aggregate_missed": miss, "max_per_position": float(per.max())})
                print(f"    {k:>7}  {miss:>20.4%}  {float(per.max()):>16.4%}")
            delta_summary["k_sweep"] = sweep

    # ---- greedy token match -------------------------------------------------
    ref_top1 = ref["top_indices"][:, 0].to(torch.int64)
    gap = (ref["top_logits"][:, 0] - ref["top_logits"][:, 1]).to(torch.float64)
    near_tie = gap < args.near_tie_threshold

    agree = engine_argmax == ref_top1
    clean = ~near_tie
    clean_match = int((agree & clean).sum())
    clean_total = int(clean.sum())
    tie_match = int((agree & near_tie).sum())
    tie_total = int(near_tie.sum())

    print()
    print("greedy token match, engine argmax vs fp64 reference argmax")
    print(f"  near-tie threshold        {args.near_tie_threshold:.1e} nats, "
          f"fp64 top1-to-top2 gap")
    print("  near-tie test is exact:   it reads only fp64 top-1 and top-2, both "
          "retained uncompressed")
    print(f"  excluded as near ties     {tie_total}/{int(near_tie.numel())}")
    print(f"  match, near ties excluded {clean_match}/{clean_total}")
    print(f"  match, near ties only     {tie_match}/{tie_total}")

    print()
    print("  sensitivity to the threshold")
    sensitivity = []
    for threshold in (0.0, 1e-3, 1e-2, 1e-1):
        mask = gap >= threshold
        matched = int((agree & mask).sum())
        kept = int(mask.sum())
        sensitivity.append({"threshold": threshold, "match": matched, "total": kept})
        print(f"    gap >= {threshold:<8.0e}  {matched}/{kept}")

    # ---- HF sanity check ----------------------------------------------------
    hf_summary: dict[str, object] | None = None
    if not args.skip_hf:
        hf_summary = hf_sanity_check(model, args)

    fingerprint = env.fingerprint()
    print()
    print(f"env  {fingerprint}")

    if not args.no_artifact:
        artifact = Artifact(
            kind="fidelity",
            env=env,
            payload={
                "corpus_sha256": corpus_sha256(),
                "positions": int(kl_compressed.numel()),
                "top_k": int(ref["top_k"]),
                "omp_num_threads": int(ref["omp_num_threads"]),
                "weight_cast": cast,
                "logit_abs_error": abs_stats,
                "logit_rel_error": rel_stats,
                "abs_error_histogram": [
                    {"bin": label, "count": count} for label, count in rows
                ],
                "kl_compressed": stats,
                "kl_method_delta": delta_summary,
                "near_tie_threshold": args.near_tie_threshold,
                "near_tie_excluded": tie_total,
                "match_excluding_near_ties": {"matched": clean_match, "total": clean_total},
                "match_near_ties_only": {"matched": tie_match, "total": tie_total},
                "threshold_sensitivity": sensitivity,
                "hf_sanity_check": hf_summary,
            },
        )
        path = artifact.write()
        print(f"artifact  {relpath(path)}")

    return 0


def hf_sanity_check(model: Qwen3, args: argparse.Namespace) -> dict[str, object]:
    """Greedy decode, engine versus HF. Exercises the decode path, not prefill.

    HF is not shape-invariant and is not ground truth. A mismatch here is a lead,
    not a verdict; the fp64 comparison above is the measurement. It is here
    because week 1's exit criterion is a working greedy batch-1 decode, and a
    prefill-only comparison would never touch the single-token KV-cache path.
    """
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.weights))
    eos = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    engine_runs: dict[str, list[int]] = {}
    for prompt in PROMPTS:
        ids = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        generated, _ = model.generate_greedy(ids, args.hf_new_tokens, eos_token_ids=eos)
        engine_runs[prompt.uid] = generated

    del model.weights
    gc.collect()
    torch.cuda.empty_cache()

    # .to("cuda") rather than device_map="cuda": device_map pulls in accelerate,
    # which is not in the dependency set the security doc fixes, and a single-GPU
    # 0.6B model does not need a dispatch plan.
    hf = (
        AutoModelForCausalLM.from_pretrained(str(args.weights), torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )

    exact_sequences = 0
    matched_tokens = 0
    total_tokens = 0
    first_divergence: list[int] = []

    for prompt in PROMPTS:
        ids = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = hf.generate(
                torch.tensor([ids], device="cuda"),
                max_new_tokens=args.hf_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_generated = out[0, len(ids):].tolist()
        ours = engine_runs[prompt.uid]

        pairs = list(zip(ours, hf_generated))
        matched = sum(1 for a, b in pairs if a == b)
        matched_tokens += matched
        total_tokens += max(len(ours), len(hf_generated))
        if ours == hf_generated:
            exact_sequences += 1
        else:
            diverged_at = next(
                (i for i, (a, b) in enumerate(pairs) if a != b), len(pairs)
            )
            first_divergence.append(diverged_at)

    print()
    print("HF sanity check  (not ground truth; HF is not shape-invariant)")
    print(f"  greedy decode, {args.hf_new_tokens} new tokens, both fp16 on the same GPU")
    print(f"  identical sequences       {exact_sequences}/{len(PROMPTS)}")
    print(f"  identical tokens          {matched_tokens}/{total_tokens}")
    if first_divergence:
        median_div = sorted(first_divergence)[len(first_divergence) // 2]
        print(f"  median first divergence   token {median_div} of {args.hf_new_tokens}")
    print("  differences are expected: this engine keeps fp32 through the RMSNorm")
    print("  weight multiply and rotates by fp32 cos/sin, where HF rounds both to fp16")

    return {
        "new_tokens": args.hf_new_tokens,
        "identical_sequences": exact_sequences,
        "prompts": len(PROMPTS),
        "identical_tokens": matched_tokens,
        "total_tokens": total_tokens,
        "first_divergence_positions": first_divergence,
    }


if __name__ == "__main__":
    raise SystemExit(main())
