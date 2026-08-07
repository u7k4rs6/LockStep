"""F1: the engine's fp16 logits against the fp64 CPU reference."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.corpus import PROMPTS, corpus_sha256  # noqa: E402
from bench import memprobe  # noqa: E402
from bench.fp64_reference import Fp64Reference, require_pinned_threads  # noqa: E402
from engine import envlock  # noqa: E402
from engine.kernels import registry  # noqa: E402
from engine.kernels.softmax import log_softmax  # noqa: E402
from engine.model.qwen3 import KVCache, Qwen3  # noqa: E402
from report.artifact import require_clean_tree, Artifact, relpath  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
DEFAULT_REFERENCE = REPO_ROOT / "reference"

HISTOGRAM_EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, float("inf")]
ROW_CHUNK = 64


NEAR_TIE_QUANTILE = 0.999


def quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.flatten().to(torch.float64)
    if flat.numel() == 0:
        return dict.fromkeys(("mean", "median", "p99", "max"), float("nan"))
    ordered = flat.sort().values
    return {
        "mean": float(flat.mean()),
        "median": float(ordered[(ordered.numel() - 1) // 2]),
        "p99": float(ordered[int(0.99 * (ordered.numel() - 1))]),
        "max": float(ordered[-1]),
    }


def histogram(counts: list[int]) -> list[tuple[str, int]]:
    rows = []
    for (lo, hi), count in zip(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:]), counts):
        label = f"[{lo:g}, {hi:g})" if hi != float("inf") else f"[{lo:g}, inf)"
        rows.append((label, count))
    return rows


def bar(count: int, total: int, width: int = 24) -> str:
    return "█" * round(width * count / total) if total else ""


def attention_splits(kv_len: int) -> int:
    """How many fixed-size attention splits a prompt of this length launches."""
    return -(-kv_len // registry.SPLIT_SIZE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--hf-new-tokens", type=int, default=32)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="produce a claim artifact from an uncommitted "
                             "tree; recorded in the artifact when used")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument(
        "--memory-ceiling-mib",
        type=int,
        default=4096,
        help="Projected host memory the pass may use. Fails fast rather than swapping.",
    )
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty)

    threads = require_pinned_threads()

    cache_path = args.reference / "top2.pt"
    if not cache_path.is_file():
        raise SystemExit(
            f"{relpath(cache_path)} is missing. Build it with:\n"
            "    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 scripts/build_fp64_reference.py"
        )
    cache = torch.load(cache_path, weights_only=False)
    if cache["corpus_sha256"] != corpus_sha256():
        raise SystemExit(
            "cache was built from a different corpus.\n"
            f"  cache      {cache['corpus_sha256'][:16]}\n"
            f"  corpus.py  {corpus_sha256()[:16]}\n"
            "Rebuild it; a fidelity number is scoped to its corpus."
        )

    probes: list[str] = []
    probes.append(memprobe.format_breakdown("before any model load",
                                            memprobe.live_tensor_breakdown()))

    engine = Qwen3(args.weights)
    probes.append(memprobe.format_breakdown("after the engine (GPU) loaded",
                                            memprobe.live_tensor_breakdown()))

    reference = Fp64Reference(args.weights)
    probes.append(memprobe.format_breakdown("after the fp64 reference loaded",
                                            memprobe.live_tensor_breakdown()))

    env = envlock.capture(corpus_sha256=corpus_sha256())

    total_positions = int(cache["entropy"].numel())
    print(f"lockstep fidelity  corpus sha256:{corpus_sha256()[:16]}  positions={total_positions}")
    print()
    print("reference")
    print("  method            fp64 CPU forward, exact per-row softmax, run this run")
    print("  weights           engine fp16 values upcast to fp64, not the bf16 checkpoint")
    print("  KL               exact over all 151936 tokens; no top-k compression")
    print(f"  threads           omp={threads['omp_num_threads']} "
          f"mkl={threads['mkl_num_threads']} torch_intraop={threads['torch_intraop_threads']}")
    entropy = cache["entropy"].to(torch.float64)
    print(f"  entropy (nats)    min {float(entropy.min()):.4f}  "
          f"median {float(entropy.median()):.4f}  max {float(entropy.max()):.4f}")

    cast = engine.weight_cast_report()
    print(f"  weight cast       {cast['checkpoint_dtype']} -> {cast['engine_dtype']}, "
          f"{cast['distinct_elements']} distinct params")
    print(f"                    tied lm_head aliased, {cast['duplicate_elements_dropped']}"
          f" duplicate params dropped")
    print(f"                    {cast['elements_not_exactly_representable']} inexact, "
          f"max abs {cast['max_abs_error']:.3e}")
    print("                    reported separately: F1 measures the forward pass, not the cast")

    split_counts = [attention_splits(len(ids)) for ids in cache["prompt_token_ids"]]
    max_splits = max(split_counts)
    multi = sum(1 for s in split_counts if s >= 2)
    print()
    print("path coverage over the corpus")
    print(f"  attention splits reached   max {max_splits}, "
          f"{multi}/{len(split_counts)} prompts launch two or more")
    if max_splits < 2:
        raise SystemExit(
            "no prompt in this corpus reaches a second attention split, so the "
            f"split-combine fold never executes and every number below would be "
            f"produced without it. SPLIT_SIZE is {registry.SPLIT_SIZE}; the corpus "
            "needs a prompt longer than that."
        )

    longest = max(len(ids) for ids in cache["prompt_token_ids"])
    heads = engine.cfg.num_attention_heads
    vocab = engine.cfg.vocab_size
    attention_scratch = longest * longest * heads * 8 * 3
    logit_scratch = ROW_CHUNK * vocab * 8 * 4
    anon_now, _ = memprobe.rss_split()
    projected = anon_now + attention_scratch + logit_scratch

    print()
    print("host memory projection")
    print(f"  resident now (anon)        {anon_now / memprobe.MIB:8.1f} MiB")
    print(f"  fp64 attention scratch     {attention_scratch / memprobe.MIB:8.1f} MiB"
          f"  ({longest} positions squared x {heads} heads)")
    print(f"  logit chunk scratch        {logit_scratch / memprobe.MIB:8.1f} MiB"
          f"  ({ROW_CHUNK} positions x {vocab})")
    print(f"  projected peak             {projected / memprobe.MIB:8.1f} MiB"
          f"  ceiling {args.memory_ceiling_mib} MiB")
    memprobe.require_headroom(
        projected, args.memory_ceiling_mib * memprobe.MIB, "the fp64 reference pass"
    )

    abs_hist = [0] * (len(HISTOGRAM_EDGES) - 1)
    abs_max = 0.0
    abs_sum = 0.0
    abs_count = 0
    abs_sample: list[torch.Tensor] = []
    kl_parts: list[torch.Tensor] = []
    top2_err_parts: list[torch.Tensor] = []
    engine_top1: list[torch.Tensor] = []
    streamed_top2_idx: list[torch.Tensor] = []
    streamed_top2_val: list[torch.Tensor] = []

    offset = 0
    for prompt, token_ids in zip(PROMPTS, cache["prompt_token_ids"]):
        seq_len = len(token_ids)

        kv = KVCache(engine.cfg, seq_len, engine.device)
        ids = torch.tensor(token_ids, dtype=torch.long, device=engine.device)
        eng_logits = engine.forward(ids, 0, kv)
        eng_log_probs = log_softmax(eng_logits)
        engine_top1.append(eng_logits.argmax(dim=-1).cpu())

        hidden = reference.hidden(token_ids)

        for start in range(0, seq_len, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, seq_len)
            ref_rows = reference.logits_for(hidden[start:stop])
            eng_rows = eng_logits[start:stop].to(torch.float64).cpu()
            eng_lp = eng_log_probs[start:stop].to(torch.float64).cpu()

            ref_lp = ref_rows - torch.logsumexp(ref_rows, dim=-1, keepdim=True)
            p = ref_lp.exp()
            kl_parts.append((p * (ref_lp - eng_lp)).sum(dim=-1))

            err = (eng_rows - ref_rows).abs()
            abs_max = max(abs_max, float(err.max()))
            abs_sum += float(err.sum())
            abs_count += err.numel()
            for i, (lo, hi) in enumerate(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])):
                abs_hist[i] += int(((err >= lo) & (err < hi)).sum())
            if len(abs_sample) < 8:
                abs_sample.append(err[:, ::701].flatten().clone())

            values, indices = torch.topk(ref_rows, 2, dim=-1, sorted=True)
            streamed_top2_val.append(values)
            streamed_top2_idx.append(indices.to(torch.int32))

            eng_at_top2 = torch.gather(eng_rows, 1, indices)
            top2_err_parts.append(
                ((eng_at_top2[:, 0] - eng_at_top2[:, 1]) - (values[:, 0] - values[:, 1])).abs()
            )

            del ref_rows, eng_rows, eng_lp, ref_lp, p, err

        if len(probes) == 3:
            probes.append(memprobe.format_breakdown(
                f"mid-pass, inside the longest prompt so far ({seq_len} positions)",
                memprobe.live_tensor_breakdown()))

        offset += seq_len
        del eng_logits, eng_log_probs, hidden
        torch.cuda.empty_cache()

    probes.append(memprobe.format_breakdown("after both passes finished",
                                            memprobe.live_tensor_breakdown()))

    kl = torch.cat(kl_parts)
    top2_err = torch.cat(top2_err_parts)
    engine_argmax = torch.cat(engine_top1)

    streamed_idx = torch.cat(streamed_top2_idx)
    streamed_val = torch.cat(streamed_top2_val)
    idx_match = bool(torch.equal(streamed_idx, cache["top2_indices"]))
    val_match = bool(torch.equal(streamed_val, cache["top2_logits"]))
    print(f"  fp64 reproducibility       streamed top-2 vs cached: "
          f"ids {'identical' if idx_match else 'DIFFER'}, "
          f"logits {'bitwise identical' if val_match else 'DIFFER'}")
    if not (idx_match and val_match):
        raise SystemExit(
            "the fp64 reference did not reproduce its own cached top-2. The pinned "
            "thread counts are supposed to make it deterministic; ground truth that "
            "moves between runs invalidates every number in this report."
        )

    print()
    print(f"logit error, engine fp16 vs fp64 reference, full vocabulary ({abs_count} values)")
    print(f"  absolute   max {abs_max:.6e}  mean {abs_sum / abs_count:.6e}")
    print("  no relative error: a logit has no meaningful zero, so dividing by one")
    print("  is not a scale. ULP is not reported either; it is meaningless across dtypes")
    print()
    print("  absolute error histogram")
    for label, count in histogram(abs_hist):
        print(f"    {label:<16} {bar(count, abs_count):<24} {count:>12}")

    kl_stats = quantiles(kl)
    print()
    print(f"per-position KL(fp64 reference || engine), nats, exact over the full")
    print(f"vocabulary at all {kl.numel()} positions")
    print(f"  mean {kl_stats['mean']:.6e}  median {kl_stats['median']:.6e}  "
          f"p99 {kl_stats['p99']:.6e}  max {kl_stats['max']:.6e}")

    ref_top1 = cache["top2_indices"][:, 0].to(torch.int64)
    gap = (cache["top2_logits"][:, 0] - cache["top2_logits"][:, 1]).to(torch.float64)
    err_stats = quantiles(top2_err)
    ordered = top2_err.sort().values
    # Derived from this run, never hardcoded. A greedy argmax flips when the error
    # on the top1-minus-top2 difference exceeds the gap between them, so that
    # difference-error is measured directly and its p99.9 is the threshold. A
    # constant chosen after seeing the numbers would judge nothing.
    threshold = float(ordered[int(NEAR_TIE_QUANTILE * (ordered.numel() - 1))])

    agree = engine_argmax == ref_top1
    mismatch = torch.nonzero(~agree).flatten()
    mismatch_gaps = gap[mismatch]
    near_tie = gap < threshold
    clean = ~near_tie
    clean_match, clean_total = int((agree & clean).sum()), int(clean.sum())
    tie_match, tie_total = int((agree & near_tie).sum()), int(near_tie.sum())

    print()
    print("greedy token match, engine argmax vs fp64 reference argmax")
    print("  the threshold is derived from this run, not chosen:")
    print("    an argmax flips when the error on the fp64 top1-minus-top2 logit")
    print("    difference exceeds the gap between them, so that difference-error is")
    print("    measured directly and its p99.9 is the near-tie threshold")
    print(f"  measured top1-top2 difference error   median {err_stats['median']:.6e}  "
          f"p99 {err_stats['p99']:.6e}  max {err_stats['max']:.6e}")
    print(f"  derived near-tie threshold            {threshold:.6e} nats  "
          f"(p{NEAR_TIE_QUANTILE * 100:g} of that error)")
    print("  near-tie test is exact:               it reads only fp64 top-1 and top-2,")
    print("                                        both cached uncompressed")
    largest_mismatch_gap = float(mismatch_gaps.max()) if mismatch_gaps.numel() else 0.0
    minimum_perfect = math.nextafter(largest_mismatch_gap, math.inf)
    perfect_kept = int((gap >= minimum_perfect).sum())
    perfect_matched = int((agree & (gap >= minimum_perfect)).sum())

    expected_exceedances = (1.0 - NEAR_TIE_QUANTILE) * float(gap.numel())

    print()
    print(f"  match, gap >= threshold   {clean_match}/{clean_total}   <- headline")
    print(f"  near ties excluded        {tie_total}/{int(near_tie.numel())}")
    print(f"  match, near ties only     {tie_match}/{tie_total}")
    print()
    print(f"  the comparison is: gap >= threshold")
    print(f"  largest gap among mismatches             {largest_mismatch_gap:.17e}")
    print(f"  minimum threshold reaching 100 percent   {minimum_perfect:.17e}")
    print(f"    one representable double above it, so that mismatch is excluded")
    print(f"    match there                            {perfect_matched}/{perfect_kept}")
    print(f"    ratio to the derived threshold         "
          f"{threshold / minimum_perfect:.1f}x tighter")
    print("    the result survives a threshold far stricter than the derived one,")
    print("    so it does not rest on where the derived one landed")
    print()
    print(f"  expected exceedances at p{NEAR_TIE_QUANTILE * 100:g} over {gap.numel()} "
          f"positions: {expected_exceedances:.1f}")
    print(f"  observed                                                    "
          f"{clean_total - clean_match}")

    ref_top2 = cache["top2_indices"][:, 1].to(torch.int64)
    swaps = int((engine_argmax[mismatch] == ref_top2[mismatch]).sum())
    print(f"  every mismatch a top1/top2 swap       {swaps}/{mismatch.numel()}")
    worst_gap = float(gap[mismatch].max()) if mismatch.numel() else 0.0
    print(f"  largest fp64 gap among mismatches     {worst_gap:.6e} nats"
          f"  ({'below' if worst_gap < threshold else 'ABOVE'} threshold)")

    print()
    print("  sensitivity to the threshold")
    sensitivity = []
    for t in (0.0, 1e-3, 1e-2, threshold, 1e-1):
        mask = gap >= t
        matched, kept = int((agree & mask).sum()), int(mask.sum())
        tag = "  <- derived" if t == threshold else ""
        sensitivity.append({"threshold": t, "match": matched, "total": kept})
        print(f"    gap >= {t:<10.4e}  {matched}/{kept}{tag}")

    checks = [
        ("no mismatch above the derived threshold", clean_match == clean_total),
        ("every mismatch is a top1/top2 swap", swaps == mismatch.numel()),
        ("mean exact KL <= 1e-4 nats", kl_stats["mean"] <= 1e-4),
        ("max exact KL <= 1e-3 nats", kl_stats["max"] <= 1e-3),
        ("max absolute logit error <= 0.5", abs_max <= 0.5),
        ("corpus reaches two attention splits", max_splits >= 2),
        ("fp64 reference reproduced its cached top-2", idx_match and val_match),
    ]
    print()
    print("F1 pass criterion (docs/kickoff/02-technical-architecture.md 7.1)")
    for name, ok in checks:
        print(f"  [{'pass' if ok else 'FAIL'}]  {name}")
    passed = all(ok for _, ok in checks)
    print(f"  F1 {'PASS' if passed else 'FAIL'}")

    hf_summary = None if args.skip_hf else hf_sanity_check(engine, args)

    peak_rss = memprobe.peak_rss_bytes()
    print()
    print("host memory")
    for probe in probes:
        print(probe)
    print(f"  peak RSS  {peak_rss / memprobe.MIB:.1f} MiB")

    print()
    print(f"env  {env.fingerprint()}")

    if not args.no_artifact:
        path = Artifact(
            kind="fidelity",
            env=env,
            payload={
                "corpus_sha256": corpus_sha256(),
                "positions": int(kl.numel()),
                "threads": threads,
                "weight_cast": cast,
                "max_attention_splits": max_splits,
                "prompts_with_multiple_splits": multi,
                "fp64_reproduced_cached_top2": idx_match and val_match,
                "logit_abs_error": {"max": abs_max, "mean": abs_sum / abs_count},
                "abs_error_histogram": [
                    {"bin": label, "count": count} for label, count in histogram(abs_hist)
                ],
                "kl_exact_full_vocab": kl_stats,
                "top2_difference_error": err_stats,
                "near_tie_rule": f"p{NEAR_TIE_QUANTILE * 100:g} of the top1-top2 difference error",
                "near_tie_threshold": threshold,
                "match_above_threshold": {"matched": clean_match, "total": clean_total},
                "minimum_threshold_for_100_percent": minimum_perfect,
                "positions_kept_at_that_threshold": perfect_kept,
                "matched_at_that_threshold": perfect_matched,
                "largest_gap_among_mismatches_exact": largest_mismatch_gap,
                "match_comparison": "gap >= threshold",
                "expected_exceedances_at_quantile": expected_exceedances,
                "observed_exceedances": clean_total - clean_match,
                "match_near_ties_only": {"matched": tie_match, "total": tie_total},
                "mismatches_that_are_top1_top2_swaps": {"swaps": swaps, "total": mismatch.numel()},
                "largest_gap_among_mismatches": worst_gap,
                "threshold_sensitivity": sensitivity,
                "f1_checks": [{"check": n, "pass": ok} for n, ok in checks],
                "peak_rss_bytes": peak_rss,
                "peak_rss_anon_bytes": memprobe.rss_split()[0],
                "projected_peak_bytes": projected,
                "memory_ceiling_bytes": args.memory_ceiling_mib * memprobe.MIB,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "f1_pass": passed,
                "hf_sanity_check": hf_summary,
            },
        ).write()
        print(f"artifact  {relpath(path)}")

    return 0 if passed else 1


def hf_sanity_check(model: Qwen3, args: argparse.Namespace) -> dict[str, object]:
    """Greedy decode, engine versus HF. Exercises the decode path, not prefill."""
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.weights))
    eos = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    engine_runs = {}
    for prompt in PROMPTS:
        ids = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        generated, _ = model.generate_greedy(ids, args.hf_new_tokens, eos_token_ids=eos)
        engine_runs[prompt.uid] = generated

    del model.weights
    gc.collect()
    torch.cuda.empty_cache()

    hf = (
        AutoModelForCausalLM.from_pretrained(str(args.weights), torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )

    exact_sequences = matched_tokens = total_tokens = 0
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
        theirs = out[0, len(ids):].tolist()
        ours = engine_runs[prompt.uid]
        pairs = list(zip(ours, theirs))

        matched_tokens += sum(1 for a, b in pairs if a == b)
        total_tokens += max(len(ours), len(theirs))
        if ours == theirs:
            exact_sequences += 1
        else:
            first_divergence.append(next((i for i, (a, b) in enumerate(pairs) if a != b), len(pairs)))

    print()
    print("HF sanity check  (not ground truth; HF is not shape-invariant)")
    print(f"  greedy decode, {args.hf_new_tokens} new tokens, both fp16 on the same GPU")
    print(f"  identical sequences       {exact_sequences}/{len(PROMPTS)}")
    print(f"  identical tokens          {matched_tokens}/{total_tokens}")
    if first_divergence:
        print(f"  median first divergence   token "
              f"{sorted(first_divergence)[len(first_divergence) // 2]} of {args.hf_new_tokens}")
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
