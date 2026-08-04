"""F1: the engine's fp16 logits against the fp64 CPU reference.

docs/02-technical-architecture.md section 7 names the reported metrics and
section 7.1 states the pass criterion this script decides.

KL is exact, over the full 151936-token vocabulary, at every position. There is
no compressed variant. An earlier version cached top-k fp64 logits so KL could be
computed without re-running the fp64 pass; the measured sweep showed top-k KL
missing 27.4 percent of the true value at k=256 and still 4.0 percent at k=8192,
closing only about 1.3x per doubling, so the cache was buying a loose lower bound
in place of a number the fp64 pass produces exactly in about two minutes. Both
passes now run in the same process and KL is accumulated in row chunks, so
nothing large is written or held.

**The near-tie threshold is derived, not chosen.** A greedy argmax flips when the
error on the top-1-minus-top-2 logit *difference* exceeds the fp64 gap between
them, so that difference-error is measured directly in this run and its 99.9th
percentile becomes the threshold. Positions whose fp64 gap falls below it are
near ties: rounding alone can reorder them, so a mismatch there carries no
information. A mismatch *above* it cannot be explained by measured rounding and
is a real signal. The previous hardcoded 1e-2 was below the median of this
distribution, which made it indefensible.

HF is a sanity check, never ground truth, because it is not shape-invariant
(architecture doc section 7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.corpus import PROMPTS, corpus_sha256  # noqa: E402
from bench.fp64_reference import Fp64Reference, require_pinned_threads  # noqa: E402
from engine import envlock  # noqa: E402
from engine.kernels import registry  # noqa: E402
from engine.kernels.softmax import log_softmax  # noqa: E402
from engine.model.qwen3 import KVCache, Qwen3  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
DEFAULT_REFERENCE = REPO_ROOT / "reference"

HISTOGRAM_EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, float("inf")]
ROW_CHUNK = 64

# Relative error is only meaningful where the denominator has magnitude. Most of
# a 151936-wide logit row sits near zero, so an unfloored relative error reports
# 1e6 for a 1e-3 absolute miss on a logit of 1e-9.
REL_ERROR_FLOOR = 1.0

# The rule, stated once here and printed in the output. See the module docstring.
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
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

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

    engine = Qwen3(args.weights)
    reference = Fp64Reference(args.weights)
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

    # -- B5: coverage of the paths the measurement is supposed to exercise ----
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

    # -- the two passes, streamed together ------------------------------------
    abs_hist = [0] * (len(HISTOGRAM_EDGES) - 1)
    abs_max = 0.0
    abs_sum = 0.0
    abs_count = 0
    rel_max = 0.0
    rel_count = 0
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
            ref_rows = reference.logits_for(hidden[start:stop])  # [chunk, vocab] fp64
            eng_rows = eng_logits[start:stop].to(torch.float64).cpu()
            eng_lp = eng_log_probs[start:stop].to(torch.float64).cpu()

            # Exact KL(P_ref || Q_engine) over the whole vocabulary.
            ref_lp = ref_rows - torch.logsumexp(ref_rows, dim=-1, keepdim=True)
            p = ref_lp.exp()
            kl_parts.append((p * (ref_lp - eng_lp)).sum(dim=-1))

            # Logit error, over the whole vocabulary rather than a top-k slice.
            err = (eng_rows - ref_rows).abs()
            abs_max = max(abs_max, float(err.max()))
            abs_sum += float(err.sum())
            abs_count += err.numel()
            # Relative error only where the reference logit carries magnitude.
            # Over the full vocabulary most logits pass near zero, and dividing
            # by one of those reports 1e6 for an absolute error of 1e-3. The
            # floor is stated in the output rather than buried in a clamp.
            big = ref_rows.abs() >= REL_ERROR_FLOOR
            if bool(big.any()):
                rel_max = max(rel_max, float((err[big] / ref_rows.abs()[big]).max()))
                rel_count += int(big.sum())
            for i, (lo, hi) in enumerate(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])):
                abs_hist[i] += int(((err >= lo) & (err < hi)).sum())
            if len(abs_sample) < 8:
                abs_sample.append(err[:, ::701].flatten().clone())

            values, indices = torch.topk(ref_rows, 2, dim=-1, sorted=True)
            streamed_top2_val.append(values)
            streamed_top2_idx.append(indices.to(torch.int32))

            # The quantity that actually flips a greedy argmax.
            eng_at_top2 = torch.gather(eng_rows, 1, indices)
            top2_err_parts.append(
                ((eng_at_top2[:, 0] - eng_at_top2[:, 1]) - (values[:, 0] - values[:, 1])).abs()
            )

            del ref_rows, eng_rows, eng_lp, ref_lp, p, err

        offset += seq_len
        del eng_logits, eng_log_probs, hidden
        torch.cuda.empty_cache()

    kl = torch.cat(kl_parts)
    top2_err = torch.cat(top2_err_parts)
    engine_argmax = torch.cat(engine_top1)

    # -- the fp64 pass must reproduce the cached top-2 exactly ----------------
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

    # -- logit error ----------------------------------------------------------
    print()
    print(f"logit error, engine fp16 vs fp64 reference, full vocabulary ({abs_count} values)")
    print(f"  absolute   max {abs_max:.6e}  mean {abs_sum / abs_count:.6e}")
    print(f"  relative   max {rel_max:.6e}   over the {rel_count} values with "
          f"|reference logit| >= {REL_ERROR_FLOOR:g}")
    print("  ULP is not reported: it is meaningless across dtypes")
    print()
    print("  absolute error histogram")
    for label, count in histogram(abs_hist):
        print(f"    {label:<16} {bar(count, abs_count):<24} {count:>12}")

    # -- KL -------------------------------------------------------------------
    kl_stats = quantiles(kl)
    print()
    print(f"per-position KL(fp64 reference || engine), nats, exact over the full")
    print(f"vocabulary at all {kl.numel()} positions")
    print(f"  mean {kl_stats['mean']:.6e}  median {kl_stats['median']:.6e}  "
          f"p99 {kl_stats['p99']:.6e}  max {kl_stats['max']:.6e}")

    # -- greedy match with a derived threshold --------------------------------
    ref_top1 = cache["top2_indices"][:, 0].to(torch.int64)
    gap = (cache["top2_logits"][:, 0] - cache["top2_logits"][:, 1]).to(torch.float64)
    err_stats = quantiles(top2_err)
    ordered = top2_err.sort().values
    threshold = float(ordered[int(NEAR_TIE_QUANTILE * (ordered.numel() - 1))])

    agree = engine_argmax == ref_top1
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
    print()
    print(f"  match, gap >= threshold   {clean_match}/{clean_total}   <- headline")
    print(f"  near ties excluded        {tie_total}/{int(near_tie.numel())}")
    print(f"  match, near ties only     {tie_match}/{tie_total}")

    # Rank discipline: a mismatch that is not a top1/top2 swap is a different
    # class of failure and rounding does not explain it.
    mismatch = torch.nonzero(~agree).flatten()
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

    # -- pass criterion, architecture doc 7.1 ---------------------------------
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
    print("F1 pass criterion (docs/02-technical-architecture.md 7.1)")
    for name, ok in checks:
        print(f"  [{'pass' if ok else 'FAIL'}]  {name}")
    passed = all(ok for _, ok in checks)
    print(f"  F1 {'PASS' if passed else 'FAIL'}")

    hf_summary = None if args.skip_hf else hf_sanity_check(engine, args)

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
                "logit_rel_error": {
                    "max": rel_max,
                    "floor": REL_ERROR_FLOOR,
                    "values_counted": rel_count,
                },
                "abs_error_histogram": [
                    {"bin": label, "count": count} for label, count in histogram(abs_hist)
                ],
                "kl_exact_full_vocab": kl_stats,
                "top2_difference_error": err_stats,
                "near_tie_rule": f"p{NEAR_TIE_QUANTILE * 100:g} of the top1-top2 difference error",
                "near_tie_threshold": threshold,
                "match_above_threshold": {"matched": clean_match, "total": clean_total},
                "match_near_ties_only": {"matched": tie_match, "total": tie_total},
                "mismatches_that_are_top1_top2_swaps": {"swaps": swaps, "total": mismatch.numel()},
                "largest_gap_among_mismatches": worst_gap,
                "threshold_sensitivity": sensitivity,
                "f1_checks": [{"check": n, "pass": ok} for n, ok in checks],
                "f1_pass": passed,
                "hf_sanity_check": hf_summary,
            },
        ).write()
        print(f"artifact  {relpath(path)}")

    return 0 if passed else 1


def hf_sanity_check(model: Qwen3, args: argparse.Namespace) -> dict[str, object]:
    """Greedy decode, engine versus HF. Exercises the decode path, not prefill.

    HF is not shape-invariant and is not ground truth. A mismatch here is a lead,
    not a verdict. It is here because a prefill-only comparison would never touch
    the single-token KV-cache path.
    """
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

    # .to("cuda") rather than device_map="cuda": device_map pulls in accelerate,
    # which is not in the dependency set the security doc fixes.
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
