"""Read the RMSNorm launch trace back and test the pre-registered prediction.

The trace is written by a patch to the subject venv's `batch_invariant.py`. Each
line is `epoch\tkind\tnum_tokens\thidden`. `num_tokens` is the exact quantity the
pre-fix launcher in `csrc/libtorch_stable/layernorm_kernels.cu` branches on:

    const int max_block_size = (num_tokens < 256) ? 1024 : 256;

Only launches that reach the C++ kernel with `hidden_size > 256` can move, so
this module filters to those and applies the kernel's own predicate rather than
a restatement of it.

Nothing here decides what counts as a pass. The predictions and falsifiers live
in `evidence/prediction-rmsnorm-blocksize.json`, committed before the run.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from report.artifact import read  # noqa: E402

# The kernel's own comparison. 256 selects the NARROW block, so this is >=.
NARROW_BLOCK_THRESHOLD = 256

CXX = "fused_add_rms_norm_cxx"


def block_width(num_tokens: int, hidden: int) -> int:
    """What the pre-fix launcher would pick. dim3 block(min(hidden, max))."""
    max_block_size = 1024 if num_tokens < NARROW_BLOCK_THRESHOLD else 256
    return min(hidden, max_block_size)


def load_trace(paths: list[Path]) -> list[tuple[float, str, int, int]]:
    rows = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            rows.append((float(parts[0]), parts[1], int(parts[2]), int(parts[3])))
    rows.sort(key=lambda r: r[0])
    return rows


def candidates(rows, start: float, end: float) -> list[tuple[int, int]]:
    """C++ launches inside a repeat's window whose block width can actually move."""
    return [
        (num_tokens, hidden)
        for epoch, kind, num_tokens, hidden in rows
        # min(hidden, max_block_size) cannot move when hidden <= 256.
        if kind == CXX and hidden > NARROW_BLOCK_THRESHOLD and start <= epoch <= end
    ]


def calls_per_forward(rows) -> int:
    """How many of these launches one forward pass issues, derived not assumed.

    Qwen3-0.6B should give 56: two norms in each of 28 layers, plus the final
    norm, less layer 0's first norm, which has no residual and goes to Triton.
    Deriving it from the trace instead means this module does not silently
    assume a model shape, and a mismatch shows up as a strange number rather
    than as a plausible wrong answer.

    Consecutive forward passes at the same num_tokens produce a run of length
    56k, so the single most common run length is one pass. The GCD would be
    exact if every run were clean, and it is not: across a full trace one run in
    186 has a length that is not a multiple of 56, which drives a GCD to 1. The
    mode survives that; `runs_not_a_multiple` reports it rather than hiding it.
    """
    counts = Counter(run_lengths(rows))
    return counts.most_common(1)[0][0] if counts else 0


def run_lengths(rows) -> list[int]:
    """Lengths of maximal runs of consecutive launches at the same num_tokens."""
    lengths, run, previous = [], 0, None
    for num_tokens, _ in rows:
        if num_tokens != previous and run:
            lengths.append(run)
            run = 0
        previous, run = num_tokens, run + 1
    if run:
        lengths.append(run)
    return lengths


def steps_in_window(rows, start: float, end: float, per_forward: int
                    ) -> list[tuple[int, int]]:
    """One entry per forward pass, in order.

    Collapsing runs of equal num_tokens would merge a repeat's 24 consecutive
    decode steps into one entry: harmless for the block-width question, wrong
    for everything else. Each run of length L at the same num_tokens is L /
    per_forward passes.

    Divided per run rather than by chunking the window from its start, so a
    single malformed run cannot shift the grouping of everything after it.
    """
    launches = candidates(rows, start, end)
    if per_forward <= 0:
        return []
    steps: list[tuple[int, int]] = []
    run, previous_n, previous_h = 0, None, None
    for num_tokens, hidden in launches:
        if num_tokens != previous_n and run:
            steps.extend([(previous_n, previous_h)]
                         * max(1, round(run / per_forward)))
            run = 0
        previous_n, previous_h = num_tokens, hidden
        run += 1
    if run:
        steps.extend([(previous_n, previous_h)]
                     * max(1, round(run / per_forward)))
    return steps


def narrow_profile(step_tokens) -> tuple[int, ...]:
    """The sizes of the launches that reduced at the narrow width, sorted.

    This is the finest statistic the trace supports. It is *measured*: each
    launch's token count is recorded directly, and every token in a launch
    reduces at that launch's width, so the number of tokens that reduced at 256
    is the sum of these with no attribution required.

    It is not the per-token width vector. Building that would need to know which
    tokens were in which launch, and the trace records only counts. See
    `attribution_limits` in certify/rmsnorm_artifact.py.
    """
    return tuple(sorted(n for n in step_tokens if n >= NARROW_BLOCK_THRESHOLD))


def summarize_repeat(steps) -> dict:
    widths = sorted({block_width(n, h) for n, h in steps})
    step_tokens = [n for n, _ in steps]
    crossing = [n for n in step_tokens if n >= NARROW_BLOCK_THRESHOLD]
    return {
        "steps": len(steps),
        "step_tokens": step_tokens,
        "max_tokens": max(step_tokens, default=0),
        "launches_at_or_over_256": len(crossing),
        "crossing_step_tokens": crossing,
        "narrow_profile": list(narrow_profile(step_tokens)),
        "tokens_reduced_at_narrow_width": sum(crossing),
        "total_tokens_computed": sum(step_tokens),
        "block_widths_used": widths,
        "used_narrow_block": 256 in widths,
    }


def analyse(artifact_path: Path, trace_paths: list[Path]) -> dict:
    doc = read(artifact_path)
    payload = doc["payload"]
    rows = load_trace(trace_paths)
    per_forward = calls_per_forward(
        candidates(rows, float("-inf"), float("inf")))

    cases = []
    for result in payload["results"]:
        repeats = []
        for witness in result["witnesses"]:
            steps = steps_in_window(rows, witness["started_epoch"],
                                    witness["ended_epoch"], per_forward)
            summary = summarize_repeat(steps)
            summary["repeat"] = witness["repeat"]
            repeats.append(summary)

        digests = result["repeat_digests"]
        width_signatures = [tuple(r["block_widths_used"]) for r in repeats]
        token_signatures = [tuple(r["step_tokens"]) for r in repeats]
        narrow_signatures = [tuple(r["narrow_profile"]) for r in repeats]

        # The prediction, evaluated pairwise. Agreement is digest equality;
        # the predictor is whether the pair used the same set of block widths.
        pairs = []
        for left in range(len(repeats)):
            for right in range(left + 1, len(repeats)):
                pairs.append({
                    "pair": [left, right],
                    "digests_agree": digests[left] == digests[right],
                    "same_block_widths": width_signatures[left] == width_signatures[right],
                    "same_step_tokens": token_signatures[left] == token_signatures[right],
                    "same_narrow_profile":
                        narrow_signatures[left] == narrow_signatures[right],
                    "same_narrow_token_count":
                        repeats[left]["tokens_reduced_at_narrow_width"]
                        == repeats[right]["tokens_reduced_at_narrow_width"],
                })

        cases.append({
            "case": result["case"],
            "requests": result["requests"],
            "max_concurrent_running": result["max_concurrent_running"],
            "clean": result["clean"],
            "divergences": len(result["divergences"]),
            "repeats": repeats,
            "pairs": pairs,
            "any_repeat_crossed_256": any(r["used_narrow_block"] for r in repeats),
            "all_repeats_same_widths": len(set(width_signatures)) == 1,
            "all_repeats_same_step_tokens": len(set(token_signatures)) == 1,
        })
    return {
        "trace_lines": len(rows),
        "trace_kinds": dict(Counter(k for _, k, _, _ in rows)),
        "calls_per_forward_pass": per_forward,
        "runs_not_a_multiple_of_a_forward_pass": sum(
            1 for length in run_lengths(candidates(rows, float("-inf"), float("inf")))
            if per_forward and length % per_forward
        ),
        "cases": cases,
    }


def verdicts(cases: list[dict]) -> dict:
    """P2 and its falsifiers, counted over every pair in every case."""
    tally = Counter()
    counterexamples = {"F1": [], "F2": [], "F3": []}
    for case in cases:
        for pair in case["pairs"]:
            agree, same_width = pair["digests_agree"], pair["same_block_widths"]
            tally["pairs"] += 1
            tally[f"agree={agree},same_widths={same_width}"] += 1
            where = {"case": case["case"], "pair": pair["pair"]}
            if not agree and pair["same_step_tokens"]:
                counterexamples["F1"].append(where)
            if not agree and same_width and not case["any_repeat_crossed_256"]:
                counterexamples["F2"].append(where)
            if agree and not same_width:
                counterexamples["F3"].append(where)
    return {"tally": dict(tally), "counterexamples": counterexamples}


def contingency(runs: list[dict]) -> dict:
    """Agreement against block width, over every pair in every lifetime.

    Three populations, because the interesting one is small. Pairs where neither
    repeat crossed 256 should always agree; pairs where exactly one crossed
    should never agree; pairs where both crossed depend on whether the same
    tokens landed in the crossing step, which the width-set signature cannot see.
    """
    table = Counter()
    neither, one, both = Counter(), Counter(), Counter()
    fired = Counter()
    # The biconditional, over every pair rather than only the contested cell.
    biconditional = Counter()
    exceptions = []
    for run in runs:
        for case in run["analysis"]["cases"]:
            by_repeat = {r["repeat"]: r for r in case["repeats"]}
            for pair in case["pairs"]:
                left, right = (by_repeat[i] for i in pair["pair"])
                agree = pair["digests_agree"]
                table[f"agree={agree},same_widths={pair['same_block_widths']}"] += 1
                crossed = (left["used_narrow_block"], right["used_narrow_block"])
                if not any(crossed):
                    neither[agree] += 1
                    if not agree:
                        fired["F2"] += 1
                elif all(crossed):
                    both[f"agree={agree},same_step_tokens={pair['same_step_tokens']}"] += 1
                else:
                    one[agree] += 1
                if not agree and pair["same_step_tokens"]:
                    fired["F1"] += 1
                if agree and not pair["same_block_widths"]:
                    fired["F3"] += 1
                same_narrow = pair["same_narrow_profile"]
                biconditional[f"agree={agree},same_narrow_profile={same_narrow}"] += 1
                if agree != same_narrow:
                    exceptions.append({
                        "case": case["case"], "pair": pair["pair"],
                        "left": left["narrow_profile"],
                        "right": right["narrow_profile"],
                        "digests_agree": agree,
                    })
    return {
        "pairs_total": sum(table.values()),
        "agreement_by_width_signature": dict(table),
        "neither_repeat_crossed_256": {str(k): v for k, v in neither.items()},
        "exactly_one_repeat_crossed_256": {str(k): v for k, v in one.items()},
        "both_repeats_crossed_256": dict(both),
        "falsifiers_fired": dict(fired),
        "biconditional_agree_iff_same_narrow_profile": {
            "table": dict(biconditional),
            "exceptions": exceptions,
            "holds": not exceptions,
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("traces", type=Path, nargs="+")
    args = parser.parse_args()

    result = analyse(args.artifact, args.traces)
    result["verdicts"] = verdicts(result["cases"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
