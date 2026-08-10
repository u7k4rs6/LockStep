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


def steps_in_window(rows, start: float, end: float) -> list[tuple[int, int]]:
    """The C++ launches inside a repeat's window, collapsed to one per step.

    A forward pass issues 56 fused_add_rms_norm calls at the same num_tokens
    (28 layers, all but layer 0's first norm, plus the final norm). Collapsing
    runs of equal num_tokens turns the call trace into the step trace without
    assuming how many calls a step contains.
    """
    steps: list[tuple[int, int]] = []
    for epoch, kind, num_tokens, hidden in rows:
        if kind != CXX or not (start <= epoch <= end):
            continue
        if hidden <= NARROW_BLOCK_THRESHOLD:
            # min(hidden, max_block_size) cannot move; not a candidate.
            continue
        if steps and steps[-1][0] == num_tokens:
            continue
        steps.append((num_tokens, hidden))
    return steps


def summarize_repeat(steps) -> dict:
    widths = sorted({block_width(n, h) for n, h in steps})
    crossing = [n for n, _ in steps if n >= NARROW_BLOCK_THRESHOLD]
    return {
        "steps": len(steps),
        "step_tokens": [n for n, _ in steps],
        "max_tokens": max((n for n, _ in steps), default=0),
        "launches_at_or_over_256": len(crossing),
        "crossing_step_tokens": crossing,
        "block_widths_used": widths,
        "used_narrow_block": 256 in widths,
    }


def analyse(artifact_path: Path, trace_paths: list[Path]) -> dict:
    doc = read(artifact_path)
    payload = doc["payload"]
    rows = load_trace(trace_paths)

    cases = []
    for result in payload["results"]:
        repeats = []
        for witness in result["witnesses"]:
            steps = steps_in_window(rows, witness["started_epoch"],
                                    witness["ended_epoch"])
            summary = summarize_repeat(steps)
            summary["repeat"] = witness["repeat"]
            repeats.append(summary)

        digests = result["repeat_digests"]
        width_signatures = [tuple(r["block_widths_used"]) for r in repeats]
        token_signatures = [tuple(r["step_tokens"]) for r in repeats]

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
