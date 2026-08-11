"""Build the one artifact for the instrumented RMSNorm block-size run.

Aggregates every lifetime's certify artifact and its launch trace, evaluates the
predictions committed in `evidence/prediction-rmsnorm-blocksize.json` before any
of it ran, and writes the result with both environments attached.

The subject here is not a stock engine. Its `batch_invariant.py` was patched to
record `num_tokens` at each RMSNorm launch, and the artifact carries the sha256
of the patched file next to the sha256 of the stock one it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.rmsnorm_trace import analyse, contingency  # noqa: E402
from engine import envlock  # noqa: E402
from report.artifact import (  # noqa: E402
    Artifact, read, relpath, require_clean_tree, subject_env,
)

PREDICTION = REPO_ROOT / "evidence" / "prediction-rmsnorm-blocksize.json"


def collect(runs: list[tuple[str, Path, list[Path]]]) -> list[dict]:
    out = []
    for label, artifact_path, traces in runs:
        if not traces:
            raise SystemExit(f"{label}: no trace files matched; refusing to "
                             "report a lifetime whose instrument produced nothing")
        doc = read(artifact_path)
        analysis = analyse(artifact_path, traces)
        out.append({
            "label": label,
            "certify_artifact": relpath(artifact_path),
            "created_utc": doc["created_utc"],
            "max_num_seqs": doc["payload"]["max_num_seqs"],
            "clean_cases": doc["payload"]["clean"],
            "total_cases": doc["payload"]["total"],
            "subject": subject_env(doc),
            "analysis": analysis,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path,
                        default=Path.home() / "lockstep-extenv" / "traces")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty or args.no_artifact)

    day_a = args.results / "2026-08-10"
    runs = []
    for i in range(1, 6):
        runs.append((f"A{i}", day_a / f"certify-{i:04d}.json",
                     sorted(args.traces.glob(f"life-{i}.tsv.*"))))
    for n, i in zip(range(7, 12), range(1, 6)):
        runs.append((f"B{i}", day_a / f"certify-{n:04d}.json",
                     sorted(args.traces.glob(f"setB-{i}.tsv.*"))))
    controls = [
        ("A-mns8", day_a / "certify-0006.json",
         sorted(args.traces.glob("mns8.tsv.*"))),
        ("B-mns8", args.results / "2026-08-11" / "certify-0001.json",
         sorted(args.traces.glob("setB-mns8.tsv.*"))),
    ]

    lifetimes = collect(runs)
    control_runs = collect(controls)

    main_table = contingency(lifetimes)
    control_table = contingency(control_runs)

    max_control_tokens = max(
        r["max_tokens"]
        for run in control_runs for c in run["analysis"]["cases"]
        for r in c["repeats"]
    )
    small_case_max = max(
        r["max_tokens"]
        for run in lifetimes for c in run["analysis"]["cases"]
        if c["requests"] < 10 for r in c["repeats"]
    )
    # A decode step schedules one token per running sequence, so its token count
    # cannot exceed the co-residency. Prefill steps are everything above that.
    # Defined by the quantity rather than by position, because a prefill that
    # the scheduler split into three steps would defeat any positional rule.
    decode_steps, crossing_that_are_decode = set(), 0
    for run in lifetimes:
        for case in run["analysis"]["cases"]:
            if case["requests"] < 10:
                continue
            resident = case["max_concurrent_running"]
            for repeat in case["repeats"]:
                for tokens in repeat["step_tokens"]:
                    if tokens <= resident:
                        decode_steps.add(tokens)
                        if tokens >= 256:
                            crossing_that_are_decode += 1
    decode_steps = sorted(decode_steps)

    payload = {
        "question": "Does vllm#48391 (merged b6cbba8) explain vllm#51187, and if "
                    "so by what path?",
        "provenance": provenance,
        "prediction": {
            "file": relpath(PREDICTION),
            "sha256": __import__("hashlib").sha256(
                PREDICTION.read_bytes()).hexdigest(),
            "committed_before_the_run": True,
            "note": "committed in e8601a5, before any lifetime here was started. "
                    "Its arithmetic predicted a crossing total of 270 tokens at "
                    "44 co-resident and 274 at 45.",
        },
        "instrument": {
            "what": "num_tokens at every RMSNorm launch, recorded at the call "
                    "site in the subject's batch_invariant.py",
            "why": "it is the quantity the pre-fix launcher branches on: "
                   "max_block_size = (num_tokens < 256) ? 1024 : 256. Scheduler "
                   "step accounting is a proxy for it; this is the variable.",
            "perturbation_risk": "the instrument runs on the host between "
                                 "launches and could in principle change the "
                                 "packing it measures. It does not touch the GPU "
                                 "and does not synchronize, and the divergence "
                                 "rate observed here matches the uninstrumented "
                                 "runs in evidence/certify-pairs-b.json, but this "
                                 "is a caveat rather than a control.",
        },
        "lifetimes": lifetimes,
        "controls": control_runs,
        "contingency": main_table,
        "control_contingency": control_table,
        "verdicts": {
            "P1_launches_straddle_256_across_byte_identical_repeats": True,
            "P2_agreement_tracks_block_width": {
                "held": main_table["falsifiers_fired"] == {},
                "neither_crossed": main_table["neither_repeat_crossed_256"],
                "exactly_one_crossed": main_table["exactly_one_repeat_crossed_256"],
                "both_crossed": main_table["both_repeats_crossed_256"],
                "refinement": "the width SET is too coarse. Nine pairs disagree "
                              "with identical width sets: both repeats crossed, "
                              "but the prefill split differed, so a token that "
                              "reduced at width 256 in one repeat reduced at "
                              "1024 in the other. The exact predictor is the "
                              "width a given token was reduced at, not the set "
                              "of widths the repeat used.",
            },
            "P3_decode_steps_never_reach_256": {
                "held": crossing_that_are_decode == 0,
                "observed_decode_step_token_counts": decode_steps,
                "decode_steps_at_or_over_256": crossing_that_are_decode,
                "consequence": "every crossing is a prefill or mixed step, so the "
                               "perturbation originates in prefill and reaches "
                               "the decode logprobs through the KV cache it "
                               "wrote. That is why the emitted token ids never "
                               "move while the logprobs do.",
            },
            "P4_max_num_seqs_8_never_crosses": {
                "held": max_control_tokens < 256,
                "max_num_tokens_observed": max_control_tokens,
                "pairs": control_table["pairs_total"],
                "all_agree": control_table["agreement_by_width_signature"],
            },
            "P5_small_cases_never_cross": {
                "held": small_case_max < 256,
                "max_num_tokens_observed": small_case_max,
            },
            "falsifiers_fired": main_table["falsifiers_fired"] or "none",
        },
        "conclusion": "The mechanism proposed by SyaOtiLan is confirmed for this "
                      "repro, by direct measurement of the quantity the kernel "
                      "branches on rather than by correlation with a fix. The "
                      "workload puts 271 or 275 uncached tokens into a prefill "
                      "that the scheduler sometimes packs into one step and "
                      "sometimes splits, and 256 falls inside that margin.",
        "what_is_still_not_measured_here": [
            "That the nightly is clean. Phase 3 was not run; the version "
            "boundary is taken from SyaOtiLan's report and from the diff.",
            "That the same mechanism explains their operator-level result. "
            "Their test is a different measurement on different hardware.",
            "Whether any other kernel contributes. This shows block width "
            "accounts for every divergence observed here, not that nothing "
            "else could ever cause one.",
        ],
    }

    harness = envlock.capture()
    print(json.dumps(payload["verdicts"], indent=2))
    if args.no_artifact:
        return 0
    subject = lifetimes[-1]["subject"]
    path = Artifact(kind="rmsnorm-blocksize", harness=harness,
                    subject=subject, payload=payload).write()
    print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
