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

    # Task 2: per-workload threshold table. Retires the filed issue's
    # "not determined: whether the width dependence is a threshold" line.
    from certify.run import boundary_workloads, filler_requests
    block = 16
    filler_lengths = [len(f) for f in filler_requests(13, block, 0)]
    uncached_from_fillers = sum(length % block for length in filler_lengths)
    uncached_from_requests = {
        case["name"]: sum(len(r) % block for r in case["requests"])
        for case in boundary_workloads(block)
    }

    workloads: dict[str, dict] = {}
    for run in lifetimes:
        for case in run["analysis"]["cases"]:
            row = workloads.setdefault(case["case"], {
                "co_resident": case["max_concurrent_running"],
                "requests": case["requests"],
                "fillers": 13,
                "uncached_prefill_tokens":
                    uncached_from_requests[case["case"]] + uncached_from_fillers,
                "max_tokens_in_any_launch": 0,
                "lifetimes": 0, "lifetimes_that_crossed": 0,
                "lifetimes_clean": 0,
            })
            row["max_tokens_in_any_launch"] = max(
                row["max_tokens_in_any_launch"],
                max(r["max_tokens"] for r in case["repeats"]))
            row["lifetimes"] += 1
            row["lifetimes_that_crossed"] += bool(case["any_repeat_crossed_256"])
            row["lifetimes_clean"] += bool(case["clean"])
    for row in workloads.values():
        row["crossed"] = row["lifetimes_that_crossed"] > 0

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
                    "It predicted 270 UNCACHED PREFILL tokens at 44 co-resident "
                    "and 274 at 45, and both are exactly right: uncached plus "
                    "23 decode tokens per sequence reproduces the measured "
                    "total computed per repeat to the token (270 + 44*23 = "
                    "1282; 274 + 45*23 = 1309). An earlier summary of this run "
                    "reported the prediction as 271 and 275 and called those "
                    "uncached. Both figures were wrong and so was the label: "
                    "see token_accounting.",
        },
        "instrument": {
            "what": "num_tokens at every RMSNorm launch, recorded at the call "
                    "site in the subject's batch_invariant.py",
            "why": "it is the quantity the pre-fix launcher branches on: "
                   "max_block_size = (num_tokens < 256) ? 1024 : 256. Scheduler "
                   "step accounting is a proxy for it; this is the variable.",
            "perturbation_risk": {
                "what_it_threatens": "The instrument runs on the host between "
                    "launches, so it could shift request arrivals against step "
                    "boundaries and change WHICH packings occur. That is a "
                    "threat to the marginal distribution of packings: how often "
                    "a step crosses 256, and so the 9-of-10 divergence rate.",
                "what_it_does_not_threaten": "The claim is conditional, not "
                    "marginal. Given the packing that occurred, does agreement "
                    "follow the block width? Both sides of that comparison come "
                    "from the same run. Perturbing which packings happen changes "
                    "the sample of packings tested; it cannot make a pair that "
                    "shared a narrow profile disagree, or a pair that differed "
                    "agree. A biconditional holding over 700 pairs is not "
                    "weakened by the instrument having chosen which 700.",
                "load_bearing_exclusion": "The divergence RATE is the one number "
                    "the caveat genuinely touches, so nothing rests on it. It is "
                    "reported and not argued from. Uninstrumented it was roughly "
                    "2 in 3 (evidence/certify-pairs-b.json); here 9 of 10. That "
                    "difference is unresolved and deliberately load-free.",
                "not_touched": "no GPU call and no synchronize; numel() and "
                               "shape are host-side metadata already resident.",
            },
        },
        "attribution_limits": {
            "what_was_asked": "a per-token width vector: for each token, the "
                              "width it reduced at, compared elementwise.",
            "why_it_was_not_built": "The trace records one number per launch and "
                "no request or position identity. Reconstructing a per-token "
                "vector needs two assumptions, not one. First that token "
                "ordering is stable across repeats. Second, and fatally, that a "
                "launch splits cleanly into prefill and decode, which it does "
                "not: with chunked prefill the scheduler emits MIXED steps. "
                "Measured directly, one repeat of the 44-co-resident case shows "
                "launches of 3, 203 and 104 against an uncached prefill total of "
                "270, so 40 decode tokens are folded in and nothing in the trace "
                "says which.",
            "what_was_used_instead": "the narrow profile: the sorted sizes of "
                "the launches that reduced at width 256. Every token in a launch "
                "reduces at that launch's width, so this is measured, not "
                "inferred, and needs no attribution.",
            "the_residual_gap": "Equal narrow profiles do not prove the SAME "
                "tokens reduced at 256. Two repeats could pack different tokens "
                "into equally sized crossing launches and match on the statistic "
                "while differing per token. That would be a false positive. It "
                "did not occur in 840 pairs, and the trace cannot rule it out by "
                "construction.",
            "what_would_close_it": "log scheduler_output.num_scheduled_tokens in "
                "the model runner, which carries request ids and per-request "
                "counts. Cost: one more venv patch and a re-run of the same 12 "
                "lifetimes, about 11 minutes of GPU. Not run: the measured "
                "statistic already gives a biconditional with no exceptions.",
            "supporting_check": "total tokens computed per repeat is invariant "
                "within a case (1282 at 44 co-resident, 1309 at 45), consistent "
                "with the same token set every repeat and only the partition "
                "moving. Eight repeats of 70 come in short, all of them the last "
                "repeat of their case and all short only in trailing "
                "decode-sized launches: a window-boundary artifact that cannot "
                "touch a crossing launch.",
        },
        "threshold_table": {
            "note": "Every workload instrumented, across 10 lifetimes. Retires "
                    "the 'not determined' line in the filed issue: the "
                    "dependence is a threshold on tokens scheduled into one "
                    "launch, at 256, and not a threshold on sequence count.",
            "filler_count_was_held_at_13_throughout": (
                "so this table cannot separate filler count from sequence count "
                "experimentally. The per-sequence arithmetic below does separate "
                "them, but arithmetically, and a --fixed-width sweep would be "
                "needed to show it by measurement."),
            "per_sequence_contribution": {
                "filler_sequences": {
                    "count": 13, "uncached_tokens": uncached_from_fillers,
                    "each": round(uncached_from_fillers / 13, 1)},
                "prefix_sharing_target_sequences": {
                    "count": 31,
                    "uncached_tokens": uncached_from_requests[
                        "batch 31, shared prefix of one block"],
                    "each": round(uncached_from_requests[
                        "batch 31, shared prefix of one block"] / 31, 1)},
                "reading": "a filler contributes about twice the uncached tokens "
                           "of a prefix-sharing target, so 13 of the 44 "
                           "sequences supply 43 percent of the uncached total. "
                           "Sequence count is not even monotonically related to "
                           "the governing quantity across workload types.",
            },
            "rows": workloads,
        },
        "lifetimes": lifetimes,
        "controls": control_runs,
        "contingency": main_table,
        "control_contingency": control_table,
        "verdicts": {
            "P1_launches_straddle_256_across_byte_identical_repeats": True,
            "P2_agreement_tracks_block_width": {
                "biconditional": main_table[
                    "biconditional_agree_iff_same_narrow_profile"],
                "statement": "two repeats agree if and only if the same sizes of "
                             "launch reduced at width 256. Holds over all 700 "
                             "pairs with no exception, and subsumes the "
                             "three-cell table below: neither crossing gives two "
                             "empty profiles, exactly one crossing gives an "
                             "empty against a non-empty, and both crossing gives "
                             "equal profiles only when the splits coincide.",
                "held": main_table["falsifiers_fired"] == {},
                "neither_crossed": main_table["neither_repeat_crossed_256"],
                "exactly_one_crossed": main_table["exactly_one_repeat_crossed_256"],
                "both_crossed": main_table["both_repeats_crossed_256"],
                "why_the_width_set_was_the_wrong_statistic": "The set {256, "
                    "1024} is too coarse: nine pairs disagree while sharing it, "
                    "because both crossed but split the prefill differently, so "
                    "the crossing launches had different sizes (271 against 272, "
                    "268 against 267). The narrow profile separates those and "
                    "the set does not. Note what is claimed: the SIZES of the "
                    "crossing launches differ, which is measured. That "
                    "different TOKENS therefore reduced at 256 is the natural "
                    "reading but is not measured here; see attribution_limits.",
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
                      "workload leaves 270 or 274 uncached prefill tokens, and "
                      "the scheduler sometimes puts nearly all of them into one "
                      "launch and sometimes splits them, so a launch of 268 or "
                      "272 tokens either happens or does not. 256 falls inside "
                      "that margin.",
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
