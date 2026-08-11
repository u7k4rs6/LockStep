"""Build the one artifact for the instrumented RMSNorm block-size run.

One claim, at the finest granularity the instruments support: two repeats of a
byte-identical workload return identical logprobs if and only if every token was
reduced at the same block width in both. Everything coarser that was true along
the way is recorded as a consequence of that, not as a parallel result.

Two instruments, both patches to the subject venv and neither part of this
repository:

  * `batch_invariant.py` records `num_tokens` at each RMSNorm launch, which is
    the quantity the pre-fix launcher branches on.
  * the v2 model runner records `scheduler_output.num_scheduled_tokens` per
    step, plus a hash of each request's prompt the first time it is scheduled,
    which is what makes a per-TOKEN claim possible rather than a per-launch one.

They are independent and they agree, which is checked here rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.rmsnorm_trace import analyse  # noqa: E402
from certify.sched_trace import load as load_sched  # noqa: E402
from certify.sched_trace import per_token_widths, step_totals  # noqa: E402
from engine import envlock  # noqa: E402
from report.artifact import (  # noqa: E402
    Artifact, read, relpath, require_clean_tree, subject_env,
)

PREDICTION = REPO_ROOT / "evidence" / "prediction-rmsnorm-blocksize.json"
ARTIFACT_LINE = re.compile(r"^artifact\s+(\S+)", re.M)


def discover(traces: Path) -> dict[str, Path]:
    """Which certify artifact each labelled lifetime produced, from its stdout."""
    found = {}
    for path in sorted(traces.glob("r4-*.stdout")):
        match = ARTIFACT_LINE.search(path.read_text(errors="replace"))
        if match:
            found[path.name[3:-7]] = REPO_ROOT / match.group(1)
    return found


def arm_of(label: str) -> str:
    if label.startswith("mns8"):
        return "max_num_seqs_8"
    if label.startswith("nofi"):
        return "flashinfer_sampler_disabled"
    return "main"


def signature_digest(signature: dict) -> str:
    """A short stable hash of a repeat's per-token width assignment.

    The assignment is one entry per token per request and is far too large to
    publish for 560 repeat-windows. Equality of these digests is equality of the
    assignment, which is the only property the claim uses.
    """
    payload = json.dumps(
        {prompt: [list(map(list, vector)) for vector in vectors]
         for prompt, vectors in sorted(signature.items())},
        separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def collect(label: str, artifact_path: Path, traces: Path) -> dict:
    rms_paths = sorted(traces.glob(f"r4-{label}.tsv.[0-9]*"))
    sched_paths = sorted(traces.glob(f"r4-{label}.tsv.sched.*"))
    if not rms_paths or not sched_paths:
        raise SystemExit(f"{label}: missing a trace; refusing to report a "
                         "lifetime whose instruments produced nothing")
    doc = read(artifact_path)
    prompts, steps = load_sched(sched_paths)
    launch_view = analyse(artifact_path, rms_paths)

    cases = []
    for case, launch_case in zip(doc["payload"]["results"], launch_view["cases"]):
        repeats = []
        for witness, launch_repeat in zip(case["witnesses"], launch_case["repeats"]):
            start, end = witness["started_epoch"], witness["ended_epoch"]
            widths = per_token_widths(steps, prompts, start, end)
            totals = step_totals(steps, start, end)
            repeats.append({
                "repeat": witness["repeat"],
                "scheduler_steps": len(totals),
                "step_tokens": totals,
                "max_step_tokens": max(totals, default=0),
                "tokens_at_narrow_width": widths["tokens_at_narrow_width"],
                "narrow_profile": sorted(t for t in totals if t >= 256),
                "requests": widths["requests"],
                "requests_without_a_recorded_prompt":
                    widths["requests_without_a_recorded_prompt"],
                "ambiguous_prompts": widths["ambiguous_prompts"],
                "per_token_width_digest": signature_digest(widths["signature"]),
                # The independent RMSNorm instrument's view of the same window.
                # Stored only when it disagrees: it is identical in 347 of 350
                # main-arm windows and duplicating it wholesale tripled the
                # artifact for no added evidence.
                "launch_view_agrees":
                    launch_repeat["step_tokens"] == totals,
                "launch_view_step_tokens":
                    None if launch_repeat["step_tokens"] == totals
                    else launch_repeat["step_tokens"],
            })

        digests = case["repeat_digests"]
        pairs = []
        for left in range(len(repeats)):
            for right in range(left + 1, len(repeats)):
                pairs.append({
                    "pair": [left, right],
                    "logprobs_agree": digests[left] == digests[right],
                    "same_per_token_widths":
                        repeats[left]["per_token_width_digest"]
                        == repeats[right]["per_token_width_digest"],
                    "same_narrow_profile":
                        repeats[left]["narrow_profile"]
                        == repeats[right]["narrow_profile"],
                })
        cases.append({
            "case": case["case"],
            "requests": case["requests"],
            "co_resident": case["max_concurrent_running"],
            "clean": case["clean"],
            "any_repeat_crossed_256":
                any(r["max_step_tokens"] >= 256 for r in repeats),
            "repeats": repeats,
            "pairs": pairs,
        })

    return {
        "label": label,
        "arm": arm_of(label),
        "certify_artifact": relpath(artifact_path),
        "created_utc": doc["created_utc"],
        "max_num_seqs": doc["payload"]["max_num_seqs"],
        "flashinfer_sampler_disabled":
            doc["payload"]["flashinfer_sampler_disabled"],
        "clean_cases": doc["payload"]["clean"],
        "total_cases": doc["payload"]["total"],
        "subject": subject_env(doc),
        "cases": cases,
        "launch_view": {
            "calls_per_forward_pass": launch_view["calls_per_forward_pass"],
            "runs_not_a_multiple_of_a_forward_pass":
                launch_view["runs_not_a_multiple_of_a_forward_pass"],
        },
    }


def biconditional(lifetimes, predicate: str) -> dict:
    table, exceptions = Counter(), []
    for run in lifetimes:
        for case in run["cases"]:
            for pair in case["pairs"]:
                agree, same = pair["logprobs_agree"], pair[predicate]
                table[f"logprobs_agree={agree},{predicate}={same}"] += 1
                if agree != same:
                    exceptions.append({"lifetime": run["label"],
                                       "case": case["case"], "pair": pair["pair"],
                                       "logprobs_agree": agree})
    return {"pairs": sum(table.values()), "table": dict(table),
            "exceptions": exceptions, "holds": not exceptions}


def cross_check(lifetimes) -> dict:
    """The two instruments are independent; do they describe the same steps?"""
    agree = disagree = 0
    for run in lifetimes:
        for case in run["cases"]:
            for repeat in case["repeats"]:
                if repeat["launch_view_agrees"]:
                    agree += 1
                else:
                    disagree += 1
    return {
        "windows_where_both_instruments_report_the_same_steps": agree,
        "windows_where_they_differ": disagree,
        "note": "the scheduler instrument records what was scheduled; the "
                "RMSNorm instrument records what the kernel was launched with. "
                "Separate patches to separate files reading separate state, so "
                "agreement is evidence neither is inventing steps. Where they "
                "differ it is the RMSNorm view reconstructing forward passes "
                "from launch counts, which is the weaker of the two and the "
                "reason the second instrument exists.",
        "every_disagreement_observed": "three windows, all of them the last "
                "repeat of the last case, where the RMSNorm view is SHORTER. "
                "That instrument flushes every 512 calls, about nine forward "
                "passes, so it loses a tail the scheduler instrument keeps. The "
                "prefixes are identical and the lost steps are decode-sized, so "
                "no crossing launch is affected.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path,
                        default=Path.home() / "lockstep-extenv" / "traces")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty or args.no_artifact)

    discovered = discover(args.traces)
    if not discovered:
        raise SystemExit(f"no r4-*.stdout under {args.traces}")
    lifetimes = [collect(label, path, args.traces)
                 for label, path in sorted(discovered.items())]

    main_arm = [r for r in lifetimes if r["arm"] == "main"]
    claim = biconditional(main_arm, "same_per_token_widths")
    consequence_profile = biconditional(main_arm, "same_narrow_profile")
    all_arms = biconditional(lifetimes, "same_per_token_widths")

    arms: dict[str, dict] = {}
    for run in lifetimes:
        entry = arms.setdefault(run["arm"], {
            "lifetimes": 0, "lifetimes_with_a_divergence": 0,
            "cases": 0, "cases_that_crossed_256": 0, "cases_that_diverged": 0,
            "max_step_tokens_observed": 0, "crossing_launch_sizes": Counter()})
        entry["lifetimes"] += 1
        entry["lifetimes_with_a_divergence"] += run["clean_cases"] < run["total_cases"]
        for case in run["cases"]:
            entry["cases"] += 1
            entry["cases_that_crossed_256"] += case["any_repeat_crossed_256"]
            entry["cases_that_diverged"] += not case["clean"]
            for repeat in case["repeats"]:
                entry["max_step_tokens_observed"] = max(
                    entry["max_step_tokens_observed"], repeat["max_step_tokens"])
                for tokens in repeat["narrow_profile"]:
                    entry["crossing_launch_sizes"][tokens] += 1
    for entry in arms.values():
        entry["crossing_launch_sizes"] = dict(
            sorted(entry["crossing_launch_sizes"].items()))

    from certify.run import boundary_workloads, filler_requests
    block = 16
    uncached_fillers = sum(len(f) % block for f in filler_requests(13, block, 0))
    uncached_requests = {c["name"]: sum(len(r) % block for r in c["requests"])
                         for c in boundary_workloads(block)}
    workloads: dict[str, dict] = {}
    for run in main_arm:
        for case in run["cases"]:
            row = workloads.setdefault(case["case"], {
                "co_resident": case["co_resident"], "requests": case["requests"],
                "fillers": 13,
                "uncached_prefill_tokens":
                    uncached_requests[case["case"]] + uncached_fillers,
                "max_tokens_in_any_launch": 0,
                "lifetimes": 0, "lifetimes_that_crossed": 0, "lifetimes_clean": 0})
            row["max_tokens_in_any_launch"] = max(
                row["max_tokens_in_any_launch"],
                max(r["max_step_tokens"] for r in case["repeats"]))
            row["lifetimes"] += 1
            row["lifetimes_that_crossed"] += case["any_repeat_crossed_256"]
            row["lifetimes_clean"] += case["clean"]

    decode_tokens, decode_at_or_over_256 = set(), 0
    for run in main_arm:
        for case in run["cases"]:
            for repeat in case["repeats"]:
                for tokens in repeat["step_tokens"]:
                    if tokens <= case["co_resident"]:
                        decode_tokens.add(tokens)
                        decode_at_or_over_256 += tokens >= 256

    payload = {
        "question": "Does vllm#48391 (merged b6cbba8) explain vllm#51187, and if "
                    "so by what path?",
        "provenance": provenance,

        "claim": {
            "statement": "Two repeats of a byte-identical workload return "
                         "identical logprobs if and only if every token was "
                         "reduced at the same RMSNorm block width in both.",
            "granularity": "per token. The step that computed each token is "
                           "known from the scheduler instrument, the step's "
                           "total token count is known, and the kernel's own "
                           "predicate turns that count into a width. Nothing "
                           "is inferred from launch counts alone.",
            "result": claim,
            "across_every_arm": all_arms,
        },

        "consequences": {
            "note": "These were separate results while the instruments were "
                    "coarser. They are recorded as consequences of the claim, "
                    "not beside it: each is the claim read at a coarser "
                    "granularity, exactly as true and less precise.",
            "narrow_profile": {
                "statement": "two repeats agree iff the same launch SIZES "
                             "reduced at width 256",
                "result": consequence_profile,
                "why_it_is_coarser": "it compares how many tokens reduced "
                                     "narrowly and in what sized groups, not "
                                     "which tokens did.",
            },
            "block_width_set": "coarser still: the SET {256, 1024} cannot "
                               "separate two repeats that both crossed with "
                               "different splits. Nine pairs in an earlier "
                               "round disagreed while sharing it, which is what "
                               "motivated building the second instrument.",
            "case_level": "a case diverged if and only if some repeat in it put "
                          "256 or more tokens into a launch. See `arms`, where "
                          "cases_that_crossed_256 equals cases_that_diverged in "
                          "every arm.",
        },

        "token_accounting": {
            "why": "an earlier summary of this work said the 44 and 45 "
                   "co-resident workloads carry 271 and 275 uncached prefill "
                   "tokens. Both figures were wrong and so was the label.",
            "uncached_prefill_tokens": {"44_co_resident": 270,
                                        "45_co_resident": 274},
            "how_that_is_confirmed": "uncached plus 23 decode tokens per "
                "sequence reproduces the measured total computed per repeat to "
                "the token: 270 + 44*23 = 1282, and 274 + 45*23 = 1309. Both "
                "totals are invariant across repeats.",
            "what_271_and_275_actually_are": "the sum of the launches during "
                "the prefill phase of a CROSSING repeat only. Such a repeat "
                "prefills in two launches, for example 3 then 268, and the "
                "second launch also carries one decode token for the request "
                "prefilled in the first. So 271 is 270 uncached plus 1 decode "
                "token: a property of one packing, not of the workload.",
            "the_earlier_explanation_was_also_wrong": "the extra token was "
                "attributed to vLLM's must-compute-at-least-one-token rule. It "
                "is not that. No request in this workload is fully cached, and "
                "the totals balance exactly without such a rule.",
            "where_it_propagated": "the pre-registration is clean and says 270 "
                "and 274 under the correct label. The error appeared only in "
                "prose written after the run: this artifact's conclusion and an "
                "earlier draft of the upstream comment. Both corrected.",
        },

        "governing_quantity": {
            "it_is": "tokens scheduled into one launch, whatever their origin.",
            "it_is_not": "uncached prefill tokens. The full-prompt-cache-hit "
                "workload has the lowest uncached total of any case at 117 and "
                "the highest maximum launch of the small cases at 147, because "
                "both its requests are fully cached and start decoding "
                "immediately while the fillers prefill, so decode tokens fold "
                "into the same launches. Max launch size is therefore not a "
                "function of uncached prefill, and an earlier framing that said "
                "so is refuted by a row of this project's own table.",
            "it_is_also_not": "co-resident sequence count. See threshold_table.",
            "why_the_wording_matters": "tokens scheduled into one launch is "
                "exactly what the kernel branches on. Any paraphrase in terms "
                "of prefill, cache state or sequence count is a proxy that "
                "happens to correlate in some workload.",
        },

        "which_models_are_affected": {
            "condition": "hidden_size strictly greater than 256, for the "
                         "fused_add_rms_norm path.",
            "why_strictly": "the launcher is dim3 block(std::min(hidden_size, "
                "max_block_size)) and max_block_size flips between 1024 and "
                "256. At hidden_size exactly 256 both branches give min(256, "
                "1024) = min(256, 256) = 256, so the reduction width never "
                "moves and that model cannot be affected. An earlier draft of "
                "the upstream comment said 'at or above 256' and was wrong at "
                "the boundary by one model width.",
            "verified_at_source": "read at the v0.26.0 and v0.27.0 tags. It is "
                "a plain std::min: no rounding, no warp alignment, no clamp to "
                "a power of two, and LAUNCH_FUSED_ADD_RMS_NORM passes the "
                "resulting dim3 straight to the kernel without modifying it. "
                "The strict inequality depends on that, which is why it was "
                "re-read rather than recalled.",
            "the_non_fused_rms_norm_has_a_different_boundary": "its launcher is "
                "std::min(hidden_size / calculated_vec_size, max_block_size) "
                "with calculated_vec_size = std::gcd(16 / sizeof(scalar_t), "
                "hidden_size), which is 8 at fp16 for any hidden size divisible "
                "by 8. So that op needs hidden_size / 8 > 256, i.e. hidden_size "
                "> 2048. This predicts SyaOtiLan's operator table exactly: at "
                "hidden 512 and 4096 with 16 seeds, fused_add_rms_norm fails "
                "32/32 because both sizes exceed 256, while rms_norm fails "
                "16/32 because 512/8 = 64 stays under the clamp and 4096/8 = "
                "512 does not.",
            "worked_example_qk_norm": "Qwen3's q_norm and k_norm are "
                "residual-free, so they take the non-fused launcher. At "
                "head_dim 128 and fp16 the vec_size is gcd(8, 128) = 8 and the "
                "width is min(16, ...) = 16 either way, so they cannot be "
                "affected. An earlier writeup reached the same conclusion using "
                "min(128, ...), which is the fused-add expression applied to a "
                "call that does not use it: right answer, wrong branch, and "
                "invisible until the two expressions were known to differ.",
            "why_hidden_1024_settles_it_twice": "in these runs the model's "
                "hidden size is 1024, and 1024 / 8 = 128, so min(128, 1024) and "
                "min(128, 256) are both 128. The non-fused op could not flip "
                "here even if the engine did reach it. That is independent of "
                "the dispatch argument below and does not depend on which path "
                "the engine takes.",
            "which_of_the_two_the_engine_reaches": "under VLLM_BATCH_INVARIANT=1 "
                "only fused_add_rms_norm, because rms_norm_batch_invariant "
                "returns into it whenever a residual is present and sends only "
                "the residual-free path to Triton.",
        },

        "threshold_table": {
            "note": "Every workload instrumented, across the main arm. Retires "
                    "the 'not determined: whether the width dependence is a "
                    "threshold' line in the filed issue. It is a threshold, at "
                    "256 tokens in one launch.",
            "filler_count_was_held_at_13_throughout":
                "so this table cannot separate filler count from sequence count "
                "experimentally. The arithmetic below separates them "
                "arithmetically; a --fixed-width sweep would do it by "
                "measurement and was not run.",
            "per_sequence_contribution": {
                "filler_sequences": {"count": 13,
                                     "uncached_tokens": uncached_fillers,
                                     "each": round(uncached_fillers / 13, 1)},
                "prefix_sharing_targets": {
                    "count": 31,
                    "uncached_tokens":
                        uncached_requests["batch 31, shared prefix of one block"],
                    "each": round(
                        uncached_requests["batch 31, shared prefix of one block"]
                        / 31, 1)},
                "reading": "a filler contributes about twice the uncached "
                           "tokens of a prefix-sharing target, so 13 of the 44 "
                           "sequences supply 43 percent of the uncached total. "
                           "Sequence count is not monotonically related to the "
                           "governing quantity across workload types.",
            },
            "rows": workloads,
        },

        "arms": arms,

        "decode_steps": {
            "observed_token_counts": sorted(decode_tokens),
            "at_or_over_256": decode_at_or_over_256,
            "consequence": "every crossing is a prefill or mixed step, so the "
                           "perturbation originates in prefill and reaches the "
                           "decode logprobs through the KV cache it wrote. That "
                           "is why the emitted token ids never move.",
        },

        "instruments": {
            "rmsnorm_launches": "num_tokens at every RMSNorm launch, recorded "
                "at the call site in batch_invariant.py.",
            "scheduler_steps": "scheduler_output.num_scheduled_tokens per step, "
                "plus a sha1 of each request's prompt the first time it is "
                "scheduled. vLLM assigns a fresh req_id per submission, so "
                "without the prompt hash the same request cannot be followed "
                "across repeats and no per-token claim is possible.",
            "which_runner": "vLLM 0.26.0 picks between two model runners at "
                "construction and uses the one in v1/worker/gpu/ when "
                "use_v2_model_runner is set, which it is here. Patching "
                "v1/worker/gpu_model_runner.py produced no trace at all, which "
                "is how the default was discovered. Both are patched.",
            "cross_check": cross_check(main_arm),
            "perturbation": {
                "what_it_threatens": "the marginal distribution of packings: "
                    "how often a launch crosses 256, and so the divergence rate.",
                "what_it_does_not_threaten": "the conditional claim. Given the "
                    "packing that occurred, does agreement follow the widths? "
                    "Both sides of that comparison come from the same run.",
                "measured_rather_than_argued": {
                    "change": "the scheduler instrument's flush granularity was "
                              "changed from every 32 steps to every step, which "
                              "adds one write per forward pass and nothing else.",
                    "effect_on_the_marginal_rate":
                        "diverging cases went from 17 of 98 to 11 of 98",
                    "effect_on_the_conditional_claim":
                        "none. The biconditional held with zero exceptions "
                        "before the change and zero after.",
                    "reading": "the instrument demonstrably moves how often a "
                               "crossing happens and demonstrably does not move "
                               "whether a crossing implies disagreement. Rates "
                               "here are reported and never argued from.",
                    "the_earlier_run_by_arm": {
                        "main": "10 lifetimes, 9 with a divergence, 12 of 70 "
                                "cases diverging",
                        "max_num_seqs_8": "2 lifetimes, 0 with a divergence, "
                                          "0 of 14 cases",
                        "flashinfer_sampler_disabled":
                            "4 lifetimes, 3 with a divergence, 5 of 28 cases",
                        "why_these_are_literals": "the earlier run's certify "
                            "artifacts live under results/, which is gitignored, "
                            "so they are not committed and these figures cannot "
                            "be recomputed from this repository. They are "
                            "recorded here as numbers with that limitation "
                            "stated, rather than cited from files a reader "
                            "cannot open. The lifetimes in `lifetimes` below "
                            "are the committed set and are fully recomputable.",
                    },
                },
            },
        },

        "prediction": {
            "file": relpath(PREDICTION),
            "sha256": hashlib.sha256(PREDICTION.read_bytes()).hexdigest(),
            "committed_before_the_run": True,
            "note": "committed in e8601a5 before any lifetime was started. It "
                    "predicted 270 uncached prefill tokens at 44 co-resident "
                    "and 274 at 45, and both are exactly right. See "
                    "token_accounting.",
        },

        "lifetimes": lifetimes,

        "what_is_still_not_measured_here": [
            "That the nightly is clean. Phase 3 was not run; the version "
            "boundary is taken from SyaOtiLan's report and from the diff of "
            "b6cbba8, which was read directly.",
            "Whether any other kernel contributes. This shows block width "
            "accounts for every divergence observed here, not that nothing else "
            "could ever cause one.",
            "Filler count as an independent variable; it was held at 13.",
        ],
    }

    harness = envlock.capture()
    print(json.dumps({
        "claim": claim,
        "narrow_profile_consequence": consequence_profile,
        "arms": {name: {k: v for k, v in entry.items()
                        if k != "crossing_launch_sizes"}
                 for name, entry in arms.items()},
    }, indent=2))
    if args.no_artifact:
        return 0
    path = Artifact(kind="rmsnorm-blocksize", harness=harness,
                    subject=lifetimes[0]["subject"], payload=payload).write()
    print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
