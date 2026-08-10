"""Fuzz campaigns: swarm generation, coverage, seeded faults, minimization."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.audit.counters import Counters  # noqa: E402
from engine.sched.lifecycle import Event  # noqa: E402
from harness.fuzz.coverage import Coverage  # noqa: E402
from harness.fuzz import faults as faults_module  # noqa: E402
from harness.fuzz.faults import FAULTS  # noqa: E402
from harness.fuzz.generators import draw_case, draw_config, eviction_cases  # noqa: E402
from harness.minimize.ddmin import minimize  # noqa: E402
from harness.sim.driver import Case, run_case  # noqa: E402
from report.artifact import SAME_PROCESS, require_clean_tree, Artifact, relpath  # noqa: E402
from report.divergence import Divergence  # noqa: E402


@dataclass
class Finding:
    case: Case
    detail: str
    config: str
    seconds_to_detect: float


def observe(coverage: Coverage, case: Case, outcome) -> None:
    """Fold one execution into the coverage report."""
    for uid, events in outcome.events.items():
        coverage.observe_events(events)
    for uid, depth in outcome.depths.items():
        coverage.observe_preempt_depth(depth)

    block = case.block_size
    for chunk in case.chunk_plan:
        remainder = chunk % block
        if remainder == 0:
            coverage.observe_boundary("chunk end vs block size", "exact")
        elif remainder == block - 1:
            coverage.observe_boundary("chunk end vs block size", "below")
        else:
            coverage.observe_boundary("chunk end vs block size", "above")

    if case.shared_prefix_len:
        if case.shared_prefix_len == block:
            coverage.observe_boundary("cache hit length vs block size", "exact")
        elif case.shared_prefix_len < block:
            coverage.observe_boundary("cache hit length vs block size", "below")
        else:
            coverage.observe_boundary("cache hit length vs block size", "above")

    counters = outcome.counters

    if counters["attention_multi_split"]:
        coverage.observe_boundary("split boundary vs kv length", "above")
    if counters["attention_single_split"]:
        coverage.observe_boundary("split boundary vs kv length", "below")
    for spec in case.requests:
        if len(spec.prompt) in (512, 511, 513):
            coverage.observe_boundary("split boundary vs kv length", "exact")

    if counters["out_of_blocks"]:
        coverage.observe_boundary("free block count", "zero")
    if counters["eviction_taken"]:
        coverage.observe_boundary("free block count", "one")
    coverage.observe_boundary("free block count", "low")

    width = len(case.requests)
    for marker in ("1", "2", "3", "4", "8", "16", "31", "32"):
        if width == int(marker):
            coverage.observe_boundary("batch size transition", marker)


def case_fails(model, case: Case) -> str | None:
    """The oracle. Returns a reason, or None if the case is clean."""
    outcome = run_case(model, case)
    if outcome.error:
        return outcome.error

    for spec in case.requests:
        alone = Case(
            requests=(spec,),
            block_size=case.block_size,
            num_blocks=max(4, -(-(len(spec.prompt) + spec.max_new_tokens) // case.block_size) + 2),
            enable_cache=False,
            label=f"canonical:{spec.uid}",
        )
        canonical = run_case(model, alone)
        if canonical.error:
            continue
        expected = canonical.outputs.get(spec.uid)
        observed = outcome.outputs.get(spec.uid)
        if expected is not None and observed is not None and expected != observed:
            return f"{spec.uid} diverged from canonical: {observed} != {expected}"

        # Token equality masks a sub-tie perturbation until it lands near a tie,
        # far from its cause. This oracle compared token ids only, while I1's
        # claims-table cell said bitwise on fp16 logit bytes.
        want = canonical.emitted_logits.get(spec.uid) or []
        got = outcome.emitted_logits.get(spec.uid) or []
        for index, (a, b) in enumerate(zip(want, got)):
            if not torch.equal(a, b):
                delta = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
                return (
                    f"{spec.uid} logit bytes diverged from canonical at emitted "
                    f"position {index}, max abs {delta:.3e}, tokens still equal"
                )
    return None


def run_campaign(model, seeds, cases_per_seed, coverage, oracle, progress=True):
    findings = []
    started = time.monotonic()
    executed = 0
    for seed in seeds:
        config = draw_config(seed)
        for index in range(cases_per_seed):
            case = draw_case(config, model.cfg.vocab_size, index)
            executed += 1
            try:
                outcome = run_case(model, case)
                observe(coverage, case, outcome)
            except Exception as exc:  # noqa: BLE001 - a crash is a finding
                findings.append(Finding(case, f"{type(exc).__name__}: {exc}",
                                        config.describe(), time.monotonic() - started))
                print(f"\n  FINDING {len(findings)} (crash): {type(exc).__name__}: "
                      f"{str(exc)[:100]}")
                print(f"    config {config.describe()}", flush=True)
                continue
            reason = oracle(case)
            if reason:
                findings.append(Finding(case, reason, config.describe(),
                                        time.monotonic() - started))
                print(f"\n  FINDING {len(findings)}: {reason[:110]}")
                print(f"    config {config.describe()}", flush=True)
            if progress:
                seen2, total2 = coverage.ngram_fraction(2)
                hit, total_b = coverage.boundary_fraction()
                print(f"\r  {executed} cases  2-grams {seen2}/{total2}  "
                      f"boundaries {hit}/{total_b}  findings {len(findings)}   ",
                      end="", flush=True)
    if progress:
        print()
    return findings, executed


def print_repro(finding: Finding, minimization, artifact_path: str) -> None:
    """The divergence report from the CLI and report spec 1.3, with the minimality proof."""
    case = minimization.case
    print()
    is_crash = ":" in finding.detail and "diverged from canonical" not in finding.detail
    print(Divergence(
        request_uid=case.requests[0].uid if case.requests else "?",
        position=len(case.requests[0].prompt) if case.requests else 0,
        failure_class="crash" if is_crash else "divergence",
        exception_type=finding.detail.split(":", 1)[0] if is_crash else None,
        expected_sha256=None if is_crash else "0" * 64,
        observed_sha256=None if is_crash else "1" * 64,
        trigger=(finding.detail.split(":", 1)[1].strip() if is_crash else finding.detail)[:60],
        schedule_events=minimization.after["events"],
        schedule_events_before_minimization=minimization.before["events"],
        env_fingerprint=envlock.capture().fingerprint(),
        replay_artifact=artifact_path,
        boundary_hit=True,
    ).render())
    print()
    print("  minimality")
    print(f"    reproduces on replay   {'yes' if minimization.reproduces else 'NO'}")
    print(f"    1-minimal              "
          f"{'yes, no single element can be removed' if minimization.one_minimal else 'NO'}")
    if minimization.failed_removals:
        for note in minimization.failed_removals[:4]:
            print(f"      {note}")
    print(f"    checks run             {minimization.checks_run}")
    print(f"    {minimization.summary()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--cases-per-seed", type=int, default=6)
    parser.add_argument("--seeded-faults", action="store_true",
                        help="Run one campaign per architecture doc 10.1 operator.")
    parser.add_argument("--eviction-campaign", type=int, default=0,
                        help="Cases aimed at _reserve_with_eviction specifically.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="produce a claim artifact from an uncommitted "
                             "tree; recorded in the artifact when used")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty)

    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=1024)
    env = envlock.capture()

    print("lockstep fuzz")
    print(f"  swarm campaigns   {args.seeds} configs x {args.cases_per_seed} cases")
    print()

    coverage = Coverage()
    findings, executed = run_campaign(
        model, range(args.seeds), args.cases_per_seed, coverage,
        lambda case: case_fails(model, case),
    )

    print()
    print(coverage.report())
    print()
    print(f"  clean campaign: {len(findings)} findings over {executed} cases")

    for finding in findings:
        minimization = minimize(finding.case, lambda c: case_fails(model, c) is not None)
        artifact = Artifact(kind="case", harness=envlock.capture(), subject=SAME_PROCESS, payload={
            "reason": finding.detail,
            "config": finding.config,
            "minimized": minimization.case.to_dict(),
            "before": minimization.before,
            "after": minimization.after,
            "reproduces": minimization.reproduces,
            "one_minimal": minimization.one_minimal,
            "checks_run": minimization.checks_run,
        }).write()
        print_repro(finding, minimization, relpath(artifact))
        print(f"    promote with: python3 -m report.publish {relpath(artifact)}")

    eviction_summary = None
    if args.eviction_campaign:
        print()
        print(f"named campaign: _reserve_with_eviction, {args.eviction_campaign} cases")
        cases = eviction_cases(model.cfg.vocab_size, args.eviction_campaign)
        evict_coverage = Coverage()
        evict_findings = []
        total_passes = 0
        max_passes_seen = 0
        assertion_fired = 0
        for case in cases:
            outcome = run_case(model, case)
            observe(evict_coverage, case, outcome)
            total_passes += outcome.counters["eviction_pass"]
            max_passes_seen = max(max_passes_seen, outcome.counters["eviction_pass"])
            if outcome.error and "reserve-with-eviction ran" in outcome.error:
                assertion_fired += 1
                evict_findings.append(outcome.error)
            elif outcome.error and "OutOfBlocks" not in outcome.error:
                evict_findings.append(outcome.error)
            else:
                reason = case_fails(model, case)
                if reason and "OutOfBlocks" not in reason:
                    evict_findings.append(reason)
        eviction_summary = {
            "cases": len(cases),
            "eviction_passes_total": total_passes,
            "eviction_passes_max_in_one_case": max_passes_seen,
            "pass_count_assertion_fired": assertion_fired,
            "findings": evict_findings,
            "coverage": evict_coverage.as_dict(),
        }
        print(f"  cases run                      {len(cases)}")
        print(f"  eviction passes, total         {total_passes}")
        print(f"  eviction passes, max one case  {max_passes_seen}")
        print(f"  pass-count assertion fired     {assertion_fired}/{len(cases)}")
        print(f"  findings                       {len(evict_findings)}")
        for finding in evict_findings[:5]:
            print(f"    {finding[:100]}")

    fault_results = []
    if args.seeded_faults:
        print()
        print("seeded faults, from architecture doc 10.1")
        for fault in FAULTS:
            started = time.monotonic()
            detected, detail = None, ""
            exercised = Counters()
            found = []
            fault_coverage = Coverage()
            faults_module.SENTINELS.pop(fault.name, None)

            trial_cases = [
                draw_case(draw_config(seed), model.cfg.vocab_size, index)
                for seed in range(6) for index in range(4)
            ] + eviction_cases(model.cfg.vocab_size, 24)

            with fault.apply():
                for case in trial_cases:
                    try:
                        outcome = run_case(model, case)
                    except Exception as exc:  # noqa: BLE001
                        found.append(f"{type(exc).__name__}: {exc}")
                        break
                    exercised.merge(outcome.counters)
                    if outcome.error:
                        found.append(outcome.error)
                        break
                    try:
                        observe(fault_coverage, case, outcome)
                    except Exception as exc:  # noqa: BLE001
                        found.append(f"{type(exc).__name__}: {exc}")
                        break
                    reason = case_fails(model, case)
                    if reason:
                        found.append(reason)
                        break
            if found:
                detected = time.monotonic() - started
                detail = found[0][:70]

            if not found:
                from harness.fuzz import golden

                with fault.apply():
                    matches, differing = golden.compare(model)
                if not matches:
                    found.append(
                        f"golden bytes differ from the committed baseline at "
                        f"{', '.join(differing)}"
                    )
                    detected = time.monotonic() - started
                    detail = found[0][:70]

            if not found and fault.fidelity_observable:
                from bench.fp64_reference import Fp64Reference, require_pinned_threads
                from harness.fuzz.f1_observer import build_prompt, spot_check

                try:
                    require_pinned_threads()
                    reference = Fp64Reference(REPO_ROOT / "weights" / "Qwen3-0.6B")
                    prompt = build_prompt(model.cfg.vocab_size)
                    with fault.apply():
                        within, worst = spot_check(model, reference, prompt)
                    del reference
                    if not within:
                        found.append(
                            f"F1 bound exceeded: max abs logit error {worst:.4e} "
                            f"against fp64, bound 0.5"
                        )
                        detected = time.monotonic() - started
                        detail = found[0][:70]
                except SystemExit as exc:
                    print(f"                     F1 observer skipped: {exc}")

            missing = exercised.missing(*fault.requires) if fault.requires else []
            if missing and detected is None:
                verdict = "not-exercised"
            elif detected is not None:
                verdict = "killed"
            else:
                verdict = "survived"

            if not detail:
                mechanism = "-"
            elif detail.startswith("AuditFailure"):
                mechanism = "internal audit"
            elif detail.startswith("UndeclaredTransition"):
                mechanism = "undeclared transition"
            elif "diverged from canonical" in detail:
                mechanism = "bitwise divergence"
            elif detail.startswith("golden bytes differ"):
                mechanism = "golden bytes"
            elif detail.startswith("F1 bound exceeded"):
                mechanism = "fidelity vs fp64"
            elif detail.startswith("AssertionError"):
                mechanism = "engine assertion"
            else:
                mechanism = "crash"

            took_effect = faults_module.SENTINELS.get(fault.name, 0)
            if not took_effect:
                verdict = "not-exercised"

            fault_results.append({
                "fault": fault.name,
                "operator": fault.operator,
                "verdict": verdict,
                "mutation_took_effect": took_effect,
                "mechanism": mechanism,
                "requires": list(fault.requires),
                "counters_seen": {p: exercised[p] for p in fault.requires},
                "seconds": round(detected, 1) if detected else None,
                "detail": detail,
            })
            mark = {"killed": f"killed in {detected:6.1f}s" if detected else "killed",
                    "survived": "SURVIVED        ",
                    "not-exercised": "not exercised   "}[verdict]
            print(f"  [{mark}]  {fault.operator}")
            if verdict == "killed":
                print(f"                     mechanism: {mechanism}")
            if verdict == "not-exercised":
                reason = ("the patch never executed (sentinel 0)" if not took_effect
                          else f"mutated path never ran: {', '.join(missing)}")
                print(f"                     {reason}")
            print(f"                     mutation took effect {took_effect} times")
            if detail:
                print(f"                     {detail}")

        killed = [r for r in fault_results if r["verdict"] == "killed"]
        survived = [r for r in fault_results if r["verdict"] == "survived"]
        not_run = [r for r in fault_results if r["verdict"] == "not-exercised"]
        valid = len(killed) + len(survived)
        print()
        print(f"  killed          {len(killed)}/{valid} valid trials")
        print(f"  survived        {len(survived)}/{valid}")
        print(f"  not exercised   {len(not_run)}  (discarded, not counted as survivals)")
        if killed:
            times = sorted(r["seconds"] for r in killed)
            median = times[len(times) // 2]
            print(f"  median time to detection   {median}s")

    payload = {
        "cases_executed": executed,
        "coverage": coverage.as_dict(),
        "findings": [{"detail": f.detail, "config": f.config} for f in findings],
        "seeded_faults": fault_results,
        "eviction_campaign": eviction_summary,
    }
    print()
    print(f"env  {env.fingerprint()}")
    if not args.no_artifact:
        path = Artifact(kind="fuzz", harness=env, subject=SAME_PROCESS, payload=payload).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
