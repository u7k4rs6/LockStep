"""Reproduce a campaign finding, minimize it, and verify the minimization.

Run:  python3 -m harness.minimize.demo --seed 20 --index N
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from harness.fuzz.campaign import case_fails, print_repro, Finding  # noqa: E402
from harness.fuzz.generators import draw_case, draw_config  # noqa: E402
from harness.minimize.ddmin import minimize  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[18, 20])
    parser.add_argument("--cases-per-seed", type=int, default=8)
    args = parser.parse_args()

    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=1024)

    found = None
    for seed in args.seeds:
        config = draw_config(seed)
        for index in range(args.cases_per_seed):
            case = draw_case(config, model.cfg.vocab_size, index)
            reason = case_fails(model, case)
            if reason:
                found = (case, reason, config)
                break
        if found:
            break

    if not found:
        print("no finding reproduced; the engine may already be fixed")
        return 0

    case, reason, config = found
    print(f"reproduced: {reason}")
    print(f"  config    {config.describe()}")
    print(f"  case      {len(case.requests)} requests, "
          f"{sum(len(r.prompt) for r in case.requests)} prompt tokens, "
          f"{len(case.preempt_at)} preemptions, {len(case.refuse_cache_at)} cache refusals, "
          f"{len(case.chunk_plan)} chunk plan entries")
    print()
    print("minimizing (requests, then schedule events, then prompt tokens)")

    result = minimize(case, lambda c: case_fails(model, c) is not None)

    env = envlock.capture()
    path = Artifact(
        kind="case",
        env=env,
        payload={
            "reason": reason,
            "config": config.describe(),
            "minimized": {
                "requests": [
                    {"uid": r.uid, "prompt": list(r.prompt), "seed": r.seed,
                     "max_new_tokens": r.max_new_tokens, "temperature": r.temperature}
                    for r in result.case.requests
                ],
                "chunk_plan": list(result.case.chunk_plan),
                "preempt_at": [list(p) for p in result.case.preempt_at],
                "refuse_cache_at": list(result.case.refuse_cache_at),
                "block_size": result.case.block_size,
                "num_blocks": result.case.num_blocks,
                "enable_cache": result.case.enable_cache,
            },
            "before": result.before,
            "after": result.after,
            "reproduces": result.reproduces,
            "one_minimal": result.one_minimal,
            "checks_run": result.checks_run,
        },
    ).write()

    print_repro(model, Finding(result.case, reason, config.describe(), 0.0),
                result, relpath(path))
    print()
    print("  the minimized case, in full")
    for r in result.case.requests:
        print(f"    request {r.uid}  prompt={list(r.prompt)}  seed={r.seed}  "
              f"max_new_tokens={r.max_new_tokens}  T={r.temperature}")
    print(f"    block_size={result.case.block_size}  num_blocks={result.case.num_blocks}  "
          f"cache={'on' if result.case.enable_cache else 'off'}")
    print(f"    chunk_plan={list(result.case.chunk_plan)}  "
          f"preempt_at={[list(p) for p in result.case.preempt_at]}  "
          f"refuse_cache_at={list(result.case.refuse_cache_at)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
