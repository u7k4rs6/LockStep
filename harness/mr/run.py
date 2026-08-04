"""Run the metamorphic relations that week 2's engine can support.

Frontend spec 1.2: quiet and factual, one line per meaningful event, never a
percentage without its denominator, and every bitwise claim tagged with the
environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.scheduler import Request  # noqa: E402
from harness.mr.relations import mr6_replay, mr7_rng_isolation  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

NOT_YET = {
    "MR1": "batch composition; needs the fuzzer's batch control (week 3)",
    "MR2": "chunk partition; needs chunked prefill (week 4)",
    "MR3": "preempt and resume; needs recompute preemption (week 4)",
    "MR4": "cache cold versus warm; needs the prefix cache (week 4)",
    "MR5": "padding and occupancy; needs the fuzzer (week 3)",
    "MR8": "temperature-0 tie-break under batch permutation (week 3)",
    "MR9": "mask no-op, fidelity side (week 4)",
}


def workload(vocab: int, count: int = 6, seed: int = 31337) -> list[Request]:
    """A small mixed workload: uneven prompts, sampling on, distinct seeds.

    Sampling is on because MR7 says nothing at temperature 0, where the relation
    follows from batch invariance and never exercises the RNG keying at all.
    """
    generator = torch.Generator().manual_seed(seed)
    requests = []
    for index in range(count):
        length = 6 + 5 * (index % 4)
        requests.append(
            Request(
                uid=f"r{index:02d}",
                prompt=torch.randint(0, vocab, (length,), generator=generator).tolist(),
                seed=1000 + index,
                max_new_tokens=6,
                temperature=0.8,
                top_p=0.95,
            )
        )
    return requests


def main() -> int:
    weights = REPO_ROOT / "weights" / "Qwen3-0.6B"
    model = Qwen3(weights, max_len=256)
    requests = workload(model.cfg.vocab_size)
    env = envlock.capture()

    print(f"lockstep verify  {len(requests)} requests, sampling on (T=0.8, top_p=0.95)")
    print()

    results = [
        mr6_replay(model, requests, num_blocks=256),
        mr7_rng_isolation(model, requests, num_blocks=256),
    ]

    for result in results:
        mark = "pass" if result.passed else "FAIL"
        print(f"  [{mark}]  {result.relation}  {result.statement}")
        print(f"          {result.detail}")

    print()
    print("  not yet runnable, and deliberately not stubbed green:")
    for relation, reason in NOT_YET.items():
        print(f"    {relation}  {reason}")

    passed = sum(1 for r in results if r.passed)
    print()
    print(f"  {passed}/{len(results)} relations pass")
    print(f"env  {env.fingerprint()}")

    path = Artifact(
        kind="verify",
        env=env,
        payload={
            "relations": [
                {
                    "id": r.relation,
                    "statement": r.statement,
                    "passed": r.passed,
                    "detail": r.detail,
                    "evidence": r.evidence,
                }
                for r in results
            ],
            "not_yet_runnable": NOT_YET,
            "passed": passed,
            "total": len(results),
        },
    ).write()
    print(f"artifact  {relpath(path)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
