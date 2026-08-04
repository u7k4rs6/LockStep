"""Run every metamorphic relation the engine can currently support.

Frontend spec 1.2: quiet and factual, one line per meaningful event, never a
percentage without its denominator, and every bitwise claim tagged with the
environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine import envlock  # noqa: E402
from engine.kv import paged  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.scheduler import Request  # noqa: E402
from harness.mr.equivalence import (  # noqa: E402
    decode_vs_prefill_kv,
    mr1_batch_composition,
    mr2_chunk_partition,
)
from harness.mr.relations import (  # noqa: E402
    mr6_cross_process,
    mr6_replay,
    mr7_rng_isolation,
)
from report.artifact import Artifact, relpath  # noqa: E402

NOT_YET = {
    "MR3": "preempt and resume; needs recompute preemption (week 4)",
    "MR4": "cache cold versus warm; needs the prefix cache (week 4)",
    "MR5": "padding and occupancy; needs the fuzzer (week 5)",
    "MR8": "temperature-0 tie-break under batch permutation (week 5)",
    "MR9": "mask no-op, fidelity side (week 4)",
}

# 600 tokens so that the 512 split boundary is reachable. Below it, every MR2
# case at split_size is silently skipped and the table looks complete.
LONG_PROMPT = 600


def workload(vocab: int, count: int = 6, seed: int = 31337) -> list[Request]:
    generator = torch.Generator().manual_seed(seed)
    return [
        Request(
            uid=f"r{index:02d}",
            prompt=torch.randint(0, vocab, (6 + 5 * (index % 4),), generator=generator).tolist(),
            seed=1000 + index,
            max_new_tokens=6,
            temperature=0.8,
            top_p=0.95,
        )
        for index in range(count)
    ]


def uneven_prompts(vocab: int, count: int, seed: int = 4242) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    lengths = [96, 33, 8, 21, 25, 41, 12, 17]
    return [
        torch.randint(0, vocab, (lengths[i % len(lengths)] + (i % 5),), generator=generator).tolist()
        for i in range(count)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[16, 64],
        help="KV block sizes to sweep the relations over.",
    )
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=1024)
    vocab = model.cfg.vocab_size
    env = envlock.capture()

    generator = torch.Generator().manual_seed(99)
    long_prompt = torch.randint(0, vocab, (LONG_PROMPT,), generator=generator).tolist()
    requests = workload(vocab)
    extra = Request(uid="rXX", prompt=[11, 22, 33], seed=7, max_new_tokens=6,
                    temperature=0.8, top_p=0.95)

    print("lockstep verify")
    print(f"  block sizes swept   {args.block_sizes}")
    print(f"  long prompt         {LONG_PROMPT} tokens (reaches the 512 split boundary)")
    print(f"  sampling            T=0.8 top_p=0.95 for the RNG relations")
    print()

    results = []
    for block_size in args.block_sizes:
        print(f"block_size = {block_size}")
        block_results = [
            decode_vs_prefill_kv(model, long_prompt, block_size=block_size),
            mr1_batch_composition(model, uneven_prompts(vocab, 32), block_size=block_size),
            mr2_chunk_partition(model, long_prompt, block_size=block_size),
            mr6_replay(model, requests, num_blocks=256),
            mr6_cross_process(model, requests, num_blocks=256, block_size=block_size),
            mr7_rng_isolation(model, requests, num_blocks=256, extra=extra,
                              block_size=block_size),
        ]
        for result in block_results:
            mark = "pass" if result.passed else "FAIL"
            print(f"  [{mark}]  {result.relation:<10} {result.detail}")
            for case in getattr(result, "cases", []) or []:
                if "partition" in case:
                    status = "ok" if case["kv_identical"] and case["logits_identical"] else "DIVERGED"
                    print(f"            {case['partition']:<40} chunks={case['chunks']:<4} {status}")
        results.append({"block_size": block_size, "results": block_results})
        print()

    print("  not yet runnable, and deliberately not stubbed green:")
    for relation, reason in NOT_YET.items():
        print(f"    {relation}  {reason}")

    flat = [r for entry in results for r in entry["results"]]
    passed = sum(1 for r in flat if r.passed)
    print()
    print(f"  {passed}/{len(flat)} relation runs pass across {len(args.block_sizes)} block sizes")
    print(f"env  {env.fingerprint()}")

    if not args.no_artifact:
        path = Artifact(
            kind="verify",
            env=env,
            payload={
                "block_sizes": args.block_sizes,
                "long_prompt_tokens": LONG_PROMPT,
                "runs": [
                    {
                        "block_size": entry["block_size"],
                        "relations": [
                            {
                                "id": r.relation,
                                "statement": r.statement,
                                "passed": r.passed,
                                "detail": r.detail,
                                "cases": getattr(r, "cases", None) or getattr(r, "evidence", None),
                            }
                            for r in entry["results"]
                        ],
                    }
                    for entry in results
                ],
                "not_yet_runnable": NOT_YET,
                "passed": passed,
                "total": len(flat),
            },
        ).write()
        print(f"artifact  {relpath(path)}")

    return 0 if passed == len(flat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
