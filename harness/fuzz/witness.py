"""Reachability witnesses for declared lifecycle transitions.

Separate from the campaign on purpose. The coverage report answers "what does the
standard swarm campaign explore unaided", and a purpose-built probe must never
inflate that number: a transition reached only by a case designed to reach it
says nothing about whether the generator explores the space.

This answers the other question, which is whether a declared transition is
reachable at all. A transition the campaign never takes is either rare or dead,
and the difference decides whether it belongs in the denominator. Two of the
three unwitnessed after a 132-case campaign turned out to be rare, and the third
was dead and had been inflating the denominator since the table was written.

    python3 -m harness.fuzz.witness

Reports, per declared transition: reached by the standard campaign, reached only
by a targeted probe, or reached by neither, which is a claim that it should be
moved to UNREACHABLE_BY_DESIGN with an argument.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.lifecycle import TRANSITIONS, UNREACHABLE_BY_DESIGN  # noqa: E402
from harness.fuzz.coverage import Coverage  # noqa: E402
from harness.fuzz.generators import draw_case, draw_config, eviction_cases  # noqa: E402
from harness.sim.driver import Case, RequestSpec, run_case  # noqa: E402


def targeted_cases(vocab: int, seed: int = 4242) -> list[Case]:
    """Cases shaped at the transitions a uniform campaign reaches rarely.

    Shape A drives a resumed request re-prefilling in chunks while it already
    holds generated tokens, which is the only way a request is preemptable
    mid-prefill. Shape B drives cache hits under allocation pressure, so eviction
    runs inside `_admit` after a hit has been applied.
    """
    rng = random.Random(seed)
    cases: list[Case] = []

    for trial in range(40):
        bs = rng.choice((8, 16, 32))
        shared = tuple(rng.randrange(vocab) for _ in range(bs * 2))
        reqs = [
            RequestSpec(uid=f"r{i:02d}",
                        prompt=shared + tuple(rng.randrange(vocab)
                                              for _ in range(rng.randint(5, 40))),
                        seed=100 + i, max_new_tokens=4)
            for i in range(rng.randint(2, 5))
        ]
        cases.append(Case(
            requests=tuple(reqs), chunk_plan=(1, 3, 7),
            preempt_at=tuple((r.uid, s) for r in reqs for s in range(1, 6)),
            block_size=bs,
            num_blocks=max(6, sum(-(-(len(r.prompt) + 4) // bs) for r in reqs) // 2),
            enable_cache=True, shared_prefix_len=bs * 2, label=f"witness-a{trial}",
        ))

    for trial in range(40):
        bs = rng.choice((8, 16))
        shared = tuple(rng.randrange(vocab) for _ in range(bs * 3))
        reqs = [
            RequestSpec(uid=f"r{i:02d}",
                        prompt=shared + tuple(rng.randrange(vocab)
                                              for _ in range(rng.randint(1, bs))),
                        seed=200 + i, max_new_tokens=2)
            for i in range(rng.randint(3, 6))
        ]
        need = sum(-(-(len(r.prompt) + 2) // bs) for r in reqs)
        cases.append(Case(
            requests=tuple(reqs), chunk_plan=(bs,),
            preempt_at=tuple((reqs[0].uid, s) for s in range(1, 4)),
            block_size=bs, num_blocks=max(4, int(need * 0.45)),
            enable_cache=True, shared_prefix_len=bs * 3, label=f"witness-b{trial}",
        ))
    return cases


def gather(model, cases) -> Coverage:
    coverage = Coverage()
    for case in cases:
        try:
            outcome = run_case(model, case)
        except Exception:  # noqa: BLE001 - a crash is the campaign's business
            continue
        for events in outcome.events.values():
            coverage.observe_transitions(events)
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--cases-per-seed", type=int, default=6)
    args = parser.parse_args()

    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=1024)
    vocab = model.cfg.vocab_size

    standard = [draw_case(draw_config(s), vocab, i)
                for s in range(args.seeds) for i in range(args.cases_per_seed)]
    standard += eviction_cases(vocab, 60)

    print(f"standard campaign  {len(standard)} cases")
    unaided = gather(model, standard)
    print(f"targeted probe     {len(targeted_cases(vocab))} cases")
    probed = gather(model, targeted_cases(vocab))

    print()
    print(f"  {'transition':<34} {'campaign':>9} {'targeted':>9}")
    print("  " + "-" * 56)
    dead = []
    for (state, event) in sorted(TRANSITIONS, key=lambda k: (k[0].value, k[1].value)):
        a = (state, event) in unaided.witnessed_transitions
        b = (state, event) in probed.witnessed_transitions
        if not (a or b):
            dead.append((state.value, event.value))
        print(f"  ({state.value}, {event.value})".ljust(36)
              + f"{'yes' if a else 'no':>9} {'yes' if b else 'no':>9}")

    print()
    print(f"  reached by the standard campaign unaided   "
          f"{len(unaided.witnessed_transitions)}/{len(TRANSITIONS)}")
    print(f"  reachable at all, with a targeted probe    "
          f"{len(unaided.witnessed_transitions | probed.witnessed_transitions)}"
          f"/{len(TRANSITIONS)}")
    print(f"  declared unreachable, with an argument     {len(UNREACHABLE_BY_DESIGN)}")
    if dead:
        print()
        print("  NEITHER reached these. Each is a claim that it belongs in")
        print("  UNREACHABLE_BY_DESIGN with an argument, or that the probe is too weak:")
        for entry in dead:
            print(f"    {entry}")
    return 1 if dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
