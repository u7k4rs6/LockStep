"""Metamorphic relations. Week 2 implements MR6 and MR7.

docs/02-technical-architecture.md section 9 lists nine. The rest arrive with the
engine features they need: MR1 and MR5 need batch composition control, MR2 needs
chunked prefill, MR3 needs preemption, MR4 needs the prefix cache. Stubbing them
now would produce green results for relations that are not being tested, which is
worse than their absence.

Each relation returns a `Result` rather than asserting, so a runner can print a
table and so a failing relation can carry the information the divergence report
needs instead of a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.sched.scheduler import Request, Scheduler


@dataclass
class Result:
    relation: str
    statement: str
    passed: bool
    detail: str = ""
    evidence: dict = field(default_factory=dict)


def _run(model, requests: list[Request], num_blocks: int, eos=None) -> tuple[dict, str, list]:
    scheduler = Scheduler(model, num_blocks=num_blocks, eos_token_ids=eos)
    for request in requests:
        scheduler.submit(request)
    outputs = scheduler.run()
    return outputs, scheduler.trajectory.hexdigest(), scheduler.trajectory.per_step


def _clone(requests: list[Request]) -> list[Request]:
    return [
        Request(
            uid=r.uid,
            prompt=list(r.prompt),
            seed=r.seed,
            max_new_tokens=r.max_new_tokens,
            temperature=r.temperature,
            top_p=r.top_p,
        )
        for r in requests
    ]


def mr6_replay(model, requests: list[Request], num_blocks: int) -> Result:
    """Identical (W, sigma, seeds) twice yields an identical trajectory hash.

    Architecture doc I3. The hash covers emitted tokens, raw fp16 logit bytes,
    the packed work list, and the allocator ledger, so this is a stronger claim
    than "the same text came out twice".
    """
    _, digest_a, steps_a = _run(model, _clone(requests), num_blocks)
    _, digest_b, steps_b = _run(model, _clone(requests), num_blocks)

    if digest_a == digest_b:
        return Result(
            "MR6",
            "replay: identical inputs twice, identical trajectory hash",
            True,
            f"{len(steps_a)} steps, sha256:{digest_a[:12]}",
            {"trajectory_sha256": digest_a, "steps": len(steps_a)},
        )

    first = next(
        (i for i, (x, y) in enumerate(zip(steps_a, steps_b)) if x != y),
        min(len(steps_a), len(steps_b)),
    )
    return Result(
        "MR6",
        "replay: identical inputs twice, identical trajectory hash",
        False,
        f"first divergence at step {first}",
        {"run_a": digest_a, "run_b": digest_b, "first_divergent_step": first},
    )


def mr7_rng_isolation(model, requests: list[Request], num_blocks: int) -> Result:
    """Deleting request A leaves request B's tokens unchanged.

    Architecture doc I4 and MR7. Run with sampling on, because with temperature 0
    the relation is implied by batch invariance and says nothing about the RNG.
    The point is that the draw is keyed on (seed, uid, position) and never on a
    global step counter, so removing a cohabitant cannot shift anyone's stream.
    """
    victim = requests[0].uid
    survivors = [r for r in requests if r.uid != victim]

    with_all, _, _ = _run(model, _clone(requests), num_blocks)
    without, _, _ = _run(model, _clone(survivors), num_blocks)

    moved = [
        uid for uid in without if with_all.get(uid) != without[uid]
    ]
    if not moved:
        return Result(
            "MR7",
            "RNG isolation: deleting a request leaves the others' tokens unchanged",
            True,
            f"removed {victim}, {len(without)} survivors bitwise unchanged",
            {"removed": victim, "survivors": len(without)},
        )
    return Result(
        "MR7",
        "RNG isolation: deleting a request leaves the others' tokens unchanged",
        False,
        f"removing {victim} moved {len(moved)} other requests: {moved}",
        {"removed": victim, "moved": moved},
    )
