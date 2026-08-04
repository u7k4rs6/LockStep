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

import json
import os
import subprocess
import sys
from pathlib import Path

from engine.kv import paged
from engine.sched.scheduler import Request, Scheduler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Result:
    relation: str
    statement: str
    passed: bool
    detail: str = ""
    evidence: dict = field(default_factory=dict)


def _run(model, requests, num_blocks, eos=None, block_size=paged.DEFAULT_BLOCK_SIZE,
         policy=None):
    scheduler = Scheduler(
        model, num_blocks=num_blocks, eos_token_ids=eos, block_size=block_size, policy=policy
    )
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


def _run_in_child(requests, num_blocks, block_size, hash_seed):
    """Run the workload in a fresh interpreter with a given PYTHONHASHSEED."""
    spec = {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "requests": [
            {"uid": r.uid, "prompt": r.prompt, "seed": r.seed,
             "max_new_tokens": r.max_new_tokens, "temperature": r.temperature,
             "top_p": r.top_p}
            for r in requests
        ],
    }
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "harness" / "mr" / "replay_child.py"), json.dumps(spec)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def mr6_cross_process(model, requests, num_blocks, block_size=paged.DEFAULT_BLOCK_SIZE) -> Result:
    """Same workload, two fresh interpreters, deliberately different hash seeds.

    An in-process replay cannot see anything fixed for the life of an
    interpreter. PYTHONHASHSEED is the concrete hazard here, because the sampler
    keys draws on the request uid: a salted hash would look perfectly
    deterministic inside one session and produce a different trajectory in the
    next. The two children are given different seeds on purpose, so agreement is
    evidence rather than coincidence.
    """
    first = _run_in_child(requests, num_blocks, block_size, hash_seed=1)
    second = _run_in_child(requests, num_blocks, block_size, hash_seed=999)

    if first["trajectory"] == second["trajectory"]:
        return Result(
            "MR6-xproc",
            "replay across processes: two fresh interpreters, identical trajectory hash",
            True,
            f"{first['steps']} steps, sha256:{first['trajectory'][:12]}, "
            f"PYTHONHASHSEED 1 vs 999",
            {"trajectory_sha256": first["trajectory"], "steps": first["steps"]},
        )
    return Result(
        "MR6-xproc",
        "replay across processes: two fresh interpreters, identical trajectory hash",
        False,
        f"seed 1 gave {first['trajectory'][:12]}, seed 999 gave {second['trajectory'][:12]}",
        {"run_a": first["trajectory"], "run_b": second["trajectory"]},
    )


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


def mr7_rng_isolation(model, requests, num_blocks, extra: Request | None = None,
                      block_size=paged.DEFAULT_BLOCK_SIZE) -> Result:
    """A request's tokens depend on (seed, uid, position) and nothing else.

    Architecture doc I4 and MR7. Swept rather than demonstrated: one deletion is
    an existence proof, and the relation claims independence from every other
    request, not from one of them.

    Four families of perturbation, each of which must leave every survivor's
    tokens bitwise unchanged:

      * remove each request in turn
      * add a request that was not there before
      * reverse the submission order, which changes admission order and
        therefore batch composition and step membership
      * re-run a single request alone in a later session, which must reproduce
        the draws it made as a cohabitant, since a uid is the whole key

    Sampling is on throughout. At temperature 0 the relation follows from batch
    invariance and never exercises the RNG keying at all.
    """
    baseline, _, _ = _run(model, _clone(requests), num_blocks, block_size=block_size)
    cases = []
    passed = True

    def compare(label, subset_outputs, expected_uids):
        nonlocal passed
        moved = [uid for uid in expected_uids if baseline.get(uid) != subset_outputs.get(uid)]
        ok = not moved
        passed &= ok
        cases.append({"perturbation": label, "survivors": len(expected_uids),
                      "unchanged": ok, "moved": moved})

    for victim in [r.uid for r in requests]:
        survivors = [r for r in requests if r.uid != victim]
        out, _, _ = _run(model, _clone(survivors), num_blocks, block_size=block_size)
        compare(f"removed {victim}", out, [r.uid for r in survivors])

    if extra is not None:
        out, _, _ = _run(model, _clone(requests) + [extra], num_blocks, block_size=block_size)
        compare(f"added {extra.uid}", out, [r.uid for r in requests])

    reversed_out, _, _ = _run(model, list(reversed(_clone(requests))), num_blocks,
                              block_size=block_size)
    compare("submission order reversed", reversed_out, [r.uid for r in requests])

    for request in requests[:3]:
        alone, _, _ = _run(model, _clone([request]), num_blocks, block_size=block_size)
        compare(f"{request.uid} alone in a later session", alone, [request.uid])

    return Result(
        "MR7",
        "RNG isolation: a request's tokens depend only on (seed, uid, position)",
        passed,
        f"{sum(1 for c in cases if c['unchanged'])}/{len(cases)} perturbations left every "
        f"survivor bitwise unchanged",
        {"cases": cases},
    )
