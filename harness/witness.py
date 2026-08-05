"""Record a case artifact that a fresh clone can replay and verify.

Every other case artifact in `evidence/` is a bug repro, and a fixed bug does not
reproduce, which is correct but makes for weak evidence: `lockstep replay` on one
reports "the engine changed under it", and a reader has to take on faith that the
replay machinery would have caught a real difference.

This records the opposite kind of artifact. A case that never failed, with its
trajectory hash, so replaying it verifies the claim the differentiator sentence
makes: execution is a pure function of (W, sigma, seeds). The hash covers emitted
tokens, raw fp16 logit bytes, the packed work list, the allocator ledger, and the
prefix cache index, so a match is a statement about all engine state rather than
about output text.

The case is deliberately not trivial. It carries a shared prefix so the cache
index participates, a chunk plan so prefill is split, a preemption so the
allocator churns, and a prompt long enough to cross the 512-token attention split
boundary, because a witness that exercises none of the machinery would pass
whatever the engine did.

    python3 -m harness.witness --out evidence/case-witness.json
    python3 -m harness.replay evidence/case-witness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from harness.sim.driver import Case, RequestSpec, run_case  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
SEED = 20260805


def build_case(vocab: int) -> Case:
    """A case that touches every subsystem the trajectory hash covers."""
    import random

    rng = random.Random(SEED)

    def toks(n: int) -> tuple[int, ...]:
        return tuple(rng.randrange(1000, vocab) for _ in range(n))

    # 544 tokens crosses the 512 split boundary, so the split-combine fold runs
    # and a change to its order would move the hash.
    shared = toks(544)
    return Case(
        requests=(
            RequestSpec(uid="w00", prompt=shared + toks(9), seed=11,
                        max_new_tokens=6),
            RequestSpec(uid="w01", prompt=shared + toks(17), seed=22,
                        max_new_tokens=6),
            RequestSpec(uid="w02", prompt=toks(37), seed=33, max_new_tokens=6),
        ),
        chunk_plan=(128, 384, 64),
        preempt_at=(("w01", 2),),
        block_size=16,
        num_blocks=192,
        enable_cache=True,
        shared_prefix_len=544,
        label="replay witness",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "evidence" / "case-witness.json")
    args = parser.parse_args()

    model = Qwen3(str(WEIGHTS), max_len=1024)
    case = build_case(model.cfg.vocab_size)

    first = run_case(model, case)
    if first.error:
        raise SystemExit(f"the witness case is not clean: {first.error}")

    # Recorded only after it has already replayed once in this process. A hash
    # written from a single execution asserts nothing about determinism.
    second = run_case(model, case)
    if second.trajectory != first.trajectory:
        raise SystemExit(
            f"refusing to record: two runs in one process disagree, "
            f"{first.trajectory[:16]} against {second.trajectory[:16]}"
        )

    env = envlock.capture()
    # Written straight to the named path rather than through `Artifact.write`,
    # which interposes a date directory. Evidence is cited by a stable filename
    # so a README link does not move every time it is regenerated.
    artifact = Artifact(kind="case", env=env, payload={
        "reason": "",
        "minimized": case.to_dict(),
        "trajectory": first.trajectory,
        "steps": first.steps,
        "reproduces": True,
        "one_minimal": False,
        "checks_run": 2,
        "note": (
            "A replay-determinism witness, not a bug repro. Replaying it "
            "verifies that the same (W, sigma, seeds) still produces the same "
            "trajectory hash over emitted tokens, raw fp16 logit bytes, the "
            "packed work list, the allocator ledger, and the prefix cache index."
        ),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")

    print(f"witness recorded  {relpath(args.out)}")
    print(f"  trajectory      {first.trajectory}")
    print(f"  steps           {first.steps}")
    print(f"  env             {env.fingerprint()}")
    print(f"  engine          {env.engine_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
