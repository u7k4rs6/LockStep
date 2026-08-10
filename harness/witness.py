"""Record a case artifact that a fresh clone can replay and verify."""

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
from report.artifact import SAME_PROCESS, Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
SEED = 20260805


def build_case(vocab: int) -> Case:
    """A case that touches every subsystem the trajectory hash covers."""
    import random

    rng = random.Random(SEED)

    def toks(n: int) -> tuple[int, ...]:
        return tuple(rng.randrange(1000, vocab) for _ in range(n))

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

    second = run_case(model, case)
    if second.trajectory != first.trajectory:
        raise SystemExit(
            f"refusing to record: two runs in one process disagree, "
            f"{first.trajectory[:16]} against {second.trajectory[:16]}"
        )

    env = envlock.capture()
    artifact = Artifact(kind="case", harness=env, subject=SAME_PROCESS, payload={
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
