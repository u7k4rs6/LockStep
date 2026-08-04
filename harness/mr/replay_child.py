"""Run one workload in a fresh process and print its trajectory hash.

MR6 replayed in-process proves the engine is a function of its inputs *within a
process*. It cannot see anything that is fixed for the life of an interpreter:
a dict iteration order, a lazily built cache, a hash seed. PYTHONHASHSEED is the
concrete one, since the sampler keys draws on the request uid, and a salted
hash would give a different trajectory in the next session while looking
perfectly deterministic inside this one.

So MR6 also runs the same workload through this script twice, in two separate
interpreters, and compares. Invoked by harness/mr/relations.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.scheduler import Request, Scheduler  # noqa: E402


def main() -> int:
    spec = json.loads(sys.argv[1])
    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=spec.get("max_len", 256))

    scheduler = Scheduler(
        model,
        num_blocks=spec["num_blocks"],
        block_size=spec.get("block_size", 16),
    )
    for request in spec["requests"]:
        scheduler.submit(Request(**request))
    outputs = scheduler.run()

    print(json.dumps({
        "trajectory": scheduler.trajectory.hexdigest(),
        "steps": scheduler.trajectory.steps,
        "outputs": outputs,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
