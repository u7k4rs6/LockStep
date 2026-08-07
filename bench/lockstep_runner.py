"""Run one lockstep configuration in its own process and print its wall time."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    spec = json.loads(sys.argv[1])
    import torch

    from engine.model.qwen3 import Qwen3
    from engine.sched.scheduler import Request, Scheduler
    from engine.model import qwen3
    from harness.mr.ablation import torch_linear

    model = Qwen3(Path(spec["weights"]), max_len=1024)
    trace = [(list(p), n) for p, n in spec["trace"]]

    original = qwen3.linear
    if not spec["invariant"]:
        qwen3.linear = torch_linear
    try:
        warmup = Scheduler(model, num_blocks=512, block_size=16, audit=False)
        for index, (prompt, new_tokens) in enumerate(trace):
            warmup.submit(Request(uid=f"w{index:02d}", prompt=list(prompt),
                                  seed=index, max_new_tokens=new_tokens,
                                  temperature=0.0))
        warmup.run()
        del warmup
        torch.cuda.synchronize()
        started = time.perf_counter()
        scheduler = Scheduler(model, num_blocks=512, block_size=16, audit=False)
        for index, (prompt, new_tokens) in enumerate(trace):
            scheduler.submit(Request(uid=f"r{index:02d}", prompt=list(prompt),
                                     seed=index, max_new_tokens=new_tokens,
                                     temperature=0.0))
        scheduler.run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        qwen3.linear = original

    print(json.dumps({"seconds": elapsed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
