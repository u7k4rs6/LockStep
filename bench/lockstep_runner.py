"""Run one lockstep configuration in its own process and print its wall time.

The parent process previously held the model for the whole benchmark, roughly
1.7 GB resident plus whatever the KV pool and activations cached. Interleaving
then scattered lockstep measurements through the run, so external measurements
routinely started behind that allocation.

It is not a subtle effect. With `torch.cuda.empty_cache()` after each lockstep
sample, which the benchmark did call, a following vLLM sample ran about twice as
slow. Without it, vLLM failed to start at all and never recovered for the rest of
the run. Even the samples that looked clean were slower than the same
configuration measured with nothing else resident: 0.565s against 0.465s.

So every measurement now runs in its own process and the parent holds no device
memory between samples. That is the only version of isolation that survives
interleaving, and interleaving is what removes the drift bias, so the two fixes
are not independent.
"""

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
        # One untimed pass first. Every sample is now its own process, so without
        # this the timed region pays Triton's JIT compilation of every kernel on
        # first launch, and the cost lands on whichever samples happen to run
        # before the compile cache is warm. That produced a 1.54x spread on the
        # fast path and the impossible result that invariant mode measured faster
        # than the mode it constrains.
        #
        # Fixing measurement isolation introduced this, exactly as fixing drift
        # introduced the contamination isolation was written to remove. Each fix
        # was correct and each moved the bias somewhere new.
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
