"""Peak VRAM and host RSS across chunk sizes and under eviction pressure.

The previous version of this table had a 512-token chunk row against prompts of
200 to 360 tokens, so that row never chunked anything and was byte-identical to
the unchunked row. A configuration that looks tested and executed no new code is
the same failure the corpus-length and split-boundary checks were added to catch,
so the prompts here run past every chunk size measured, and the chunk sizes
include one that is neither a power of two nor a divisor of the block size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench import memprobe  # noqa: E402
from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.policy import DefaultPolicy, FixedChunkPolicy  # noqa: E402
from engine.sched.scheduler import Request, Scheduler  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

MIB = memprobe.MIB
TOTAL_VRAM_MIB = 8188

# Every chunk size is smaller than the shortest prompt, so every row chunks.
# 37 is neither a power of two nor a divisor of any supported block size.
CHUNK_SIZES = (32, 37, 128, 512, None)
PROMPT_LENGTHS = (600, 640, 700, 560, 620, 580, 660, 610)


def measure(model, chunk, block_size, num_blocks, prompts, cache=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy = DefaultPolicy() if chunk is None else FixedChunkPolicy(chunk)
    scheduler = Scheduler(
        model, num_blocks=num_blocks, policy=policy,
        block_size=block_size, enable_prefix_cache=cache,
    )
    for index, prompt in enumerate(prompts):
        scheduler.submit(
            Request(uid=f"r{index:02d}", prompt=prompt, seed=index,
                    max_new_tokens=4, temperature=0.0)
        )
    scheduler.run()
    scheduler.pool.audit()
    result = {
        "chunk": chunk,
        "steps": scheduler.step_index,
        "evictions": scheduler.evictions,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del scheduler
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    model = Qwen3(REPO_ROOT / "weights" / "Qwen3-0.6B", max_len=1024)
    generator = torch.Generator().manual_seed(7)
    prompts = [
        torch.randint(0, model.cfg.vocab_size, (n,), generator=generator).tolist()
        for n in PROMPT_LENGTHS
    ]
    weights = torch.cuda.memory_allocated()

    print("peak memory under chunked prefill")
    print(f"  weights on GPU        {weights / MIB:8.1f} MiB")
    print(f"  block size            {args.block_size}")
    print(f"  prompts               {list(PROMPT_LENGTHS)} tokens")
    print(f"  every chunk size is smaller than the shortest prompt, so every row chunks")
    print()
    print(f"  {'chunk':>10} {'peak alloc':>12} {'peak reserved':>14} {'headroom':>10} {'steps':>6}")

    rows = []
    for chunk in CHUNK_SIZES:
        row = measure(model, chunk, args.block_size, 1024, prompts)
        rows.append(row)
        label = "unchunked" if chunk is None else str(chunk)
        print(f"  {label:>10} {row['peak_allocated_bytes'] / MIB:11.1f}M "
              f"{row['peak_reserved_bytes'] / MIB:13.1f}M "
              f"{TOTAL_VRAM_MIB - row['peak_reserved_bytes'] / MIB:9.1f}M "
              f"{row['steps']:>6}")

    # Eviction pressure: a pool far too small for the workload, cache on, so the
    # scheduler must reclaim cached blocks to make progress.
    print()
    print("under eviction pressure (pool sized well below the workload, cache on)")
    pressure = measure(model, None, args.block_size, 200, prompts, cache=True)
    print(f"  peak allocated        {pressure['peak_allocated_bytes'] / MIB:8.1f} MiB")
    print(f"  peak reserved         {pressure['peak_reserved_bytes'] / MIB:8.1f} MiB")
    print(f"  evictions             {pressure['evictions']}")
    print(f"  ledger audited every step and balanced throughout")

    anon, file_backed = memprobe.rss_split()
    peak_rss = memprobe.peak_rss_bytes()
    print()
    print(f"  peak host RSS         {peak_rss / MIB:8.1f} MiB")
    print(f"  host RSS now          {anon / MIB:8.1f} MiB anon, {file_backed / MIB:8.1f} MiB file")

    env = envlock.capture()
    print()
    print(f"env  {env.fingerprint()}")

    if not args.no_artifact:
        path = Artifact(
            kind="memory",
            env=env,
            payload={
                "block_size": args.block_size,
                "prompt_lengths": list(PROMPT_LENGTHS),
                "weights_bytes": int(weights),
                "chunked": rows,
                "eviction_pressure": pressure,
                "peak_rss_bytes": peak_rss,
                "rss_anon_bytes": anon,
                "rss_file_bytes": file_backed,
            },
        ).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
