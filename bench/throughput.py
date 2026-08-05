"""Cost of determinism, as ratios.

PRD headline 3: "Invariant mode versus this engine's own fast mode, then as a
fraction of vLLM `VLLM_BATCH_INVARIANT=1` and SGLang deterministic mode. Same
GPU, same model, same committed trace, median of 5 runs. Comparing against their
deterministic modes rather than only vanilla is what makes the number
undismissable."

Absolute tokens per second is not published. This engine has no CUDA graphs and
makes no attempt to be fast; an absolute number invites the dismissal the PRD's
risk table names, and the ratio is the quantity a reader actually needs.

The workload is committed: `committed_trace` in this file holds it, seeded and
fixed, and every configuration runs the same one.

External engines run out of a separate virtual environment, because vLLM pins
torch versions that would fight the locked environment this project's claims are
scoped to. `--external-python` points at it. If it is absent, the external rows
are reported as not measured rather than estimated or omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from engine.sched.scheduler import Request, Scheduler  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
RUNS = 5


def committed_trace(vocab: int, seed: int = 20260805) -> list[tuple[list[int], int]]:
    """The one workload every configuration runs. Committed, not generated live."""
    generator = torch.Generator().manual_seed(seed)
    lengths = [64, 128, 96, 192, 80, 160, 112, 144]
    return [
        (torch.randint(0, vocab, (length,), generator=generator).tolist(), 32)
        for length in lengths
    ]


def time_lockstep(model, trace, invariant: bool) -> float:
    """Seconds for the whole trace. `invariant=False` is the fast path.

    The fast path differs from the invariant path in exactly one way: it lets
    torch pick the GEMM, which is what every non-invariant engine does and what
    the ablation showed turns I1 red. Everything else, including the attention
    kernel and the scheduler, is identical, so the ratio isolates the cost of the
    constraint rather than the cost of two different engines.
    """
    from engine.model import qwen3
    from harness.mr.ablation import torch_linear

    original = qwen3.linear
    if not invariant:
        qwen3.linear = torch_linear
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        scheduler = Scheduler(model, num_blocks=512, block_size=16, audit=False)
        for index, (prompt, new_tokens) in enumerate(trace):
            scheduler.submit(
                Request(uid=f"r{index:02d}", prompt=list(prompt),
                        seed=index, max_new_tokens=new_tokens, temperature=0.0)
            )
        scheduler.run()
        torch.cuda.synchronize()
        return time.perf_counter() - started
    finally:
        qwen3.linear = original


def time_external(python: Path, engine: str, deterministic: bool, trace) -> float | None:
    """Run one external configuration in its own interpreter."""
    script = REPO_ROOT / "bench" / "external_runner.py"
    payload = json.dumps({
        "engine": engine,
        "deterministic": deterministic,
        "weights": str(WEIGHTS),
        "trace": [[list(p), n] for p, n in trace],
    })
    # vLLM shells out to ninja when it compiles its custom ops, and a venv-local
    # ninja is not on PATH by default, so the engine core dies at startup with a
    # FileNotFoundError that reads like an unsupported GPU.
    env = dict(os.environ, PATH=f"{python.parent}:{os.environ.get('PATH', '')}")
    try:
        out = subprocess.run(
            [str(python), str(script), payload],
            capture_output=True, text=True, timeout=1800, check=True, env=env,
        )
        return float(json.loads(out.stdout.strip().splitlines()[-1])["seconds"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError, KeyError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-python", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    model = Qwen3(WEIGHTS, max_len=512)
    trace = committed_trace(model.cfg.vocab_size)
    total_tokens = sum(len(p) + n for p, n in trace)

    print("cost of determinism")
    print(f"  trace              {len(trace)} requests, {total_tokens} tokens, committed")
    print(f"  runs               median of {args.runs}")
    print("  ratios only; absolute tokens per second is not a claim this project makes")
    print()

    rows: list[dict] = []

    for label, invariant in (("lockstep, fast mode", False), ("lockstep, invariant", True)):
        samples = []
        for _ in range(args.runs):
            samples.append(time_lockstep(model, trace, invariant))
            torch.cuda.empty_cache()
        rows.append({"config": label, "seconds": statistics.median(samples),
                     "samples": samples, "measured": True})

    external = [
        ("vLLM default", "vllm", False),
        ("vLLM VLLM_BATCH_INVARIANT=1", "vllm", True),
        ("SGLang deterministic", "sglang", True),
    ]
    for label, engine, deterministic in external:
        if args.external_python is None or not args.external_python.exists():
            rows.append({"config": label, "seconds": None, "measured": False,
                         "why": "no external environment supplied"})
            continue
        samples = [
            s for s in (time_external(args.external_python, engine, deterministic, trace)
                        for _ in range(args.runs))
            if s is not None
        ]
        rows.append({
            "config": label,
            "seconds": statistics.median(samples) if samples else None,
            "samples": samples,
            "measured": bool(samples),
            "why": "" if samples else "engine not present in the external environment",
        })

    by = {r["config"]: r["seconds"] for r in rows if r["measured"]}
    baseline = by["lockstep, fast mode"]

    print(f"  {'configuration':<32} {'wall time vs lockstep fast':>27}")
    print("  " + "-" * 61)
    for row in rows:
        if not row["measured"]:
            print(f"  {row['config']:<32} {'not measured':>27}   {row.get('why', '')}")
            continue
        print(f"  {row['config']:<32} {row['seconds'] / baseline:>26.2f}x")

    # Absolute standing is the headline. The within-engine cost of determinism
    # is deliberately NOT printed as a side-by-side against vLLM's: this engine's
    # fast path is already constrained in ways vLLM's is not, so the two ratios
    # measure different things and pairing them implies a comparison that does
    # not hold.
    lock = by["lockstep, invariant"] / by["lockstep, fast mode"]
    derived = {"lockstep_invariant_over_fast": lock}
    print()
    print("  headline: standing against the engines being certified")
    for other in ("vLLM default", "vLLM VLLM_BATCH_INVARIANT=1", "SGLang deterministic"):
        if other in by:
            ratio = by["lockstep, invariant"] / by[other]
            print(f"    lockstep invariant is {ratio:5.1f}x the wall time of {other}")
            derived[f"lockstep_invariant_over_{other}"] = ratio
    for other in ("vLLM default",):
        if other in by:
            print(f"    lockstep invariant runs at "
                  f"{100 * by[other] / by['lockstep, invariant']:.0f} percent of "
                  f"{other} throughput")

    print()
    print("  separately, and not comparable to any figure above:")
    print(f"    lockstep invariant over its own fast path   {lock:.2f}x")
    print("    That measures the GEMM constraint alone. The fast path differs in")
    print("    exactly one way, letting torch pick the GEMM; attention and the")
    print("    scheduler are identical in both, and it is already constrained in")
    print("    ways an unconstrained engine's fast path is not.")

    env = envlock.capture()
    print()
    print(f"env  {env.fingerprint()}")
    if not args.no_artifact:
        path = Artifact(kind="throughput", env=env,
                        payload={"trace_tokens": total_tokens, "runs": args.runs,
                                 "rows": rows, "derived": derived}).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
