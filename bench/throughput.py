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
import random
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
from report.artifact import require_clean_tree, Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
RUNS = 5


def committed_trace(vocab: int, seed: int = 20260805) -> list[tuple[list[int], int]]:
    """The one workload every configuration runs. Committed, not generated live.

    **Four of the eight prompts cross the 512-token attention split boundary.**
    The first version of this trace topped out at 192 prompt tokens against a
    512-token model length, so no request ever reached a second split and the
    benchmark for a project about split-combine invariance never executed the
    split-combine fold. The cost being measured was therefore the cost of
    everything except the mechanism the engine exists to constrain.

    Lengths are a fixed list rather than a draw, so the shape is inspectable
    without running anything, and the token contents come from a seeded
    generator so the trace is reproducible from this file alone.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = [64, 544, 96, 600, 80, 520, 112, 700]
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


def time_external(python: Path, engine: str, deterministic: bool, trace,
                  cuda_graphs: bool = False) -> float | None:
    """Run one external configuration in its own interpreter."""
    script = REPO_ROOT / "bench" / "external_runner.py"
    payload = json.dumps({
        "engine": engine,
        "deterministic": deterministic,
        "cuda_graphs": cuda_graphs,
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


def gpu_state() -> dict:
    """Clock, temperature and power at the moment of a measurement.

    Recorded per sample rather than per run. Drift was previously reconstructed
    after the fact from a ratio moving 1.75x between two runs, which only worked
    because the shift was large; a subtler one would have been invisible.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        sm, temp, power, util = (x.strip() for x in out.stdout.strip().split(","))
        return {"sm_mhz": int(float(sm)), "temp_c": int(float(temp)),
                "power_w": float(power), "util_pct": int(float(util))}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-python", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="produce a claim artifact from an uncommitted "
                             "tree; recorded in the artifact when used")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty)

    model = Qwen3(WEIGHTS, max_len=1024)
    trace = committed_trace(model.cfg.vocab_size)
    total_tokens = sum(len(p) + n for p, n in trace)

    print("cost of determinism")
    crossing = sum(1 for p, n in trace if len(p) + n > 512)
    print(f"  trace              {len(trace)} requests, {total_tokens} tokens, committed")
    print(f"                     {crossing} of {len(trace)} cross the 512-token "
          f"attention split boundary")
    print(f"  runs               median of {args.runs}")
    print("  ratios only; absolute tokens per second is not a claim this project makes")
    print()

    # Every measurement as an independent unit, then shuffled, so configurations
    # are interleaved rather than blocked. Blocking by configuration was the
    # earlier design and it maps any drift over the run straight onto the
    # A-versus-B ratio: all of A ran while the machine was cool and all of B
    # after it was not. Two runs of identical code then disagreed by 1.75x on
    # every eager configuration while the graphed ones held, which was enough to
    # refute a published claim. Interleaving makes drift hit both arms roughly
    # equally, so a cross-mode ratio is a measurement rather than a caveat.
    units: list[tuple[str, object]] = []
    for label, invariant in (("lockstep, fast mode", False),
                             ("lockstep, invariant", True)):
        for _ in range(args.runs):
            units.append((label, ("lockstep", invariant)))

    external_specs = [
        ("vLLM default, eager", "vllm", False, False),
        ("vLLM VLLM_BATCH_INVARIANT=1, eager", "vllm", True, False),
        ("vLLM default, cudagraphs", "vllm", False, True),
        ("vLLM VLLM_BATCH_INVARIANT=1, cudagraphs", "vllm", True, True),
        ("SGLang deterministic", "sglang", True, False),
    ]
    if args.external_python is not None and args.external_python.exists():
        for label, engine, deterministic, graphs in external_specs:
            for _ in range(args.runs):
                units.append((label, ("external", engine, deterministic, graphs)))

    # Seeded, so the interleaving is a property of the committed code rather than
    # of the day it ran, and a rerun executes the same order.
    random.Random(20260806).shuffle(units)

    samples: dict[str, list[float]] = {}
    observed: dict[str, list[dict]] = {}
    print(f"  {len(units)} measurements, interleaved and shuffled", flush=True)
    for index, (label, spec) in enumerate(units, start=1):
        before = gpu_state()
        if spec[0] == "lockstep":
            seconds = time_lockstep(model, trace, spec[1])
            torch.cuda.empty_cache()
        else:
            seconds = time_external(args.external_python, spec[1], spec[2],
                                    trace, spec[3])
        if seconds is None:
            continue
        samples.setdefault(label, []).append(seconds)
        observed.setdefault(label, []).append(before)
        print(f"\r    {index}/{len(units)}  {label[:38]:<38} {seconds:6.3f}s "
              f"sm={before.get('sm_mhz', 0)}MHz {before.get('temp_c', 0)}C   ",
              end="", flush=True)
    print()

    rows: list[dict] = []
    for label, _engine, _det, _graphs in [("lockstep, fast mode", None, None, None),
                                          ("lockstep, invariant", None, None, None)] + \
                                         [(spec[0], spec[1], spec[2], spec[3])
                                          for spec in external_specs]:
        got = samples.get(label)
        if not got:
            rows.append({"config": label, "seconds": None, "measured": False,
                         "why": "no external environment supplied"
                                if label.startswith(("vLLM", "SGLang"))
                                else "not measured"})
            continue
        rows.append({
            "config": label, "seconds": statistics.median(got), "samples": got,
            "measured": True,
            "spread_ratio": max(got) / min(got),
            "gpu_state": observed.get(label, []),
        })

    # Both graph modes for vLLM. Eager against eager is the like-for-like
    # comparison, since this engine has no graphs; graphed is what vLLM actually
    # runs in production and is the number a reader deciding between the two
    # would use. Publishing only the first flatters this engine by degrading the
    # comparator, and publishing only the second hides that the gap is partly a
    # feature this engine simply does not have.
    by = {r["config"]: r["seconds"] for r in rows if r["measured"]}
    baseline = by["lockstep, fast mode"]

    print(f"  {'configuration':<40} {'vs fast':>8}   {'sample spread':>13}")
    print("  " + "-" * 68)
    for row in rows:
        if not row["measured"]:
            print(f"  {row['config']:<32} {'not measured':>27}   {row.get('why', '')}")
            continue
        spread = row.get("spread_ratio", 1.0)
        flag = "  <- unstable" if spread > 1.25 else ""
        print(f"  {row['config']:<40} {row['seconds'] / baseline:>8.2f}x"
              f"   spread {spread:4.2f}x{flag}")

    # Absolute standing is the headline. The within-engine cost of determinism
    # is deliberately NOT printed as a side-by-side against vLLM's: this engine's
    # fast path is already constrained in ways vLLM's is not, so the two ratios
    # measure different things and pairing them implies a comparison that does
    # not hold.
    lock = by["lockstep, invariant"] / by["lockstep, fast mode"]
    derived = {"lockstep_invariant_over_fast": lock}
    print()
    print("  headline: standing against the engines being certified")
    for other in ("vLLM default, eager", "vLLM VLLM_BATCH_INVARIANT=1, eager",
                  "vLLM default, cudagraphs", "vLLM VLLM_BATCH_INVARIANT=1, cudagraphs",
                  "SGLang deterministic"):
        if other in by:
            ratio = by["lockstep, invariant"] / by[other]
            print(f"    lockstep invariant is {ratio:5.1f}x the wall time of {other}")
            derived[f"lockstep_invariant_over_{other}"] = ratio
    for other in ("vLLM default, eager",):
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
                                 "interleaved": True,
                                 "measurement_order_seed": 20260806,
                                 "provenance": provenance,
                                 "rows": rows, "derived": derived}).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
