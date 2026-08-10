"""Start an engine, certify it, tear it down, all in one process lifetime."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.observable import TOP_LOGPROBS, describe  # noqa: E402
from certify import subject as subject_env  # noqa: E402
from certify.run import (  # noqa: E402
    FILLER_WIDTHS, RELATIONS, EngineInfo, certify, concurrency_cap, guard,
)
from engine import envlock  # noqa: E402
from report.artifact import require_clean_tree, Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
PORT = 30011
BASE_URL = f"http://127.0.0.1:{PORT}"


def launch_vllm(python: Path, block_size: int, invariant: bool, log: Path,
                prefix_caching: bool = True, dev_endpoints: bool = False,
                max_num_batched_tokens: int | None = None,
                max_num_seqs: int | None = None,
                rmsnorm_trace: Path | None = None):
    """vLLM's OpenAI-compatible server, batch-invariant mode optional."""
    env = dict(os.environ)
    env["PATH"] = f"{python.parent}:{env.get('PATH', '')}"
    if rmsnorm_trace is not None:
        # Read by the LockStep patch in the subject venv's batch_invariant.py.
        # Without the patch installed this is inert, which is why the artifact
        # records whether the trace actually appeared rather than assuming it.
        env["LOCKSTEP_RMSNORM_TRACE"] = str(rmsnorm_trace)
    if dev_endpoints:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if invariant:
        env["VLLM_BATCH_INVARIANT"] = "1"
    else:
        env.pop("VLLM_BATCH_INVARIANT", None)

    command = [
        str(python), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(WEIGHTS),
        "--served-model-name", "qwen3",
        "--dtype", "float16",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--gpu-memory-utilization", "0.55",
        "--max-model-len", "1024",
        "--block-size", str(block_size),
        "--enforce-eager",
    ]
    if max_num_batched_tokens is not None:
        command += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    if max_num_seqs is not None:
        command += ["--max-num-seqs", str(max_num_seqs)]
    if not prefix_caching:
        command.append("--no-enable-prefix-caching")
    handle = log.open("w")
    return subprocess.Popen(
        command, env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    ), handle


def wait_ready(process, timeout: int, log: Path) -> bool:
    """Poll the health endpoint until the server answers or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=4) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(4)
    return False


def teardown(process, handle) -> None:
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path,
                        default=Path.home() / "lockstep-extenv" / "vllmdet" / "bin" / "python")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Configured at launch; vLLM exposes no endpoint that reports it.")
    parser.add_argument("--no-invariant", action="store_true",
                        help="Certify the default mode instead of the deterministic one.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--startup-timeout", type=int, default=420)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "certify" / "config.json")
    parser.add_argument("--cache-mode", choices=("cold", "warm", "disabled"),
                        default="cold",
                        help="cold resets the prefix cache between repeats, warm "
                             "primes once and never resets, disabled turns prefix "
                             "caching off at the server")
    parser.add_argument("--filler-mode", choices=("fixed", "varying"),
                        default="varying",
                        help="fixed holds batch geometry constant across repeats; "
                             "varying changes the cohabitant count")
    parser.add_argument("--fixed-width", type=int, default=13,
                        help="cohabitant count when --filler-mode fixed; the "
                             "rung of the concurrency ladder")
    parser.add_argument("--sequential", action="store_true",
                        help="submit one request at a time; the negative control "
                             "that reproduces effective batch 1 on this code path")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="D1: raise above the whole workload so no prompt is "
                             "chunked, isolating chunk invariance from batch "
                             "invariance")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="D3: scheduler width, varied at fixed request count "
                             "to separate scheduler geometry from batch size")
    parser.add_argument("--order-mode", choices=("stable", "identical", "permuted"),
                        default="stable",
                        help="permuted keeps the request multiset byte-identical "
                             "across repeats and changes only submission order, "
                             "which is the perturbation vLLM's guarantee names")
    parser.add_argument("--label", default="",
                        help="a name for this cell, recorded in the artifact")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="produce a claim artifact from an uncommitted "
                             "tree; recorded in the artifact when used")
    parser.add_argument("--rmsnorm-trace", type=Path, default=None,
                        help="record num_tokens at every RMSNorm launch in the "
                             "server's process. Requires the LockStep patch in "
                             "the subject venv's batch_invariant.py; inert "
                             "without it.")
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    provenance = require_clean_tree(args.allow_dirty)

    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    guard(BASE_URL, config)

    invariant = not args.no_invariant
    mode = "VLLM_BATCH_INVARIANT=1" if invariant else "default"
    log = (Path.home() / "lockstep-extenv" / "logs" /
           f"vllm-certify-{int(invariant)}-{args.cache_mode}-{args.filler_mode}"
           f"-w{args.fixed_width}{'-seq' if args.sequential else ''}"
           f"{f'-mnbt{args.max_num_batched_tokens}' if args.max_num_batched_tokens else ''}"
           f"{f'-mns{args.max_num_seqs}' if args.max_num_seqs else ''}"
           f"{('-' + args.label.replace(' ', '_')) if args.label else ''}.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    print(f"lockstep certify  vLLM, {mode}")
    print(f"  endpoint            {BASE_URL}  (local, started by this process)")
    print(f"  block size          {args.block_size}  configured at launch")
    relation = RELATIONS.get((args.cache_mode, args.filler_mode), "unclassified")
    print(f"  cell                cache={args.cache_mode} filler={args.filler_mode}"
          + (f"  [{args.label}]" if args.label else ""))
    print(f"  relation under test {relation}")
    print(f"  repeats per case    {args.repeats}, filler widths "
          f"{list(FILLER_WIDTHS) if args.filler_mode == 'varying' else f'fixed at {args.fixed_width}'}")
    print(f"  submission          "
          + ("SEQUENTIAL, one request in flight: the negative control"
             if args.sequential
             else "concurrent; co-residency measured from vllm:num_requests_running"))
    print()
    print(f"  {describe()}")
    print()
    print(f"  starting server, up to {args.startup_timeout}s ...", flush=True)

    process, handle = launch_vllm(args.python, args.block_size, invariant, log,
                                  prefix_caching=args.cache_mode != "disabled",
                                  dev_endpoints=args.cache_mode in ("cold", "warm"),
                                  max_num_batched_tokens=args.max_num_batched_tokens,
                                  max_num_seqs=args.max_num_seqs,
                                  rmsnorm_trace=args.rmsnorm_trace)
    try:
        if not wait_ready(process, args.startup_timeout, log):
            tail = log.read_text().splitlines()[-6:]
            print("  server did not become ready. Last lines:")
            for line in tail:
                print(f"    {line[:150]}")
            return 1
        print("  ready", flush=True)

        subject = subject_env.capture(args.python, log, f"vLLM ({mode})")
        print(f"  subject             {subject['engine_version']}, "
              f"torch {subject['torch_version']}, triton {subject['triton_version']}, "
              f"cu{subject['cuda_version']}")
        print("                      read from the server's startup output and its "
              "own interpreter, not this process")

        info = EngineInfo(
            name=f"vLLM ({mode})", model="qwen3",
            block_size=args.block_size,
            discovered_from="configured at launch; vLLM exposes no endpoint reporting it",
        )
        results = certify(BASE_URL, info, args.block_size, args.repeats,
                          args.max_tokens, concurrency_cap(config),
                          cache_mode=args.cache_mode, filler_mode=args.filler_mode,
                          fixed_width=args.fixed_width, sequential=args.sequential,
                          order_mode=args.order_mode)
    finally:
        teardown(process, handle)

    clean = sum(1 for r in results if r["clean"])
    print()
    print(f"  {'boundary case':<52} {'reqs':>5} {'positions':>10} {'batch':>5}  verdict")
    print("  " + "-" * 84)
    for row in results:
        if row.get("vacuous"):
            verdict = "NOT BATCHED"
        elif row["clean"]:
            verdict = "clean"
        else:
            verdict = f"DIVERGED ({len(row['divergences'])})"
        print(f"  {row['case']:<52} {row['requests']:>5} {row['positions']:>10} "
              f"{row.get('max_concurrent_running', 0):>5}  {verdict}")
        for d in row["divergences"][:3]:
            print(f"      request {d['request']} repeat {d['repeat']}: {d['detail']}")
        if row.get("vacuous"):
            print("      never observed more than one request running; this case "
                  "formed no batch and its verdict is worth nothing")

    vacuous = [r for r in results if r.get("vacuous")]
    print()
    print(f"  {clean}/{len(results)} boundary cases clean at this observable")
    formed = sum(1 for r in results if r.get("max_concurrent_running", 0) > 1)
    print(f"  {formed}/{len(results)} cases actually formed a batch"
          + ("  <- the rest tested nothing" if vacuous else "")
          + ("  (sequential control: forming no batch is the point)"
             if formed == 0 and not vacuous else ""))
    if vacuous:
        print("  a case that never cohabited cannot certify batch invariance, so "
              "these are reported rather than scored")

    env = envlock.capture()
    print(f"env  {env.fingerprint()}")
    if not args.no_artifact:
        path = Artifact(kind="certify", harness=env, subject=subject, payload={
            "engine": info.name, "mode": mode, "model": info.model,
            "block_size": args.block_size, "block_size_assumed": False,
            "block_size_source": info.discovered_from,
            "observable": describe(), "top_logprobs": TOP_LOGPROBS,
            "repeats": args.repeats, "results": results,
            "clean": clean, "total": len(results),
            "batched_cases": sum(1 for r in results if r.get("batched")),
            "cache_mode": args.cache_mode, "filler_mode": args.filler_mode,
            "cell_label": args.label, "relation": relation,
            "fixed_width": args.fixed_width, "sequential": args.sequential,
            "order_mode": args.order_mode,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "rmsnorm_trace": (str(args.rmsnorm_trace.name)
                              if args.rmsnorm_trace else None),
            "server_log": log.name,
            "max_concurrent_running": max(
                (r.get("max_concurrent_running", 0) for r in results), default=0),
            "filler_widths": list(FILLER_WIDTHS),
        }).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
