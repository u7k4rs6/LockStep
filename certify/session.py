"""Start an engine, certify it, tear it down, all in one process lifetime.

An earlier attempt kept the server up across separate tool invocations and lost
it every time. That was a shell problem rather than a design problem, and this is
the fix: the server is a child of this process, readiness is polled with a
timeout, the campaign runs, and teardown happens in a `finally` so a crash in the
campaign never leaves a GPU pinned by an orphan.

docs/03-security-and-access.md section 2 still applies in full, and is checked by
`certify.run.guard` before a single request is sent. The server this starts is on
127.0.0.1, started by this developer, and nothing here can be pointed at a hosted
endpoint: the URL is constructed locally rather than accepted as input.
"""

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
from certify.run import EngineInfo, certify, guard  # noqa: E402
from engine import envlock  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
PORT = 30011
BASE_URL = f"http://127.0.0.1:{PORT}"


def launch_vllm(python: Path, block_size: int, invariant: bool, log: Path):
    """vLLM's OpenAI-compatible server, batch-invariant mode optional."""
    env = dict(os.environ)
    env["PATH"] = f"{python.parent}:{env.get('PATH', '')}"
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
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    guard(BASE_URL, config)

    invariant = not args.no_invariant
    mode = "VLLM_BATCH_INVARIANT=1" if invariant else "default"
    log = Path.home() / "lockstep-extenv" / "logs" / f"vllm-certify-{int(invariant)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    print(f"lockstep certify  vLLM, {mode}")
    print(f"  endpoint            {BASE_URL}  (local, started by this process)")
    print(f"  block size          {args.block_size}  configured at launch")
    print(f"  repeats per case    {args.repeats}")
    print()
    print(f"  {describe()}")
    print()
    print(f"  starting server, up to {args.startup_timeout}s ...", flush=True)

    process, handle = launch_vllm(args.python, args.block_size, invariant, log)
    try:
        if not wait_ready(process, args.startup_timeout, log):
            tail = log.read_text().splitlines()[-6:]
            print("  server did not become ready. Last lines:")
            for line in tail:
                print(f"    {line[:150]}")
            return 1
        print("  ready", flush=True)

        info = EngineInfo(
            name=f"vLLM ({mode})", model="qwen3",
            block_size=args.block_size,
            discovered_from="configured at launch; vLLM exposes no endpoint reporting it",
        )
        results = certify(BASE_URL, info, args.block_size, args.repeats, args.max_tokens)
    finally:
        teardown(process, handle)

    clean = sum(1 for r in results if r["clean"])
    print()
    print(f"  {'boundary case':<52} {'reqs':>5} {'positions':>10}  verdict")
    print("  " + "-" * 78)
    for row in results:
        verdict = "clean" if row["clean"] else f"DIVERGED ({len(row['divergences'])})"
        print(f"  {row['case']:<52} {row['requests']:>5} {row['positions']:>10}  {verdict}")
        for d in row["divergences"][:3]:
            print(f"      request {d['request']} repeat {d['repeat']}: {d['detail']}")

    print()
    print(f"  {clean}/{len(results)} boundary cases clean at this observable")

    env = envlock.capture()
    print(f"env  {env.fingerprint()}")
    if not args.no_artifact:
        path = Artifact(kind="certify", env=env, payload={
            "engine": info.name, "mode": mode, "model": info.model,
            "block_size": args.block_size, "block_size_assumed": False,
            "block_size_source": info.discovered_from,
            "observable": describe(), "top_logprobs": TOP_LOGPROBS,
            "repeats": args.repeats, "results": results,
            "clean": clean, "total": len(results),
        }).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
