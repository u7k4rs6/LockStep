"""Black-box certification of an external engine's deterministic mode.

docs/03-security-and-access.md section 2 governs this file and is enforced, not
documented:

  * Only endpoints the developer started. `i_control_this_endpoint` has no
    default and the run refuses without it.
  * The run refuses if any hosted API key is in the environment, as a hard guard
    against a misconfigured base URL sending a campaign to a paid endpoint.
  * Rate and concurrency caps default low.
  * No workload resembling a denial-of-service pattern. Memory pressure is
    driven through the policy seam in-process, never through request floods.

The relation is the one in `certify/observable.py`, which is strictly weaker than
the internal relations. That is stated in the report rather than left for a
reader to infer.

**Boundary workloads are generated against the engine's own block size**, which
is discovered from the running server rather than assumed. SGLang's known
corruption case is `prefix_len == block_size` at 64; probing it with this
project's 16 would run the campaign and test nothing, which is the failure this
repository has caught five times already.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.observable import (  # noqa: E402
    TOP_LOGPROBS,
    Completion,
    compare,
    describe,
)
from engine import envlock  # noqa: E402
from report.artifact import Artifact, relpath  # noqa: E402

# Any of these in the environment aborts the run. Fail closed.
HOSTED_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
    "TOGETHER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "FIREWORKS_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
)

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


class RefusedToRun(SystemExit):
    """A security precondition was not met."""


def concurrency_cap(config: dict) -> int:
    """The declared cap, read rather than assumed.

    `max_concurrency` sat in certify/config.json from week 8 and nothing ever
    read it. Declared surface wider than enforced surface is this project's own
    defect class, so it is enforced here: the submission pool is sized by it, and
    a workload that would need more than the cap allows fails loudly instead of
    quietly running narrower than intended and reporting a clean result.
    """
    cap = config.get("max_concurrency")
    if not isinstance(cap, int) or cap < 1:
        raise RefusedToRun(
            "certify config must declare an integer max_concurrency of at least 1"
        )
    return cap


def guard(base_url: str, config: dict) -> None:
    """Every check from the security doc, before a single request is sent."""
    if not config.get("i_control_this_endpoint"):
        raise RefusedToRun(
            "certify config must set i_control_this_endpoint: true, deliberately "
            "and with no default. This is a speed bump against accident, not a "
            "security control, and it is documented as such."
        )

    present = [key for key in HOSTED_KEYS if os.environ.get(key)]
    if present:
        raise RefusedToRun(
            f"refusing to run with hosted API keys in the environment: "
            f"{', '.join(present)}. A misconfigured base URL would send a fuzz "
            "campaign to a paid hosted endpoint. Unset them and rerun."
        )

    host = base_url.split("//", 1)[-1].split(":")[0].split("/")[0]
    if host not in LOCAL_HOSTS:
        raise RefusedToRun(
            f"base URL host is {host!r}. This certifier runs only against "
            "endpoints on this machine that the developer started. Never a "
            "hosted API, a shared cluster, or anyone's demo deployment."
        )


def post(base_url: str, path: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(base_url: str, path: str, timeout: int = 30) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


class BatchWitness:
    """Proof that requests actually shared a batch, rather than an assumption.

    The first version of this certifier submitted every request sequentially and
    blocking, then named its cases "co-batched" and "batch 31". Nothing ever
    cohabited: each request ran at effective batch 1, and the property the whole
    project is about was the one thing the workload never produced. The cases
    passed, and the passes meant almost nothing.

    Concurrent submission alone does not fix that, because a server is free to
    serialize whatever it likes. So co-residency is *measured*: this polls the
    engine's own `vllm:num_requests_running` gauge and records the maximum it
    ever sees. A case whose witness never exceeds 1 is reported as not having
    formed a batch, and the run says so loudly instead of scoring it.

    This is the same discipline as the engine's execution counters, applied to an
    engine whose internals are not visible: assert that the path ran, using
    whatever the black box is willing to expose.
    """

    GAUGE = "vllm:num_requests_running"

    def __init__(self, base_url: str, interval: float = 0.02):
        self.base_url = base_url
        self.interval = interval
        self.max_running = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=2) as response:
                body = response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
            return
        for line in body.splitlines():
            if line.startswith(self.GAUGE):
                try:
                    value = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                self.max_running = max(self.max_running, int(value))
                self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def __enter__(self) -> "BatchWitness":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


@dataclass
class EngineInfo:
    name: str
    model: str
    block_size: int | None
    discovered_from: str


def discover(base_url: str, engine: str) -> EngineInfo:
    """Read the engine's own block size rather than assuming this project's.

    SGLang exposes `/get_server_info`; vLLM's cache block size is in
    `/v1/models` metadata on some builds and otherwise has to be supplied. When
    it cannot be discovered, that is reported and the boundary workloads are
    generated for a stated assumed value rather than silently for 16.
    """
    models = get(base_url, "/v1/models") or {}
    model = (models.get("data") or [{}])[0].get("id", "unknown")

    info = get(base_url, "/get_server_info")
    if info:
        for key in ("page_size", "block_size", "attention_page_size"):
            if key in info and isinstance(info[key], int):
                return EngineInfo(engine, model, info[key], f"/get_server_info[{key}]")
        server_args = info.get("server_args") or {}
        for key in ("page_size", "block_size"):
            if isinstance(server_args.get(key), int):
                return EngineInfo(engine, model, server_args[key],
                                  f"/get_server_info.server_args[{key}]")

    return EngineInfo(engine, model, None, "not exposed by this engine")


def complete(base_url: str, model: str, tokens: list[int], max_tokens: int) -> Completion:
    """One greedy completion, reduced to the observable."""
    body = post(base_url, "/v1/completions", {
        "model": model,
        "prompt": tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": TOP_LOGPROBS,
        "seed": 0,
    })
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}

    ids = lp.get("tokens") or []
    token_ids = [int(t) if isinstance(t, int) else hash(t) for t in ids]
    chosen = lp.get("token_logprobs") or []
    top = lp.get("top_logprobs") or []

    alternatives = []
    for entry in top:
        if isinstance(entry, dict):
            alternatives.append(sorted(
                ((hash(k) if not isinstance(k, int) else k, float(v))
                 for k, v in entry.items()),
                key=lambda pair: (-pair[1], pair[0]),
            ))
        else:
            alternatives.append([])

    return Completion(tokens=token_ids, logprobs=[
        float(x) if x is not None else None for x in chosen
    ], alternatives=alternatives)


def boundary_workloads(block_size: int, vocab: int = 100000, seed: int = 424242):
    """The cases that matter, generated against *their* block size."""
    import random

    rng = random.Random(seed)

    def toks(n):
        return [rng.randrange(1000, vocab) for _ in range(n)]

    cases = []
    for delta, label in ((-1, "block_size - 1"), (0, "block_size"), (1, "block_size + 1")):
        length = block_size + delta
        if length < 1:
            continue
        shared = toks(length)
        cases.append({
            "name": f"prefix_len == {label} ({length})",
            "requests": [shared + toks(6), shared + toks(9)],
        })

    shared = toks(block_size * 2)
    cases.append({
        "name": "zero-prefix request co-batched with a nonzero-prefix request",
        "requests": [toks(11), shared + toks(5), shared + toks(7)],
    })
    whole = toks(block_size * 3)
    cases.append({
        "name": "cache hit covering the full prompt",
        "requests": [whole, whole],
    })
    for width in (31, 32):
        base = toks(block_size)
        cases.append({
            "name": f"batch {width}, shared prefix of one block",
            "requests": [base + toks(3 + i % 5) for i in range(width)],
        })
    return cases


# How many extra requests ride alongside the case's own, per repeat. Different
# widths across repeats is what makes this a batch-composition perturbation
# rather than three identical passes.
FILLER_WIDTHS = (0, 5, 13)


def filler_requests(count: int, block_size: int, salt: int, vocab: int = 100000):
    """Cohabitants whose only job is to change the batch geometry.

    Seeded from the repeat index so a campaign is reproducible, and deliberately
    not sharing a prefix with the case's requests: a filler that collided with
    the case's prefix would change what the cache holds and confound a prefix
    boundary case with a cache-content difference.
    """
    import random

    rng = random.Random(90000 + salt * 977)
    return [
        [rng.randrange(vocab // 2, vocab) for _ in range(block_size + 3 + i)]
        for i in range(count)
    ]


def submit_concurrently(base_url, model, requests, fillers, max_tokens, cap):
    """All requests in flight at once. Returns the case's completions, in order.

    A thread pool rather than asyncio because `complete` is a blocking urllib
    call and the pool is the smaller change; the requests are I/O bound on a
    local server, so the GIL is not in the way.
    """
    everything = list(requests) + list(fillers)
    if len(everything) > cap:
        raise RefusedToRun(
            f"this case needs {len(everything)} requests in flight but "
            f"max_concurrency is {cap}. Running it narrower would form a smaller "
            "batch than the case name claims, which is the vacuous pass this "
            "certifier exists to avoid. Raise the cap deliberately or drop the case."
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(everything)) as pool:
        futures = [
            pool.submit(complete, base_url, model, tokens, max_tokens)
            for tokens in everything
        ]
        # Fillers are awaited too: leaving them running would let the next
        # observation start while the previous batch is still draining, which
        # would make the witness a measurement of the wrong thing.
        settled = [f.result() for f in futures]
    return settled[: len(requests)]


def certify(base_url: str, info: EngineInfo, block_size: int, repeats: int,
            max_tokens: int, cap: int = 48) -> list[dict]:
    """Run each boundary case under varying batch composition and compare.

    **Every request in a case is submitted concurrently.** vLLM and SGLang batch
    whatever is in flight, so concurrency is the only way a black-box client can
    make requests cohabit, and cohabitation is the entire property under test. An
    earlier version submitted sequentially and blocking, which certified
    repeat-stability at effective batch 1 and named the cases "co-batched"
    anyway.

    Concurrency alone would still be a weak probe, because three identical
    concurrent passes can produce the same batch composition each time. So each
    observation submits the case's requests **alongside a different number of
    filler requests**, which is MR1 through a black box: the target requests are
    fixed, their cohabitants are not, and a batch-invariant engine must return
    identical output regardless. The fillers are distinct from the case's
    requests and their outputs are discarded; they exist to move the batch
    geometry.

    Co-residency is measured rather than assumed, via `BatchWitness`. A case
    whose witness never records more than one request running is reported as
    not-batched, and a clean verdict on such a case is worth nothing.

    SGLang's #22819 reproduces as bursts of concurrent requests where the one at
    `prefix_len == block_size` is corrupted and the `prefix_len == 0` requests
    beside it are not, so this shape is what the upstream bug actually needs.
    """
    results = []
    for case in boundary_workloads(block_size):
        observations = []
        witnesses = []
        for repeat in range(repeats):
            fillers = filler_requests(FILLER_WIDTHS[repeat % len(FILLER_WIDTHS)],
                                      block_size, repeat)
            with BatchWitness(base_url) as witness:
                batch = submit_concurrently(
                    base_url, info.model, case["requests"], fillers, max_tokens, cap
                )
            observations.append(batch)
            witnesses.append({
                "repeat": repeat,
                "fillers": len(fillers),
                "max_concurrent_running": witness.max_running,
                "gauge_samples": witness.samples,
            })

        divergences = []
        for index in range(len(case["requests"])):
            first = observations[0][index]
            for repeat in range(1, len(observations)):
                verdict = compare(first, observations[repeat][index])
                if not verdict.identical:
                    divergences.append({
                        "request": index,
                        "repeat": repeat,
                        "detail": verdict.detail,
                        "max_logprob_delta": verdict.max_logprob_delta,
                        "first_token_divergence": verdict.first_token_divergence,
                    })

        observed_batch = max((w["max_concurrent_running"] for w in witnesses), default=0)
        gauge_seen = any(w["gauge_samples"] for w in witnesses)
        # A clean verdict on a case that never formed a batch is the vacuous pass
        # this repository has caught five times internally. It is not scored as
        # clean; it is scored as not-batched and reported separately.
        batched = observed_batch > 1
        results.append({
            "case": case["name"],
            "requests": len(case["requests"]),
            "positions": sum(c.positions() for c in observations[0]),
            "repeats": repeats,
            "filler_widths": [w["fillers"] for w in witnesses],
            "max_concurrent_running": observed_batch,
            "batch_gauge_available": gauge_seen,
            "batched": batched,
            "divergences": divergences,
            "clean": not divergences and batched,
            "vacuous": not batched,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--engine", default="sglang")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "certify" / "config.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--assume-block-size", type=int, default=None)
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    guard(args.base_url, config)

    info = discover(args.base_url, args.engine)
    block_size = info.block_size or args.assume_block_size
    assumed = info.block_size is None

    print(f"lockstep certify  {info.name}")
    print(f"  endpoint            {args.base_url}  (local, developer-started)")
    print(f"  model               {info.model}")
    print(f"  block size          {block_size}"
          f"{'  ASSUMED, not discovered' if assumed else f'  from {info.discovered_from}'}")
    print(f"  repeats per case    {args.repeats}")
    print()
    print(f"  {describe()}")
    print()

    if block_size is None:
        raise RefusedToRun(
            "block size neither discovered nor supplied. Probing boundary "
            "conditions against the wrong block size runs the campaign and tests "
            "nothing; pass --assume-block-size deliberately if it cannot be read."
        )

    results = certify(args.base_url, info, block_size, args.repeats, args.max_tokens,
                      concurrency_cap(config))

    clean = sum(1 for r in results if r["clean"])
    print(f"  {'case':<52} {'reqs':>5} {'positions':>10} {'batch':>5}  verdict")
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
    print(f"  {len(results) - len(vacuous)}/{len(results)} cases actually formed a batch"
          + ("  <- the rest tested nothing" if vacuous else ""))
    if vacuous:
        print("  a case that never cohabited cannot certify batch invariance, so "
              "these are reported rather than scored")

    env = envlock.capture()
    print(f"env  {env.fingerprint()}")
    if not args.no_artifact:
        path = Artifact(kind="certify", env=env, payload={
            "engine": info.name,
            "model": info.model,
            "block_size": block_size,
            "block_size_assumed": assumed,
            "block_size_source": info.discovered_from,
            "observable": describe(),
            "top_logprobs": TOP_LOGPROBS,
            "repeats": args.repeats,
            "results": results,
            "clean": clean,
            "total": len(results),
            "batched_cases": sum(1 for r in results if r.get("batched")),
            "max_concurrent_running": max(
                (r.get("max_concurrent_running", 0) for r in results), default=0),
            "filler_widths": list(FILLER_WIDTHS),
        }).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
