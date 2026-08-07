"""Black-box certification of an external engine's deterministic mode."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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

HOSTED_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
    "TOGETHER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "FIREWORKS_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
)

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


class RefusedToRun(SystemExit):
    """A security precondition was not met."""


def concurrency_cap(config: dict) -> int:
    """The declared cap, read rather than assumed."""
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
    """Proof that requests actually shared a batch, rather than an assumption."""

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
    """Read the engine's own block size rather than assuming this project's."""
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


def _stable_id(token: str) -> int:
    """A stable integer for a token an engine returned as a string."""
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")


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
    token_ids = [int(t) if isinstance(t, int) else _stable_id(t) for t in ids]
    chosen = lp.get("token_logprobs") or []
    top = lp.get("top_logprobs") or []

    alternatives = []
    for entry in top:
        if isinstance(entry, dict):
            alternatives.append(sorted(
                ((_stable_id(k) if not isinstance(k, int) else k, float(v))
                 for k, v in entry.items()),
                key=lambda pair: (-pair[1], pair[0]),
            ))
        else:
            alternatives.append([])

    return Completion(tokens=token_ids, logprobs=[
        float(x) if x is not None else None for x in chosen
    ], alternatives=alternatives)


# Generated against the engine's own block size, discovered rather than assumed.
# SGLang's known corruption is at prefix_len == block_size with theirs at 64;
# probing it with this project's 16 would run the campaign and test nothing.
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


RELATIONS = {
    ("warm", "fixed"): (
        "baseline: neither factor varies, so a divergence here is neither "
        "property and indicts the measurement itself"
    ),
    ("warm", "varying"): (
        "batch composition alone (MR1 analog): cache is warm for every repeat, "
        "only the cohabitants change"
    ),
    ("cold", "fixed"): (
        "cache cold versus warm alone (MR4 analog): batch geometry is fixed, "
        "only cache state changes"
    ),
    ("cold", "varying"): (
        "both factors, confounded: the configuration the first concurrent run "
        "used, kept for comparability"
    ),
    ("disabled", "varying"): (
        "batch composition with prefix caching off: the cleanest single test, "
        "because the cache cannot contribute at all"
    ),
    ("disabled", "fixed"): (
        "neither factor varies and caching is off: the strictest baseline"
    ),
}


def reset_prefix_cache(base_url: str) -> bool:
    """Ask the engine to drop its prefix cache. Returns whether it worked."""
    request = urllib.request.Request(f"{base_url}/reset_prefix_cache", data=b"",
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status in (200, 204)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# Varying the cohabitants across repeats is MR1 through a black box: the target
# requests are fixed, only who shares their batch changes.
FILLER_WIDTHS = (0, 5, 13)

FIXED_FILLER_WIDTH = 13


def permute(items: list, repeat: int) -> list:
    """A deterministic, repeat-dependent permutation of the submission order."""
    if not items or repeat == 0:
        return list(items)
    cut = repeat % len(items)
    return list(items[cut:]) + list(items[:cut])


def filler_requests(count: int, block_size: int, salt: int, vocab: int = 100000):
    """Cohabitants whose only job is to change the batch geometry."""
    import random

    rng = random.Random(90000 + salt * 977)
    return [
        [rng.randrange(vocab // 2, vocab) for _ in range(block_size + 3 + i)]
        for i in range(count)
    ]


def submit_concurrently(base_url, model, requests, fillers, max_tokens, cap,
                        order_seed: int = 0):
    """All requests in flight at once. Returns the case's completions, in order."""
    everything = list(requests) + list(fillers)
    index = permute(list(range(len(everything))), order_seed)
    everything = [everything[i] for i in index]
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
        settled = [f.result() for f in futures]
    restored = [None] * len(settled)
    for position, original in enumerate(index):
        restored[original] = settled[position]
    return restored[: len(requests)]


def certify(base_url: str, info: EngineInfo, block_size: int, repeats: int,
            max_tokens: int, cap: int = 48, cache_mode: str = "cold",
            filler_mode: str = "varying", fixed_width: int = FIXED_FILLER_WIDTH,
            sequential: bool = False, order_mode: str = "stable") -> list[dict]:
    """Run each boundary case under varying batch composition and compare."""
    results = []
    for case in boundary_workloads(block_size):
        observations = []
        witnesses = []
        digests: list[str] = []

        if cache_mode == "warm":
            prime_width = (fixed_width if filler_mode == "fixed"
                           else FILLER_WIDTHS[0])
            prime_salt = 0 if order_mode in ("identical", "permuted") else 999
            reset_prefix_cache(base_url)
            submit_concurrently(base_url, info.model, case["requests"],
                                filler_requests(prime_width, block_size, prime_salt),
                                max_tokens, cap)

        for repeat in range(repeats):
            width = (fixed_width if filler_mode == "fixed"
                     else FILLER_WIDTHS[repeat % len(FILLER_WIDTHS)])
            fillers = filler_requests(
                width, block_size,
                repeat if order_mode == "stable" else 0,
            )

            cache_reset_ok = True
            if cache_mode == "cold":
                cache_reset_ok = reset_prefix_cache(base_url)

            with BatchWitness(base_url) as witness:
                if sequential:
                    batch = [complete(base_url, info.model, tokens, max_tokens)
                             for tokens in case["requests"]]
                    for tokens in fillers:
                        complete(base_url, info.model, tokens, max_tokens)
                else:
                    batch = submit_concurrently(
                        base_url, info.model, case["requests"], fillers,
                        max_tokens, cap,
                        order_seed=repeat if order_mode == "permuted" else 0,
                    )
            observations.append(batch)
            digests.append(hashlib.sha256(repr([
                (c.tokens, c.logprobs, c.alternatives) for c in batch
            ]).encode()).hexdigest())
            witnesses.append({
                "repeat": repeat,
                "fillers": len(fillers),
                "max_concurrent_running": witness.max_running,
                "gauge_samples": witness.samples,
                "cache_reset_ok": cache_reset_ok,
            })

        divergences = []
        pair_results = []
        for left in range(len(observations)):
            for right in range(left + 1, len(observations)):
                differing = 0
                for index in range(len(case["requests"])):
                    verdict = compare(observations[left][index],
                                      observations[right][index])
                    if not verdict.identical:
                        differing += 1
                        divergences.append({
                            "request": index,
                            "repeat": right,
                            "pair": [left, right],
                            "detail": verdict.detail,
                            "max_logprob_delta": verdict.max_logprob_delta,
                            "first_token_divergence": verdict.first_token_divergence,
                        })
                pair_results.append({"pair": [left, right], "requests_differing": differing})

        observed_batch = max((w["max_concurrent_running"] for w in witnesses), default=0)
        cold_enforced = all(w["cache_reset_ok"] for w in witnesses)
        gauge_seen = any(w["gauge_samples"] for w in witnesses)
        batched = observed_batch > 1
        cold_ok = cache_mode != "cold" or cold_enforced
    # A cold cell whose reset never took is not a cold cell, and a case that never
        # cohabited cannot certify batch invariance. Both score vacuous rather than
        # clean: the first version of this certifier submitted sequentially and
        # reported 7 of 7 on cases that never formed a batch.
        configured = cold_ok and (batched or sequential)
        results.append({
            "case": case["name"],
            "cache_mode": cache_mode,
            "filler_mode": filler_mode,
            "relation": RELATIONS.get((cache_mode, filler_mode), "unclassified"),
            "cold_enforced": cold_enforced,
            "requests": len(case["requests"]),
            "positions": sum(c.positions() for c in observations[0]),
            "repeats": repeats,
            "filler_widths": [w["fillers"] for w in witnesses],
            "max_concurrent_running": observed_batch,
            "batch_gauge_available": gauge_seen,
            "batched": batched,
            "divergences": divergences,
            "pairs": pair_results,
            "repeat_digests": digests,
            "later_repeats_agree": all(
                p["requests_differing"] == 0 for p in pair_results if p["pair"][0] > 0
            ),
            "first_repeat_is_odd": (
                all(p["requests_differing"] > 0 for p in pair_results if p["pair"][0] == 0)
                and all(p["requests_differing"] == 0 for p in pair_results if p["pair"][0] > 0)
            ),
            "clean": not divergences and configured,
            "vacuous": not configured,
            "sequential": sequential,
            "order_mode": order_mode,
            "fixed_width": fixed_width if filler_mode == "fixed" else None,
            "vacuous_reason": (
                "" if configured
                else ("never formed a batch" if not batched
                      else "cold mode requested but /reset_prefix_cache did not respond")
            ),
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
