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
import json
import os
import sys
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


def certify(base_url: str, info: EngineInfo, block_size: int, repeats: int,
            max_tokens: int) -> list[dict]:
    """Run each boundary case twice and apply the observable.

    Twice through the *same* server, with the cache warmed by the first pass, is
    the hit-versus-miss comparison a black box can make: the first request
    populates whatever prefix cache the engine keeps, the second is served with
    it warm, and a deterministic mode must produce the same thing either way.
    """
    results = []
    for case in boundary_workloads(block_size):
        observations = []
        for _ in range(repeats):
            batch = [complete(base_url, info.model, tokens, max_tokens)
                     for tokens in case["requests"]]
            observations.append(batch)

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

        results.append({
            "case": case["name"],
            "requests": len(case["requests"]),
            "positions": sum(c.positions() for c in observations[0]),
            "repeats": repeats,
            "divergences": divergences,
            "clean": not divergences,
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

    results = certify(args.base_url, info, block_size, args.repeats, args.max_tokens)

    clean = sum(1 for r in results if r["clean"])
    print(f"  {'case':<52} {'reqs':>5} {'positions':>10}  verdict")
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
        }).write()
        print(f"artifact  {relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
