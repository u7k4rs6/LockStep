"""MR3, MR4, MR5, MR8, and the block-boundary cases."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine.audit.counters import Counters  # noqa: E402
from engine.kv import paged  # noqa: E402
from engine.audit.counters import require_fired  # noqa: E402
from engine.sched.policy import (  # noqa: E402
    DefaultPolicy,
    FixedChunkPolicy,
    NoCacheHitPolicy,
    PreemptAtStepPolicy,
    PreemptAtStepsPolicy,
)
from engine.sched.scheduler import Request, Scheduler  # noqa: E402


@dataclass
class Result:
    relation: str
    statement: str
    passed: bool
    detail: str = ""
    cases: list = field(default_factory=list)


def _request(uid: str, prompt: list[int], new_tokens: int = 6, temperature: float = 0.0):
    return Request(uid=uid, prompt=list(prompt), seed=1234, max_new_tokens=new_tokens,
                   temperature=temperature, top_p=0.95)


def blocks_for(kv_tokens: int, block_size: int) -> int:
    """Blocks holding `kv_tokens` tokens."""
    return max(4, -(-kv_tokens // block_size))


def run(model, requests, num_blocks, block_size, policy=None, cache=False, audit=True):
    scheduler = Scheduler(
        model, num_blocks=num_blocks, policy=policy or DefaultPolicy(),
        block_size=block_size, enable_prefix_cache=cache, audit=audit,
    )
    for request in requests:
        scheduler.submit(request)
    outputs = scheduler.run()
    torch.cuda.empty_cache()
    return outputs, scheduler


def mr3_preempt_resume(model, prompt, kv_tokens, block_size, new_tokens=16) -> Result:
    """Preemption at every decode step, and at depths 1, 2, and 3."""
    blocks = blocks_for(kv_tokens, block_size)
    canonical, _ = run(model, [_request("r0", prompt, new_tokens)], blocks, block_size)
    expected = canonical["r0"]

    cases = []
    passed = True
    depth_reached = {0: 0, 1: 0, 2: 0, 3: 0}
    total = Counters()

    schedules = [((step,), 1) for step in range(new_tokens + 2)]
    schedules += [((a, b), 2) for a, b in ((1, 3), (2, 5), (4, 9), (1, 8))]
    schedules += [((a, b, c), 3) for a, b, c in ((1, 3, 5), (2, 5, 9), (1, 6, 12))]

    for steps, intended_depth in schedules:
        policy = PreemptAtStepsPolicy("r0", at_steps=steps)
        outputs, scheduler = run(
            model, [_request("r0", prompt, new_tokens)], blocks, block_size, policy=policy
        )
        identical = outputs.get("r0") == expected
        actual_depth = next(
            (r.preempt_count for r in scheduler.done if r.uid == "r0"), 0
        )
        depth_reached[min(actual_depth, 3)] += 1
        total.merge(scheduler.counters)
        passed &= identical
        cases.append({
            "preempt_at_steps": list(steps),
            "intended_depth": intended_depth,
            "depth_reached": actual_depth,
            "identical": identical,
        })

    require_fired(total, "preempt_fired", "resume", what="MR3")
    if depth_reached[3] == 0:
        passed = False
    return Result(
        "MR3",
        "preempt and resume: preemption at any depth up to 3 leaves bits unchanged",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} schedules identical to "
        f"canonical; depth distribution reached "
        f"{{0: {depth_reached[0]}, 1: {depth_reached[1]}, 2: {depth_reached[2]}, "
        f"3: {depth_reached[3]}}}"
        + ("  <- DEPTH 3 NEVER REACHED" if depth_reached[3] == 0 else "")
        + f", block_size={block_size}",
        cases,
    )


def mr4_cache_cold_vs_warm(model, prompt, kv_tokens, block_size, new_tokens=6) -> Result:
    """Hit and miss produce identical bits, across chunk boundaries."""
    needed = max(3 * block_size, len(prompt))
    if len(prompt) < needed:
        prompt = (prompt * (needed // max(1, len(prompt)) + 1))[:needed]
    cold, _ = run(model, [_request("c", prompt, new_tokens)], blocks_for(kv_tokens, block_size), block_size,
                  policy=NoCacheHitPolicy(), cache=True)
    expected = cold["c"]

    writers = {
        "unchunked write": DefaultPolicy(),
        "chunked write (block_size)": FixedChunkPolicy(block_size),
        "chunked write (block_size + 1)": FixedChunkPolicy(block_size + 1),
        "chunked write (7, misaligned)": FixedChunkPolicy(7),
    }
    readers = {
        "unchunked read": lambda: DefaultPolicy(),
        "chunked read (block_size)": lambda: FixedChunkPolicy(block_size),
        "chunked read (13, misaligned)": lambda: FixedChunkPolicy(13),
    }

    cases = []
    passed = True
    mr4_counters = Counters()
    for write_label, write_policy in writers.items():
        for read_label, read_policy in readers.items():
            scheduler = Scheduler(model, num_blocks=blocks_for(kv_tokens, block_size), policy=write_policy,
                                  block_size=block_size, enable_prefix_cache=True)
            scheduler.submit(_request("writer", prompt, new_tokens))
            scheduler.run()

            scheduler.policy = read_policy()
            scheduler.submit(_request("reader", prompt, new_tokens))
            scheduler.run()
            scheduler.pool.audit()

            mr4_counters.merge(scheduler.counters)
            reader = next(r for r in scheduler.done if r.uid == "reader")
            identical = reader.generated == expected
            passed &= identical
            cases.append({
                "write": write_label,
                "read": read_label,
                "hit_tokens": reader.cache_hit_tokens,
                "identical": identical,
            })

    hits = sum(1 for c in cases if c["hit_tokens"] > 0)
    require_fired(mr4_counters, "cache_hit", "cache_insert", what="MR4")
    if hits == 0:
        passed = False
    return Result(
        "MR4",
        "cache cold versus warm: a hit produces the bits a miss would, at any chunking",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} write/read chunking pairs "
        f"identical to cold, {hits} of which actually hit the cache"
        + ("  <- NOTHING HIT THE CACHE, this configuration tested nothing" if hits == 0 else "")
        + f", prompt {len(prompt)} tokens, block_size={block_size}",
        cases,
    )


def mr5_occupancy(model, prompt, kv_tokens, block_size) -> Result:
    """A cohabitant that only changes tile occupancy leaves others' bits fixed."""
    tile = 16
    canonical, canonical_sched = run(
        model, [_request("r0", prompt)], blocks_for(kv_tokens * 4, block_size), block_size
    )
    expected = canonical["r0"]
    canonical_rows = canonical_sched.emitted_logits.get("r0") or []

    cases = []
    passed = True
    for filler_len in (1, tile - 1, tile, tile + 1, 2 * tile + 3):
        filler = [7] * filler_len
        outputs, scheduler = run(
            model,
            [_request("r0", prompt), _request("pad", filler)],
            blocks_for(kv_tokens * 4, block_size),
            block_size,
        )
        identical = outputs.get("r0") == expected
        rows_identical = True
        for a, b in zip(canonical_rows, scheduler.emitted_logits.get("r0") or []):
            if not torch.equal(a, b):
                rows_identical = False
                break
        identical = identical and rows_identical
        passed &= identical
        cases.append({
            "filler_tokens": filler_len,
            "packed_tokens": len(prompt) + filler_len,
            "identical": identical,
            "logit_bytes_identical": rows_identical,
        })

    return Result(
        "MR5",
        "padding and occupancy: a cohabitant that only shifts tile occupancy changes nothing",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} occupancy shifts left r0 "
        f"bitwise unchanged in tokens and decode-phase logit bytes, "
        f"block_size={block_size}",
        cases,
    )


def eos_finish(model, prompt, kv_tokens, block_size) -> Result:
    """A request that stops on EOS produces what an unbounded one would."""
    blocks = blocks_for(kv_tokens, block_size)
    unbounded, _ = run(model, [_request("e0", prompt, new_tokens=8)], blocks, block_size)
    baseline = unbounded["e0"]
    if len(baseline) < 3:
        return Result("EOS", "EOS finishing matches an unbounded run", False,
                      "the unbounded run emitted too few tokens to stop inside")

    stop_at = baseline[2]
    scheduler = Scheduler(model, num_blocks=blocks, block_size=block_size,
                          eos_token_ids={stop_at})
    scheduler.submit(_request("e0", prompt, new_tokens=8))
    outputs = scheduler.run()
    got = outputs["e0"]

    expected = baseline[: baseline.index(stop_at) + 1]
    identical = got == expected
    fired = scheduler.counters["finish_eos"] > 0
    require_fired(scheduler.counters, "finish_eos", what="the EOS relation")
    return Result(
        "EOS",
        "a request stopping on EOS emits the prefix an unbounded run would",
        identical and fired,
        f"stopped on token {stop_at} after {len(got)} tokens, "
        f"{'prefix matches' if identical else 'PREFIX DIFFERS'} the unbounded run "
        f"({len(baseline)} tokens), finish_eos fired {scheduler.counters['finish_eos']}x, "
        f"block_size={block_size}",
        [{"stop_token": stop_at, "emitted": len(got), "unbounded": len(baseline),
          "identical": identical, "finish_eos": scheduler.counters["finish_eos"]}],
    )


def mr8_tiebreak_under_permutation(model, prompts, kv_tokens, block_size) -> Result:
    """Argmax is stable under logit-preserving permutations of the batch."""
    requests = [_request(f"r{i:02d}", p) for i, p in enumerate(prompts)]
    baseline, _ = run(model, requests, blocks_for(kv_tokens, block_size), block_size)

    orders = {
        "reversed": list(reversed(range(len(prompts)))),
        "rotated by one": list(range(1, len(prompts))) + [0],
        "interleaved": [i for i in range(len(prompts)) if i % 2 == 0]
        + [i for i in range(len(prompts)) if i % 2 == 1],
    }

    cases = []
    passed = True
    for label, order in orders.items():
        permuted = [_request(f"r{i:02d}", prompts[i]) for i in order]
        outputs, _ = run(model, permuted, blocks_for(kv_tokens, block_size), block_size)
        moved = [uid for uid in baseline if baseline[uid] != outputs.get(uid)]
        passed &= not moved
        cases.append({"permutation": label, "moved": moved, "identical": not moved})

    return Result(
        "MR8",
        "temperature-0 tie-break: argmax is stable under batch permutation",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} permutations left every "
        f"request bitwise unchanged, block_size={block_size}",
        cases,
    )


def boundary_cases(model, kv_tokens, block_size, vocab, seed=555) -> Result:
    """The cases the upstream bug lives at, hit on purpose."""
    generator = torch.Generator().manual_seed(seed)

    def tokens(count):
        return torch.randint(0, vocab, (count,), generator=generator).tolist()

    cases = []
    passed = True
    boundary_counters = Counters()

    def record(label, ok, detail=""):
        nonlocal passed
        passed &= ok
        cases.append({"case": label, "ok": ok, "detail": detail})

    for delta, suffix in ((-1, " - 1"), (0, ""), (1, " + 1")):
        prefix_len = block_size + delta
        if prefix_len < 1:
            continue
        shared = tokens(prefix_len)
        first = shared + tokens(5)
        second = shared + tokens(7)

        cold, _ = run(model, [_request("a", second)], blocks_for(kv_tokens, block_size), block_size,
                      policy=NoCacheHitPolicy(), cache=True)
        scheduler = Scheduler(model, num_blocks=blocks_for(kv_tokens, block_size), block_size=block_size,
                              enable_prefix_cache=True)
        scheduler.submit(_request("warm0", first))
        scheduler.run()
        scheduler.submit(_request("a", second))
        scheduler.run()
        scheduler.pool.audit()
        boundary_counters.merge(scheduler.counters)
        got = next(r for r in scheduler.done if r.uid == "a")
        expected_hit = (prefix_len // block_size) * block_size
        record(
            f"prefix_len == block_size{suffix} ({prefix_len})",
            got.generated == cold["a"] and got.cache_hit_tokens == expected_hit,
            f"hit {got.cache_hit_tokens} tokens, expected {expected_hit}",
        )

    shared = tokens(block_size * 2)
    scheduler = Scheduler(model, num_blocks=blocks_for(kv_tokens * 2, block_size), block_size=block_size,
                          enable_prefix_cache=True)
    scheduler.submit(_request("seed", shared + tokens(4)))
    scheduler.run()
    cold_zero, _ = run(model, [_request("zero", tokens(9))], blocks_for(kv_tokens, block_size), block_size)
    zero_prompt = list(scheduler.done[0].prompt[:0]) + tokens(9)
    cold_z, _ = run(model, [_request("z", zero_prompt)], blocks_for(kv_tokens, block_size), block_size)
    scheduler.submit(_request("z", zero_prompt))
    scheduler.submit(_request("w", shared + tokens(6)))
    scheduler.run()
    scheduler.pool.audit()
    z = next(r for r in scheduler.done if r.uid == "z")
    w = next(r for r in scheduler.done if r.uid == "w")
    record(
        "prefix_len == 0 batched with a nonzero-prefix request",
        z.generated == cold_z["z"] and z.cache_hit_tokens == 0 and w.cache_hit_tokens > 0,
        f"zero-prefix hit {z.cache_hit_tokens}, cohabitant hit {w.cache_hit_tokens}",
    )

    whole = tokens(block_size * 3)
    cold_whole, _ = run(model, [_request("f", whole)], blocks_for(kv_tokens, block_size), block_size,
                        policy=NoCacheHitPolicy(), cache=True)
    scheduler = Scheduler(model, num_blocks=blocks_for(kv_tokens * 2, block_size), block_size=block_size,
                          enable_prefix_cache=True)
    scheduler.submit(_request("prime", whole))
    scheduler.run()
    scheduler.submit(_request("f", whole))
    scheduler.run()
    scheduler.pool.audit()
    full = next(r for r in scheduler.done if r.uid == "f")
    expected_full_hit = len(whole) - block_size
    record(
        "cache hit covering the whole prompt",
        full.generated == cold_whole["f"] and full.cache_hit_tokens == expected_full_hit,
        f"hit {full.cache_hit_tokens}/{len(whole)} tokens, last block recomputed "
        f"so the request has logits to sample",
    )

    tight = tokens(block_size * 2)
    scheduler = Scheduler(model, num_blocks=blocks_for(block_size * 6, block_size), block_size=block_size,
                          enable_prefix_cache=True)
    ok = True
    try:
        for index in range(4):
            scheduler.submit(_request(f"e{index}", tokens(block_size * 2), new_tokens=2))
            scheduler.run()
            scheduler.pool.audit()
        boundary_counters.merge(scheduler.counters)
    except paged.OutOfBlocks:
        ok = False
    record(
        "eviction under pressure, ledger audited every step",
        ok and scheduler.evictions > 0,
        f"{scheduler.evictions} evictions, ledger balanced throughout",
    )

    require_fired(boundary_counters, "cache_hit", "eviction_taken",
                  what="the boundary cases")
    hit_cases = [c for c in cases if "hit" in c["detail"] and "hit 0 " not in c["detail"]]
    if not hit_cases:
        passed = False
    return Result(
        "BOUNDARY",
        "block-boundary cases hit deliberately, not left to the fuzzer",
        passed,
        f"{sum(1 for c in cases if c['ok'])}/{len(cases)} cases pass, "
        f"{len(hit_cases)} of which registered a real cache hit"
        + ("  <- NO CASE HIT THE CACHE" if not hit_cases else "")
        + f", block_size={block_size}",
        cases,
    )
