"""MR3, MR4, MR5, MR8, and the block-boundary cases.

Week 4's risk is state rather than numerics. Chunk invariance is already proven
at the tensor level, so preemption and caching are sound *if* the bookkeeping
around them is. The bookkeeping is refcounts, eviction, and the prefix cache
interacting, which is where the upstream divergence this project hunts actually
lives: SGLang's open case is a scheduler-layer boundary condition at
`prefix_len == block_size`, not a kernel bug.

So the boundary cases below are hit deliberately at every supported block size
rather than left for the fuzzer to stumble into. Finding your own engine's
version of a known upstream bug by accident, later, would be the wrong order.
"""

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
    """Blocks holding `kv_tokens` tokens.

    Relations are sized in tokens, not blocks. A fixed block count would make a
    sweep over block sizes also a sweep over pool sizes, so block_size=128 would
    allocate eight times the KV of block_size=16 and run out of device memory
    while looking like a determinism failure.
    """
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


# ---- MR3: preempt and resume ------------------------------------------------


def mr3_preempt_resume(model, prompt, kv_tokens, block_size, new_tokens=16) -> Result:
    """Preemption at every decode step, and at depths 1, 2, and 3.

    Swept, not sampled, in two dimensions. Depth 1 walks the preemption point
    across every step. Depths 2 and 3 preempt the same request repeatedly, which
    is where allocator state gets interesting: a block freed by the first
    preemption can be handed to another request and then be needed again by the
    second. Architecture doc 8.2 names preemption depth up to 3 as a coverage
    dimension for exactly that reason.
    """
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


# ---- MR4: cache cold versus warm, crossed with chunking ---------------------


def mr4_cache_cold_vs_warm(model, prompt, kv_tokens, block_size, new_tokens=6) -> Result:
    """Hit and miss produce identical bits, across chunk boundaries.

    Cold versus warm alone would not test the thing that can go wrong. Week 3
    proved chunk invariance in the compute path; what remains is whether the
    cache can launder a *different* chunk boundary into a stored result. So the
    cache is written by one chunking and read by another, in both directions.

    The prompt is sized to at least three whole blocks. Only whole blocks are
    cacheable, and a full-prompt hit deliberately gives the last block back, so a
    prompt shorter than two blocks can never register a hit: at block_size 128 a
    96-token prompt produced twelve green cells and zero cache hits, which is a
    configuration that looks tested and executed no new code. `hits == 0` is
    therefore a failure of the relation, not a pass.
    """
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

            # Same pool and cache, second request over the same prompt.
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


# ---- MR5: padding and occupancy ---------------------------------------------


def mr5_occupancy(model, prompt, kv_tokens, block_size) -> Result:
    """A cohabitant that only changes tile occupancy leaves others' bits fixed.

    Distinct from MR1: MR1 varies how many requests are resident, MR5 holds the
    count fixed and varies only how the packed token count lands against the
    GEMM tile. A kernel that padded to a tile and read the padding would pass
    MR1 and fail this.
    """
    tile = 16  # BLOCK_M in the pinned registry
    canonical, _ = run(model, [_request("r0", prompt)], blocks_for(kv_tokens * 4, block_size), block_size)
    expected = canonical["r0"]

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
        passed &= identical
        cases.append({
            "filler_tokens": filler_len,
            "packed_tokens": len(prompt) + filler_len,
            "identical": identical,
        })

    return Result(
        "MR5",
        "padding and occupancy: a cohabitant that only shifts tile occupancy changes nothing",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} occupancy shifts left r0 "
        f"bitwise unchanged, block_size={block_size}",
        cases,
    )


# ---- MR8: temperature-0 tie-break -------------------------------------------


def mr8_tiebreak_under_permutation(model, prompts, kv_tokens, block_size) -> Result:
    """Argmax is stable under logit-preserving permutations of the batch.

    At temperature 0 the sampler is pure argmax with ties broken by the lowest
    token id, so permuting the batch must not move a single token. Permutation
    changes admission order, packing order, and therefore each request's offset
    inside every GEMM, without changing any request's own logits.
    """
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


# ---- the block-boundary cases ------------------------------------------------


def boundary_cases(model, kv_tokens, block_size, vocab, seed=555) -> Result:
    """The cases the upstream bug lives at, hit on purpose.

    `prefix_len == block_size` is SGLang's open corruption shape. The rest are
    the neighbours the architecture doc's coverage section asks for at value,
    value minus one, and value plus one, plus the pairings that made the
    upstream bug visible.
    """
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

    # A shared prefix of exactly block_size, and one either side.
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

    # prefix_len == 0 beside a nonzero-prefix request in the same batch. This
    # pairing is what made the upstream bug visible.
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

    # A hit covering the entire prompt: zero new prefill needed.
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
    # The hit deliberately stops one block short of the prompt: a request with
    # nothing left to compute has no logits to sample from, because the forward
    # pass that would have produced them never ran. So the expectation is
    # len(whole) - block_size, and the tokens must still match the cold run.
    expected_full_hit = len(whole) - block_size
    record(
        "cache hit covering the whole prompt",
        full.generated == cold_whole["f"] and full.cache_hit_tokens == expected_full_hit,
        f"hit {full.cache_hit_tokens}/{len(whole)} tokens, last block recomputed "
        f"so the request has logits to sample",
    )

    # Eviction of a block whose refcount just dropped to zero in the same step.
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

    # A fork whose copy-on-write triggers exactly at a block boundary.
    pool = paged.PagedKVCache(
        num_blocks=8, num_layers=1, num_kv_heads=1, head_dim=8,
        device="cpu", dtype=torch.float16, block_size=block_size,
    )
    pool.create("parent")
    pool.reserve("parent", block_size * 2)
    pool.k[0][pool.sequences["parent"].block_ids[1]].fill_(2.5)
    pool.fork("parent", "child")
    private = pool.ensure_writable("child", 1)
    pool.audit()
    boundary_counters.merge(pool.counters)
    record(
        "fork with copy-on-write at a block boundary",
        private != pool.sequences["parent"].block_ids[1]
        and torch.equal(pool.k[0][private], pool.k[0][pool.sequences["parent"].block_ids[1]])
        and pool.refcount[private] == 1,
        f"copied logical block 1 into physical {private}",
    )

    require_fired(boundary_counters, "cache_hit", "eviction_taken", "cow_performed",
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
