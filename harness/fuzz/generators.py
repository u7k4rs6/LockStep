"""Swarm generation. Groce et al., ISSTA 2012.

docs/02-technical-architecture.md section 8.3: "each campaign randomly disables
or boosts feature subsets, which is what escapes the easy middle of the
distribution."

A uniform generator produces the average case every time: middling prompts,
middling batch widths, chunking on sometimes. Swarm testing instead draws a
*configuration* per campaign, switching whole features off or cranking them up,
so some campaigns never chunk and some chunk every token, some never preempt and
some preempt constantly. The interesting states live at those extremes.

Every draw comes from a seeded generator, so a campaign is reproducible from its
seed alone and a finding replays exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from engine.kv.paged import SUPPORTED_BLOCK_SIZES
from harness.sim.driver import Case, RequestSpec

# Values chosen to sit on and around the boundaries the coverage report tracks:
# the block sizes, the 512 attention split, and small free-block counts.
PROMPT_SHAPES = {
    "tiny": (1, 2, 3, 7),
    "sub_block": (5, 9, 15),
    "around_block": (15, 16, 17, 31, 32, 33, 63, 64, 65),
    "around_split": (511, 512, 513),
    "long": (200, 300, 600),
}
# "just_below" exists because the below case is what catches a rounding-up bug,
# and one of the two real bugs found so far was a rounding-up bug. A generator
# that only produced aligned and over-aligned chunks would never reach it.
CHUNK_SHAPES = {
    "none": (),
    "one_token": (1,),
    "block_aligned": (16, 32, 64),
    "just_below": (7, 15, 31, 63),
    "misaligned": (13, 37),
    "ragged": (1, 5, 17, 3, 64),
}


@dataclass
class SwarmConfig:
    """One campaign's feature configuration. Printed with any finding."""

    seed: int
    prompt_shapes: tuple[str, ...]
    chunk_shape: str
    block_size: int
    preempt_rate: float
    cache_enabled: bool
    refuse_cache_rate: float
    pool_pressure: str          # roomy | tight | starved
    max_requests: int

    def describe(self) -> str:
        return (
            f"seed={self.seed} prompts={'+'.join(self.prompt_shapes)} "
            f"chunks={self.chunk_shape} block={self.block_size} "
            f"preempt={self.preempt_rate:.2f} cache={'on' if self.cache_enabled else 'off'} "
            f"refuse={self.refuse_cache_rate:.2f} pool={self.pool_pressure} "
            f"n<={self.max_requests}"
        )


def draw_config(seed: int) -> SwarmConfig:
    """Draw a campaign configuration. Features are switched off, not scaled."""
    rng = random.Random(seed)
    shapes = tuple(rng.sample(sorted(PROMPT_SHAPES), rng.randint(1, 3)))
    return SwarmConfig(
        seed=seed,
        prompt_shapes=shapes,
        chunk_shape=rng.choice(sorted(CHUNK_SHAPES)),
        block_size=rng.choice(SUPPORTED_BLOCK_SIZES),
        # Bimodal on purpose: mostly never, sometimes constantly.
        preempt_rate=rng.choice((0.0, 0.0, 0.15, 0.5, 0.9)),
        cache_enabled=rng.random() > 0.25,
        refuse_cache_rate=rng.choice((0.0, 0.0, 0.3, 1.0)),
        pool_pressure=rng.choice(("roomy", "tight", "tight", "starved")),
        # 31 and 32 are in the list because off-by-one against a 32-wide tile is
        # where tile-boundary bugs live, and co-batching pressure there is a
        # different thing from MR1's clean sweep.
        max_requests=rng.choice((1, 2, 4, 8, 16, 31, 32)),
    )


def draw_case(config: SwarmConfig, vocab: int, index: int) -> Case:
    """Draw one case from a campaign configuration."""
    rng = random.Random((config.seed << 20) ^ index)

    lengths = []
    for shape in config.prompt_shapes:
        lengths.extend(PROMPT_SHAPES[shape])

    # Draw the exact boundary widths rather than uniformly below the cap. With
    # randint(1, 31) the chance of landing on 31 is 1 in 31, so the predicate
    # that is most likely to find something was the one least likely to be hit.
    # BLOCK_M is 16, so a batch of 31 spans two tiles with the second partial,
    # which is exactly where a tile-boundary bug lives.
    # Stratified rather than uniform: the boundary widths are guaranteed to
    # appear but are a minority, because a batch of 31 costs 31 forward passes
    # per step and letting them dominate turned a 40-minute campaign into a
    # five-hour one. 15 percent keeps the predicate reachable in a 240-case
    # campaign while leaving most of the budget for cheap cases.
    if config.max_requests >= 16 and rng.random() < 0.15:
        count = rng.choice((16, 31, 32))
    else:
        count = rng.randint(1, min(config.max_requests, 8))
    shared_prefix = None
    shared_len = 0
    if config.cache_enabled and rng.random() < 0.6:
        # A shared prefix is what makes a cache hit possible at all; without one
        # a cache-enabled campaign would only ever record misses.
        # At, one below, and one above the block size: the shape SGLang's open
        # corruption bug sits at, driven on purpose.
        shared_len = max(1, rng.choice((config.block_size, config.block_size * 2,
                                        config.block_size + 1, config.block_size - 1)))
        shared_prefix = tuple(rng.randrange(vocab) for _ in range(shared_len))

    requests = []
    for i in range(count):
        length = rng.choice(lengths)
        body = tuple(rng.randrange(vocab) for _ in range(length))
        prompt = (shared_prefix + body) if shared_prefix and rng.random() < 0.8 else body
        requests.append(RequestSpec(
            uid=f"r{i:02d}",
            prompt=prompt,
            seed=1000 + i,
            # Short generations. Findings cluster in the first few steps, so a
            # long decode tail buys coverage of 2-grams already reached and
            # costs a forward pass per request per step.
            max_new_tokens=rng.choice((1, 2, 4)) if count <= 8 else 1,
            temperature=rng.choice((0.0, 0.0, 0.8)),
            top_p=0.95,
        ))

    total_tokens = sum(len(r.prompt) + r.max_new_tokens for r in requests)
    blocks_needed = sum(-(-(len(r.prompt) + r.max_new_tokens) // config.block_size)
                        for r in requests)
    pressure = {"roomy": 3.0, "tight": 1.1, "starved": 0.6}[config.pool_pressure]
    num_blocks = max(2, int(blocks_needed * pressure))

    preempt_at = ()
    if config.preempt_rate > 0:
        picks = []
        for request in requests:
            for step in range(6):
                if rng.random() < config.preempt_rate:
                    picks.append((request.uid, step))
        preempt_at = tuple(picks[:12])

    refuse = ()
    if config.refuse_cache_rate > 0:
        refuse = tuple(s for s in range(8) if rng.random() < config.refuse_cache_rate)

    return Case(
        requests=tuple(requests),
        chunk_plan=CHUNK_SHAPES[config.chunk_shape],
        preempt_at=preempt_at,
        refuse_cache_at=refuse,
        block_size=config.block_size,
        num_blocks=num_blocks,
        enable_cache=config.cache_enabled,
        shared_prefix_len=shared_len,
        label=f"swarm{config.seed}#{index}",
    )


def eviction_cases(vocab: int, count: int, seed: int = 7717) -> list[Case]:
    """A campaign aimed at `_reserve_with_eviction` specifically.

    The fragile surface was identified before the tool was pointed at it, so it
    is targeted deliberately rather than left for a uniform campaign to wander
    into. Three shapes, all of which put admission and eviction in contention:

      * oscillating pressure: pools sized just at, just under, and just over what
        the workload needs, so the free pool empties and refills repeatedly
      * requests arriving exactly when the pool empties, which is when admission
        races the reclaim path
      * heavy prefix sharing, so most blocks are cache-pinned and the evictable
        set is small and changes every step

    The pass-count assertion inside the loop is the thing under test: a livelock
    should be a loud assertion, not a hang the campaign reports as a timeout.
    """
    rng = random.Random(seed)
    cases: list[Case] = []

    for index in range(count):
        block_size = rng.choice((8, 16, 32))
        shared = tuple(rng.randrange(vocab) for _ in range(block_size * rng.randint(1, 3)))

        requests = []
        width = rng.randint(2, 6)
        for i in range(width):
            tail = tuple(rng.randrange(vocab) for _ in range(rng.randint(1, block_size * 2)))
            # Most requests share the prefix, so the cache fills and the
            # evictable set is contended rather than plentiful.
            prompt = shared + tail if rng.random() < 0.85 else tail
            requests.append(RequestSpec(
                uid=f"r{i:02d}", prompt=prompt, seed=2000 + i,
                max_new_tokens=rng.choice((1, 2, 4)),
                temperature=0.0, top_p=1.0,
            ))

        needed = sum(-(-(len(r.prompt) + r.max_new_tokens) // block_size) for r in requests)
        # Just enough for the largest single request, and never enough for all:
        # the pool must churn to make progress.
        largest = max(-(-(len(r.prompt) + r.max_new_tokens) // block_size) for r in requests)
        num_blocks = max(largest, int(needed * rng.choice((0.35, 0.5, 0.65))))

        cases.append(Case(
            requests=tuple(requests),
            chunk_plan=rng.choice(((), (1,), (block_size,), (7,))),
            preempt_at=tuple(
                (requests[rng.randrange(width)].uid, rng.randrange(5))
                for _ in range(rng.randint(0, 3))
            ),
            block_size=block_size,
            num_blocks=num_blocks,
            enable_cache=True,
            label=f"eviction#{index}",
        ))
    return cases
