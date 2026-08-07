"""Swarm generation. Groce et al., ISSTA 2012."""

from __future__ import annotations

import random
from dataclasses import dataclass

from engine.kv.paged import SUPPORTED_BLOCK_SIZES
from harness.sim.driver import Case, RequestSpec

PROMPT_SHAPES = {
    "tiny": (1, 2, 3, 7),
    "sub_block": (5, 9, 15),
    "around_block": (15, 16, 17, 31, 32, 33, 63, 64, 65),
    "around_split": (511, 512, 513),
    "long": (200, 300, 600),
}
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
    pool_pressure: str
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
        preempt_rate=rng.choice((0.0, 0.0, 0.15, 0.5, 0.9)),
        cache_enabled=rng.random() > 0.25,
        refuse_cache_rate=rng.choice((0.0, 0.0, 0.3, 1.0)),
        pool_pressure=rng.choice(("roomy", "tight", "tight", "starved")),
        max_requests=rng.choice((1, 2, 4, 8, 16, 31, 32)),
    )


def draw_case(config: SwarmConfig, vocab: int, index: int) -> Case:
    """Draw one case from a campaign configuration."""
    rng = random.Random((config.seed << 20) ^ index)

    lengths = []
    for shape in config.prompt_shapes:
        lengths.extend(PROMPT_SHAPES[shape])

    if config.max_requests >= 16 and rng.random() < 0.15:
        count = rng.choice((16, 31, 32))
    else:
        count = rng.randint(1, min(config.max_requests, 8))
    shared_prefix = None
    shared_len = 0
    if config.cache_enabled and rng.random() < 0.6:
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
    """A campaign aimed at `_reserve_with_eviction` specifically."""
    rng = random.Random(seed)
    cases: list[Case] = []

    for index in range(count):
        block_size = rng.choice((8, 16, 32))
        shared = tuple(rng.randrange(vocab) for _ in range(block_size * rng.randint(1, 3)))

        requests = []
        width = rng.randint(2, 6)
        for i in range(width):
            tail = tuple(rng.randrange(vocab) for _ in range(rng.randint(1, block_size * 2)))
            prompt = shared + tail if rng.random() < 0.85 else tail
            requests.append(RequestSpec(
                uid=f"r{i:02d}", prompt=prompt, seed=2000 + i,
                max_new_tokens=rng.choice((1, 2, 4)),
                temperature=0.0, top_p=1.0,
            ))

        needed = sum(-(-(len(r.prompt) + r.max_new_tokens) // block_size) for r in requests)
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
