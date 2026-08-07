"""Naive continuous batching. Every decision goes through the policy seam."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from engine.audit.counters import Counters
from engine.audit.trajectory import TrajectoryHash
from engine.cache.prefix import PrefixCache
from engine.kernels import registry
from engine.kv import paged
from engine.model.qwen3 import Qwen3
from engine.sampler import philox
from engine.sched.policy import DefaultPolicy, Event, Policy, RecordingPolicy, SchedulerState


@dataclass
class Request:
    """A request `r = (prompt x, sampling params s, seed k, uid)`."""

    uid: str
    prompt: list[int]
    seed: int = 0
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0

    generated: list[int] = field(default_factory=list)
    finished: bool = False

    kv_len: int = 0
    preempt_count: int = 0
    cache_hit_tokens: int = 0

    def context(self) -> list[int]:
        """Every token whose KV should eventually exist for this request."""
        return self.prompt + self.generated

    def total_tokens(self) -> int:
        return len(self.prompt) + len(self.generated)

    def needs_compute(self) -> int:
        return len(self.context()) - self.kv_len


class OversizedRequest(ValueError):
    """A request that cannot fit the pool even when the pool is empty."""


class Scheduler:
    """Continuous batching over a paged KV pool."""

    def __init__(
        self,
        model: Qwen3,
        num_blocks: int,
        policy: Policy | None = None,
        eos_token_ids: set[int] | None = None,
        audit: bool = True,
        block_size: int = paged.DEFAULT_BLOCK_SIZE,
        enable_prefix_cache: bool = False,
    ):
        self.model = model
        self.policy = policy or DefaultPolicy()
        self.eos = eos_token_ids or set()
        self.audit_enabled = audit
        self.counters = Counters()
        self.lifecycle: dict[str, list] = {}

        self.pool = paged.PagedKVCache(
            num_blocks=num_blocks,
            num_layers=model.cfg.num_hidden_layers,
            num_kv_heads=model.cfg.num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=model.device,
            dtype=torch.float16,
            block_size=block_size,
            counters=self.counters,
        )
        self.cache = (
            PrefixCache(block_size=block_size, counters=self.counters)
            if enable_prefix_cache else None
        )
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.done: list[Request] = []
        self.step_index = 0
        self.evictions = 0
        self.trajectory = TrajectoryHash()
        self.emitted_logits: dict[str, list] = {}

    def submit(self, request: Request) -> None:
        """Queue a request, refusing one that can never fit."""
        needed = self._blocks_needed(request)
        if needed > self.pool.num_blocks:
            raise OversizedRequest(
                f"{request.uid} needs {needed} blocks "
                f"({len(request.prompt)} prompt + {request.max_new_tokens} new tokens "
                f"at block_size {self.pool.block_size}) but the pool holds "
                f"{self.pool.num_blocks}. No eviction or preemption can satisfy it."
            )
        self.waiting.append(request)

    def _event(self, uid: str, event) -> None:
        self.lifecycle.setdefault(uid, []).append(event)

    def _state(self) -> SchedulerState:
        return SchedulerState(
            step=self.step_index,
            waiting=tuple(r.uid for r in self.waiting),
            running=tuple(r.uid for r in self.running),
            free_blocks=self.pool.free_blocks,
            total_blocks=self.pool.num_blocks,
            reclaimable_blocks=(
                len(self.cache.evictable_blocks(self.pool)) if self.cache else 0
            ),
        )

    def _blocks_needed(self, request: Request) -> int:
        total = len(request.prompt) + request.max_new_tokens
        return -(-total // self.pool.block_size)

    def _reserve_with_eviction(self, uid: str, tokens: int) -> None:
        """Reserve blocks, evicting cached-but-unused ones under pressure."""
        max_passes = len(self.cache.evictable_blocks(self.pool)) + 1 if self.cache else 1
        passes = 0
        while True:
            try:
                self.pool.reserve(uid, tokens)
                return
            except paged.OutOfBlocks:
                if self.cache is None:
                    raise
                passes += 1
                self.counters.hit("eviction_pass")
                assert passes <= max_passes, (
                    f"reserve-with-eviction ran {passes} passes for {uid} with only "
                    f"{max_passes - 1} candidates at entry. Each pass must evict "
                    "exactly one block and no pass may create a candidate, so this "
                    "is a livelock, not slow progress."
                )
                candidates = self.cache.evictable_blocks(self.pool)
                if not candidates:
                    raise
                victim = self.policy.evict_victim(self._state(), tuple(candidates))
                if victim not in candidates:
                    raise ValueError(
                        f"policy {self.policy.name} chose block {victim} to evict, "
                        f"which is not in the candidate set; a block held by a "
                        f"running sequence must never be reclaimed"
                    )
                self.cache.evict(victim, self.pool)
                self.evictions += 1
                self._event(uid, Event.EVICT)
                if isinstance(self.policy, RecordingPolicy):
                    self.policy.record(self.step_index, Event.EVICT, uid, (victim,))

    def _admit(self, request: Request, state: SchedulerState) -> None:
        """Create the sequence, honour any prefix hit, and reserve its blocks."""
        self.pool.create(request.uid)
        request.kv_len = 0
        request.cache_hit_tokens = 0

        if self.cache is not None:
            hit_tokens, blocks = self.cache.lookup(request.prompt)

            if hit_tokens >= len(request.prompt) and blocks:
                hit_tokens -= self.pool.block_size
                blocks = blocks[:-1]
                self.counters.hit("cache_full_prompt_trimmed")

            if hit_tokens and not self.policy.honor_cache_hit(state, request.uid, hit_tokens):
                self.counters.hit("cache_hit_refused")
            elif hit_tokens:
                self._event(request.uid, Event.CACHE_HIT)
                self.pool.adopt(request.uid, blocks)
                request.kv_len = hit_tokens
                request.cache_hit_tokens = hit_tokens
                if isinstance(self.policy, RecordingPolicy):
                    self.policy.record(
                        self.step_index, Event.CACHE_HIT, request.uid, (hit_tokens,)
                    )

        self._reserve_with_eviction(
            request.uid, len(request.prompt) + request.max_new_tokens
        )
        self.pool.sequences[request.uid].length = request.kv_len

    def _preempt(self, request: Request) -> None:
        """Recompute preemption: drop the KV, keep the tokens."""
        self.pool.release(request.uid)
        request.kv_len = 0
        request.preempt_count += 1
        self.running.remove(request)
        self.waiting.insert(0, request)
        if isinstance(self.policy, RecordingPolicy):
            self.policy.record(self.step_index, Event.PREEMPT_RC, request.uid)

    def step(self) -> bool:
        """Run one scheduler step. Returns False when there is nothing left."""
        if not self.waiting and not self.running:
            return False

        state = self._state()
        for request in list(self.waiting):
            if not self.policy.admit(state, request.uid, self._blocks_needed(request)):
                self.counters.hit("admit_refused")
                continue
            self.counters.hit("admit")
            if request.preempt_count:
                self.counters.hit("resume")
                self._event(request.uid, Event.RESUME)
            else:
                self._event(request.uid, Event.ADMIT)
            self.waiting.remove(request)
            self._admit(request, state)
            self.running.append(request)
            state = self._state()

        if not self.running:
            smallest = min(self._blocks_needed(r) for r in self.waiting)
            raise paged.OutOfBlocks(
                f"{len(self.waiting)} requests waiting, smallest needs {smallest} blocks, "
                f"{self.pool.free_blocks} free and "
                f"{self._state().reclaimable_blocks} reclaimable, nothing running to free more"
            )

        for request in list(self.running):
            if request.kv_len == 0 or not request.generated:
                continue
            if self.policy.should_preempt(state, request.uid):
                self._preempt(request)
                self.counters.hit("preempt_fired")
                self._event(request.uid, Event.PREEMPT_RC)
                if request.preempt_count == 2:
                    self.counters.hit("preempt_depth_2")
                elif request.preempt_count >= 3:
                    self.counters.hit("preempt_depth_3_plus")
                state = self._state()

        if not self.running:
            self.step_index += 1
            return True

        work: list[tuple[str, list[int], int]] = []
        for request in self.running:
            context = request.context()
            remaining = len(context) - request.kv_len
            take = self.policy.chunk_boundary(state, request.uid, remaining)
            if not 1 <= take <= remaining:
                raise ValueError(
                    f"policy {self.policy.name} returned chunk {take} for "
                    f"{request.uid} with {remaining} remaining; must be in [1, remaining]"
                )
            start = request.kv_len
            end = start + take
            work.append((request.uid, context[start:end], start))

            if take == 1 and start == len(context) - 1:
                self.counters.hit("decode_step")
                self._event(request.uid, Event.DECODE)
            else:
                self.counters.hit("prefill_chunk")
                self._event(request.uid,
                            Event.CHUNK if end < len(context) else Event.DECODE)
            if end < len(context):
                if end % self.pool.block_size:
                    self.counters.hit("chunk_boundary_mid_block")
                else:
                    self.counters.hit("chunk_boundary_on_block")
                if end % registry.SPLIT_SIZE:
                    self.counters.hit("chunk_boundary_mid_split")
            if end > registry.SPLIT_SIZE:
                self.counters.hit("attention_multi_split")
            else:
                self.counters.hit("attention_single_split")

        logits = self.model.forward_batch(self.pool, work)

        emitted: list[tuple[str, int]] = []
        for request, (_, tokens, start) in zip(self.running, work):
            request.kv_len = start + len(tokens)
            self.pool.sequences[request.uid].length = request.kv_len
            if request.kv_len < len(request.context()):
                continue
            if request.kv_len == len(request.prompt) and not request.generated:
                self.counters.hit("prefill_complete")
            row = logits[request.uid][-1]
            position = request.total_tokens() - 1
            self.emitted_logits.setdefault(request.uid, []).append(
                row.detach().cpu().clone()
            )
            if request.temperature <= 0:
                token = philox.greedy(row)
            else:
                token = philox.top_p(
                    row,
                    seed=request.seed,
                    uid=request.uid,
                    position=position,
                    p=request.top_p,
                    temperature=request.temperature,
                )
            request.generated.append(token)
            emitted.append((request.uid, token))

        self.trajectory.observe_step(
            step=self.step_index,
            work=work,
            emitted=emitted,
            logits={uid: rows for uid, rows in logits.items()},
            pool_state=(
                self.pool.state_digest(),
                self.cache.state_digest() if self.cache else (),
            ),
        )

        if self.audit_enabled:
            self.pool.audit()

        for request in list(self.running):
            done = request.generated and (
                len(request.generated) >= request.max_new_tokens
                or request.generated[-1] in self.eos
            )
            if done:
                request.finished = True
                self.counters.hit("finish")
                if request.generated and request.generated[-1] in self.eos:
                    self.counters.hit("finish_eos")
                else:
                    self.counters.hit("finish_limit")
                self._event(request.uid, Event.FINISH)
                self.running.remove(request)
                self.done.append(request)
                if self.cache is not None:
                    self.cache.insert(
                        request.context()[: request.kv_len],
                        self.pool.sequences[request.uid].block_ids,
                        self.pool,
                    )
                self.pool.release(request.uid)
                if isinstance(self.policy, RecordingPolicy):
                    self.policy.record(self.step_index, Event.FINISH, request.uid)

        self.step_index += 1
        return True

    def run(self, max_steps: int = 1024) -> dict[str, list[int]]:
        while self.step() and self.step_index < max_steps:
            pass
        if self.audit_enabled:
            self.pool.audit()
        return {r.uid: list(r.generated) for r in self.done}
