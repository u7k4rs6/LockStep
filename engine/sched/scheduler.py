"""Naive continuous batching. Every decision goes through the policy seam.

Week 2 scope from the PRD: requests join and leave the running batch. Chunked
prefill, preemption, and the prefix cache are week 4 and are not here; the policy
interface already names them so that adding them is filling in a decision point
rather than inventing one.

The step structure, which is what a schedule is a sequence of:

    1. Ask the policy which waiting requests to admit. Admitted requests prefill
       their whole prompt this step.
    2. Every already-running request decodes one token.
    3. Requests that hit their limit or an EOS finish and release their blocks.

Prefill and decode ride in the same packed batch, which is what makes this
continuous batching rather than a queue of phases, and which is what puts a
batch-dependent M in front of every GEMM.

Nothing here chooses. `admit` is the policy's. The order of the running set is
insertion order, which is a property of the recorded schedule rather than a
decision; the trajectory hash covers it so a reordering cannot pass unnoticed.
"""

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

    # Positions whose KV is in the pool. Prefill and decode both advance it,
    # which is what lets one code path serve both: a decode step is a prefill of
    # exactly one token at the end of the context. Preemption sets it back to
    # whatever survived, and recompute simply prefills the gap.
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
        # The engine owns its own lifecycle event stream. Recording it in a
        # policy would mean a policy that declines to record produces a
        # different coverage report from one that does, and would emit ADMIT
        # where the state machine says RESUME because a policy cannot see that a
        # request was previously preempted.
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
        # uid -> one fp16 logit row per emitted token, in emission order.
        self.emitted_logits: dict[str, list] = {}

    def submit(self, request: Request) -> None:
        """Queue a request, refusing one that can never fit.

        Found by the fuzzer against the unmutated engine, minimized to a single
        61-token request with max_new_tokens=4 against a 2-block pool at
        block_size 32: 65 tokens needs 3 blocks, so the request could never be
        admitted, but it sat in the queue until nothing was running and the
        scheduler raised "N requests waiting, M blocks free, and nothing running
        to free them". That message named free blocks the request could never
        have used, which pointed at pressure rather than at the real cause.

        A request larger than the whole pool is not a pressure condition and no
        amount of eviction or preemption will help, so it is refused at the door
        with the arithmetic that decides it.
        """
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
        """Reserve blocks, evicting cached-but-unused ones under pressure.

        Which block to evict is a policy decision (architecture doc section 6),
        so the candidate set comes from the cache and the choice comes from the
        policy. The candidate set is only blocks whose sole remaining holder is
        the cache index; including a block a running sequence holds is mutation
        operator 10.1's "eviction eligible-set includes a running sequence".
        """
        # The loop is bounded by the candidate count at entry, not by an argument
        # that the set shrinks. Each pass evicts exactly one block and no pass
        # creates a candidate, so more passes than there were candidates means
        # the invariant broke. A hostile schedule probing this surface should get
        # a loud assertion, not a hang the campaign reports as a timeout.
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

            # A hit covering the whole prompt leaves nothing to compute, and a
            # request with nothing to compute has no logits to sample from: the
            # KV for the last position exists, but the forward pass that would
            # have produced its logits never ran. So the last block is always
            # recomputed. This costs one block of prefill and is why a full-prompt
            # hit is not a no-op.
            #
            # It cannot change the result: recomputing that block reproduces the
            # cached KV exactly, which is chunk invariance, the same property
            # that makes the cache sound in the first place.
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
        """Recompute preemption: drop the KV, keep the tokens.

        The request goes back to the front of the waiting queue with its
        generated tokens intact and kv_len reset, so re-admission rebuilds the
        whole context in one prefill. That is only bitwise sound because
        decode-produced KV equals prefill-produced KV, which harness/mr/
        equivalence.py proves at the tensor level.
        """
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
            # Nothing could be admitted and nothing is running. Every waiting
            # request fits the pool in principle, since submit() refuses those
            # that do not, so this is genuine fragmentation or a policy that
            # will not admit. Naming the smallest waiting request makes the
            # difference visible.
            smallest = min(self._blocks_needed(r) for r in self.waiting)
            raise paged.OutOfBlocks(
                f"{len(self.waiting)} requests waiting, smallest needs {smallest} blocks, "
                f"{self.pool.free_blocks} free and "
                f"{self._state().reclaimable_blocks} reclaimable, nothing running to free more"
            )

        # Preemption, before any work is packed. A preempted request loses its
        # blocks and its KV; recompute rebuilds them from the context, which is
        # sound exactly because decode-produced KV is bit-identical to
        # prefill-produced KV (architecture doc 4.1).
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

        # Prefill and decode ride in the same packed batch, through one code
        # path: a decode step is a prefill of one token at the end of the
        # context, so there is no second path for it to diverge from.
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
                # Mid-context: this was a partial chunk, so nothing is emitted.
                continue
            if request.kv_len == len(request.prompt) and not request.generated:
                self.counters.hit("prefill_complete")
            row = logits[request.uid][-1]
            position = request.total_tokens() - 1
            # The raw logit row at every emitted position, kept so a relation can
            # compare decode-phase bits against canonical. MR1 compared logit
            # bytes only for whole-prompt prefills at position 0; every other
            # relation and the fuzz oracle compared emitted token ids, so a
            # batch-dependent perturbation below the top-1-to-top-2 gap survived
            # every check until it happened to flip a token, which is exactly the
            # masking harness/mr/equivalence.py warns about in its own docstring.
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
                # Which half of the disjunction fired. Without this the coverage
                # report cannot tell that EOS finishing has never executed.
                if request.generated and request.generated[-1] in self.eos:
                    self.counters.hit("finish_eos")
                else:
                    self.counters.hit("finish_limit")
                self._event(request.uid, Event.FINISH)
                self.running.remove(request)
                self.done.append(request)
                if self.cache is not None:
                    # Index whole blocks before releasing, so the cache takes its
                    # reference while the blocks are still alive.
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
