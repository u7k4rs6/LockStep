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

from engine.audit.trajectory import TrajectoryHash
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

    def total_tokens(self) -> int:
        return len(self.prompt) + len(self.generated)


class Scheduler:
    """Continuous batching over a paged KV pool."""

    def __init__(
        self,
        model: Qwen3,
        num_blocks: int,
        policy: Policy | None = None,
        eos_token_ids: set[int] | None = None,
        audit: bool = True,
    ):
        self.model = model
        self.policy = policy or DefaultPolicy()
        self.eos = eos_token_ids or set()
        self.audit_enabled = audit

        self.pool = paged.PagedKVCache(
            num_blocks=num_blocks,
            num_layers=model.cfg.num_hidden_layers,
            num_kv_heads=model.cfg.num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=model.device,
            dtype=torch.float16,
        )
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.done: list[Request] = []
        self.step_index = 0
        self.trajectory = TrajectoryHash()

    def submit(self, request: Request) -> None:
        self.waiting.append(request)

    def _state(self) -> SchedulerState:
        return SchedulerState(
            step=self.step_index,
            waiting=tuple(r.uid for r in self.waiting),
            running=tuple(r.uid for r in self.running),
            free_blocks=self.pool.free_blocks,
            total_blocks=self.pool.num_blocks,
        )

    def _blocks_needed(self, request: Request) -> int:
        total = len(request.prompt) + request.max_new_tokens
        return -(-total // paged.BLOCK_SIZE)

    def step(self) -> bool:
        """Run one scheduler step. Returns False when there is nothing left."""
        if not self.waiting and not self.running:
            return False

        state = self._state()
        admitted: list[Request] = []
        for request in list(self.waiting):
            if self.policy.admit(state, request.uid, self._blocks_needed(request)):
                self.waiting.remove(request)
                self.pool.create(request.uid)
                self.pool.reserve(request.uid, len(request.prompt) + request.max_new_tokens)
                self.running.append(request)
                admitted.append(request)
                state = self._state()

        if not self.running:
            # Nothing could be admitted and nothing is running: the pool cannot
            # satisfy the head of the queue. Failing loudly beats spinning.
            raise paged.OutOfBlocks(
                f"{len(self.waiting)} requests waiting, {self.pool.free_blocks} blocks free, "
                "and nothing running to free them"
            )

        work: list[tuple[str, list[int], int]] = []
        for request in self.running:
            if request in admitted:
                work.append((request.uid, list(request.prompt), 0))
            else:
                work.append((request.uid, [request.generated[-1]], request.total_tokens() - 1))

        logits = self.model.forward_batch(self.pool, work)

        emitted: list[tuple[str, int]] = []
        for request in self.running:
            row = logits[request.uid][-1]
            position = request.total_tokens() - 1
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
            self.pool.sequences[request.uid].length = request.total_tokens()

        self.trajectory.observe_step(
            step=self.step_index,
            work=work,
            emitted=emitted,
            logits={uid: rows for uid, rows in logits.items()},
            pool_state=self.pool.state_digest(),
        )

        if self.audit_enabled:
            self.pool.audit()

        for request in list(self.running):
            done = (
                len(request.generated) >= request.max_new_tokens
                or request.generated[-1] in self.eos
            )
            if done:
                request.finished = True
                self.running.remove(request)
                self.done.append(request)
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
