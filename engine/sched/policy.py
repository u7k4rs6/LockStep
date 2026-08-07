"""The seam. Every scheduler decision is a call into a policy object."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NotYetScheduled(NotImplementedError):
    """A decision point exists in the interface but nothing calls it yet."""


class Event(str, Enum):
    """The per-request alphabet from architecture doc section 2."""

    ADMIT = "admit"
    CHUNK = "chunk"
    DECODE = "decode"
    PREEMPT_RC = "preempt_rc"
    RESUME = "resume"
    EVICT = "evict"
    CACHE_HIT = "cache_hit"
    FINISH = "finish"


@dataclass(frozen=True, order=True)
class Decision:
    """One scheduler decision, recorded. The unit ddmin slices."""

    step: int
    event: Event
    uid: str
    detail: tuple = ()

    def as_tuple(self) -> tuple:
        return (self.step, self.event.value, self.uid, self.detail)


@dataclass(frozen=True)
class SchedulerState:
    """What a policy is allowed to see when it decides."""

    step: int
    waiting: tuple[str, ...]
    running: tuple[str, ...]
    free_blocks: int
    total_blocks: int
    reclaimable_blocks: int = 0


class Policy:
    """Base class. Every decision point in the engine is a method here."""

    name = "base"

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        raise NotImplementedError

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        """How many prompt tokens to prefill in this step."""
        raise NotImplementedError

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        """Week 4."""
        raise NotYetScheduled("preemption is week 4")

    def evict_victim(self, state: SchedulerState, candidates: tuple[int, ...]) -> int:
        """Which block to reclaim. Week 4."""
        raise NotYetScheduled("eviction under pressure is week 4")

    def honor_cache_hit(self, state: SchedulerState, uid: str, hit_length: int) -> bool:
        """Whether to reuse a matched prefix. Week 4."""
        raise NotYetScheduled("the prefix cache is week 4")


class DefaultPolicy(Policy):
    """The production heuristic. Trivial in week 2, and that is the point."""

    name = "default"

    def __init__(self, max_running: int = 32):
        self.max_running = max_running

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        if len(state.running) >= self.max_running:
            return False
        return blocks_needed <= state.free_blocks + state.reclaimable_blocks

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        """Prefill the whole remainder. The production heuristic does not chunk."""
        return remaining

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        """Never, by default. Preemption is a response to pressure, and the"""
        return False

    def evict_victim(self, state: SchedulerState, candidates: tuple[int, ...]) -> int:
        """Lowest block id among the candidates."""
        return min(candidates)

    def honor_cache_hit(self, state: SchedulerState, uid: str, hit_length: int) -> bool:
        """Always. Refusing is what the fuzzer does to test MR4 both ways."""
        return True


class RecordingPolicy(Policy):
    """Wraps another policy and records every decision it makes."""

    def __init__(self, inner: Policy):
        self.inner = inner
        self.name = f"recording({inner.name})"
        self.decisions: list[Decision] = []

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        verdict = self.inner.admit(state, uid, blocks_needed)
        if verdict:
            self.decisions.append(Decision(state.step, Event.ADMIT, uid, (blocks_needed,)))
        return verdict

    def record(self, step: int, event: Event, uid: str, detail: tuple = ()) -> None:
        """For events the engine takes without asking, such as finishing."""
        self.decisions.append(Decision(step, event, uid, detail))


class ReplayPolicy(Policy):
    """Replays a recorded sigma exactly, and refuses to improvise."""

    name = "replay"

    def __init__(self, decisions: list[Decision]):
        self.decisions = list(decisions)
        self._admits = {(d.step, d.uid) for d in self.decisions if d.event is Event.ADMIT}

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        return (state.step, uid) in self._admits


class FixedChunkPolicy(DefaultPolicy):
    """Prefill in fixed-size chunks. The configuration shipped engines test."""

    def __init__(self, chunk: int, max_running: int = 32):
        super().__init__(max_running=max_running)
        self.chunk = chunk
        self.name = f"fixed-chunk({chunk})"

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        return min(self.chunk, remaining)


class ScriptedChunkPolicy(DefaultPolicy):
    """Prefill along an explicitly given partition."""

    def __init__(self, partitions: dict[str, list[int]], max_running: int = 32):
        super().__init__(max_running=max_running)
        self.partitions = {uid: list(sizes) for uid, sizes in partitions.items()}
        self.name = "scripted-chunk"

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        queue = self.partitions.get(uid)
        if not queue:
            return remaining
        return min(queue.pop(0), remaining)


class PreemptAtStepPolicy(DefaultPolicy):
    """Preempt one named request at one named decode step, exactly once."""

    def __init__(self, uid: str, at_step: int, max_running: int = 32):
        super().__init__(max_running=max_running)
        self.uid = uid
        self.at_step = at_step
        self.fired = False
        self.name = f"preempt({uid}@{at_step})"

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        if self.fired or uid != self.uid or state.step != self.at_step:
            return False
        self.fired = True
        return True


class PreemptAtStepsPolicy(DefaultPolicy):
    """Preempt one request at several named steps, for depth sweeps."""

    def __init__(self, uid: str, at_steps: tuple[int, ...], max_running: int = 32):
        super().__init__(max_running=max_running)
        self.uid = uid
        self.at_steps = set(at_steps)
        self.fired_at: list[int] = []
        self.name = f"preempt({uid}@{sorted(at_steps)})"

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        if uid != self.uid or state.step not in self.at_steps:
            return False
        if state.step in self.fired_at:
            return False
        self.fired_at.append(state.step)
        return True


class NoCacheHitPolicy(DefaultPolicy):
    """Refuse every prefix hit. The cold half of MR4."""

    name = "no-cache-hit"

    def honor_cache_hit(self, state: SchedulerState, uid: str, hit_length: int) -> bool:
        return False
