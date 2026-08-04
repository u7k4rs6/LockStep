"""The seam. Every scheduler decision is a call into a policy object.

docs/02-technical-architecture.md section 6: "The single most important
architectural decision: `sched/policy.py` is the seam. Every scheduler decision
(admit, chunk boundary, preempt or not, which block to evict, whether to honor a
cache hit) is a call into a policy object. Production uses a heuristic policy;
the fuzzer supplies an adversarial one; replay supplies a recorded one. Because
execution is a pure function of (W, sigma, seeds), ddmin is exact rather than
best-effort. This is the entire differentiator versus live-server trace fuzzing.
Do not let scheduling logic leak outside this seam."

The interface is defined in full in week 2 even though week 2 only calls two of
its methods. That is deliberate. The seam is worth nothing if it is widened later
to fit whatever the scheduler happens to need in week 4: the fuzzer's power comes
from every decision being enumerable *now*, and a method added after the
scheduler already made that decision inline is a decision that was made outside
the seam first.

Decisions not yet reachable raise `NotYetScheduled` in the default policy rather
than returning a plausible default, so that a caller wiring up chunked prefill in
week 4 gets an error instead of a silent policy that was never reviewed.

`Decision` values are what a recorded schedule is made of. They are plain data
and must stay comparable and hashable so `replay` can assert it is replaying the
same sigma it recorded, and so ddmin can slice a schedule.
"""

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
    """What a policy is allowed to see when it decides.

    Deliberately a snapshot rather than the live engine. A policy that could
    reach into the engine could read a batch-derived quantity and route it into a
    kernel config, which is the thing the whole design forbids; passing a frozen
    view makes that a visible act rather than an accident.
    """

    step: int
    waiting: tuple[str, ...]
    running: tuple[str, ...]
    free_blocks: int
    total_blocks: int
    # Blocks the prefix cache holds that no live sequence is using. They are
    # capacity, not occupancy: admission that counted only free_blocks would
    # refuse a request the pool could serve by evicting, which makes the whole
    # eviction path unreachable.
    reclaimable_blocks: int = 0


class Policy:
    """Base class. Every decision point in the engine is a method here."""

    name = "base"

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        raise NotImplementedError

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        """How many prompt tokens to prefill in this step.

        Must return at least 1 and at most `remaining`. The engine imposes no
        alignment: a policy may return any value in that range, including ones
        that land mid-block and mid-split, because I2 says the bits must not
        care. A policy that only ever returned multiples of the block size would
        make the suite green without testing anything.
        """
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
    """The production heuristic. Trivial in week 2, and that is the point.

    Admit in arrival order while blocks remain. There is no cleverness to hide a
    bug behind yet, which makes it a good baseline for the fuzzer to diverge from
    later.
    """

    name = "default"

    def __init__(self, max_running: int = 32):
        self.max_running = max_running

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        if len(state.running) >= self.max_running:
            return False
        return blocks_needed <= state.free_blocks + state.reclaimable_blocks

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        """Prefill the whole remainder. The production heuristic does not chunk.

        Chunking exists to bound step latency, which is week 7's concern. The
        seam is exercised regardless: FixedChunkPolicy and ScriptedChunkPolicy
        drive it in the tests, which is the point of the decision living here.
        """
        return remaining

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        """Never, by default. Preemption is a response to pressure, and the
        default policy admits only what fits, so it never creates any."""
        return False

    def evict_victim(self, state: SchedulerState, candidates: tuple[int, ...]) -> int:
        """Lowest block id among the candidates.

        Not least-recently-used: recency is wall-clock-shaped state, and a
        replayed schedule would have to reproduce it to reproduce the eviction.
        Lowest-id is a pure function of the candidate set, which keeps eviction
        inside (workload, schedule, seeds).
        """
        return min(candidates)

    def honor_cache_hit(self, state: SchedulerState, uid: str, hit_length: int) -> bool:
        """Always. Refusing is what the fuzzer does to test MR4 both ways."""
        return True


class RecordingPolicy(Policy):
    """Wraps another policy and records every decision it makes.

    The recorded list is the sigma that `ReplayPolicy` re-executes and that MR6
    asserts is identical across runs.
    """

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
    """Replays a recorded sigma exactly, and refuses to improvise.

    If the engine asks something the recording does not answer, that is a
    divergence between the recorded run and this one, and it fails loudly rather
    than falling back to a heuristic and producing a run that merely resembles
    the original.
    """

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
    """Prefill along an explicitly given partition.

    Used by MR2 to drive partitions that land exactly on, and one either side of,
    the block size, the split size, and the sequence end. Those boundaries are
    where a fencepost lives, and a random partition reaches them only by luck.
    """

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
    """Preempt one named request at one named decode step, exactly once.

    MR3 sweeps the preemption point rather than sampling it, so the relation
    needs to place the preemption at a chosen step rather than hope a heuristic
    lands there.
    """

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
    """Preempt one request at several named steps, for depth sweeps.

    Preemption depth is a coverage dimension in architecture doc 8.2, up to 3.
    Depth matters because preempt-resume-preempt is where allocator state gets
    interesting: a block freed by the first preemption can be reallocated to
    another request and then be needed again by the second.
    """

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
