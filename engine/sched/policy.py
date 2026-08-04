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


class Policy:
    """Base class. Every decision point in the engine is a method here."""

    name = "base"

    def admit(self, state: SchedulerState, uid: str, blocks_needed: int) -> bool:
        raise NotImplementedError

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        """How many prompt tokens to prefill in this step. Week 4."""
        raise NotYetScheduled("chunked prefill is week 4")

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
        return blocks_needed <= state.free_blocks


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
