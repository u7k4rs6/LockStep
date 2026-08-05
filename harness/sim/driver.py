"""Deterministic simulation driver: execution is a function of (W, sigma, seeds).

docs/02-technical-architecture.md section 6: "Because execution is a pure
function of (W, sigma, seeds), ddmin is exact rather than best-effort. This is
the entire differentiator versus live-server trace fuzzing."

A Case is that triple. Running one twice gives the same trajectory hash; running
a subset of one is a well-defined smaller case, which is what makes minimization
exact rather than a search that sometimes reproduces.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.audit.counters import Counters  # noqa: E402
from engine.kv import paged  # noqa: E402
from engine.sched.lifecycle import Event  # noqa: E402
from engine.sched.policy import DefaultPolicy, SchedulerState  # noqa: E402
from engine.sched.scheduler import OversizedRequest, Request, Scheduler  # noqa: E402


@dataclass(frozen=True)
class RequestSpec:
    uid: str
    prompt: tuple[int, ...]
    seed: int
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0

    def to_request(self) -> Request:
        return Request(
            uid=self.uid, prompt=list(self.prompt), seed=self.seed,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p,
        )


@dataclass(frozen=True)
class Case:
    """(W, sigma, seeds). Everything that determines an execution."""

    requests: tuple[RequestSpec, ...]
    chunk_plan: tuple[int, ...] = ()       # cycled through for chunk_boundary
    preempt_at: tuple[tuple[str, int], ...] = ()   # (uid, step)
    refuse_cache_at: tuple[int, ...] = ()  # steps where a hit is declined
    block_size: int = 16
    num_blocks: int = 64
    enable_cache: bool = True
    shared_prefix_len: int = 0   # true common prefix across requests, before rounding
    label: str = ""

    def shrunk(self, **changes) -> "Case":
        return replace(self, **changes)


class ScriptedPolicy(DefaultPolicy):
    """Drives every decision point from the Case. No heuristic anywhere.

    This is the seam doing its job: the fuzzer supplies decisions, the engine
    supplies none, so replaying the Case replays the schedule exactly.
    """

    def __init__(self, case: Case, events: dict[str, list[Event]]):
        super().__init__(max_running=32)
        self.case = case
        self.events = events
        self.name = f"scripted({case.label or 'case'})"
        self._chunk_index = 0
        self._preempted: set[tuple[str, int]] = set()

    def chunk_boundary(self, state: SchedulerState, uid: str, remaining: int) -> int:
        if not self.case.chunk_plan:
            return remaining
        take = self.case.chunk_plan[self._chunk_index % len(self.case.chunk_plan)]
        self._chunk_index += 1
        return max(1, min(take, remaining))

    def should_preempt(self, state: SchedulerState, uid: str) -> bool:
        key = (uid, state.step)
        if key in self.case.preempt_at and key not in self._preempted:
            self._preempted.add(key)
            return True
        return False

    def evict_victim(self, state: SchedulerState, candidates: tuple[int, ...]) -> int:
        return min(candidates)

    def honor_cache_hit(self, state: SchedulerState, uid: str, hit_length: int) -> bool:
        return state.step not in self.case.refuse_cache_at


@dataclass
class Outcome:
    trajectory: str
    outputs: dict
    counters: Counters
    events: dict
    steps: int
    error: str | None = None
    depths: dict = field(default_factory=dict)


def run_case(model, case: Case, audit: bool = True) -> Outcome:
    """Execute a Case. Any internal failure is captured, not raised.

    A campaign has to keep going past a finding, and a finding is exactly an
    exception from the audit or a divergence, so the driver returns it as data.
    """
    events: dict[str, list[Event]] = {}
    policy = ScriptedPolicy(case, events)
    scheduler = Scheduler(
        model,
        num_blocks=case.num_blocks,
        policy=policy,
        block_size=case.block_size,
        enable_prefix_cache=case.enable_cache,
        audit=audit,
    )
    rejected = []
    for spec in case.requests:
        try:
            scheduler.submit(spec.to_request())
        except OversizedRequest:
            # A refusal at the door is correct behaviour, not a finding. The
            # case simply contains a request this pool can never serve.
            rejected.append(spec.uid)

    error = None
    try:
        scheduler.run(max_steps=512)
    except (paged.AuditFailure, paged.OutOfBlocks, AssertionError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    depths = {}
    for request in scheduler.done + scheduler.running + scheduler.waiting:
        depths[request.uid] = request.preempt_count

    return Outcome(
        trajectory=scheduler.trajectory.hexdigest(),
        outputs={r.uid: list(r.generated) for r in scheduler.done},
        counters=scheduler.counters,
        events=scheduler.lifecycle,
        steps=scheduler.step_index,
        error=error,
        depths=depths,
    )
