"""The request lifecycle as a state machine, and the coverage denominator."""

from __future__ import annotations

from enum import Enum
from itertools import product

from engine.sched.policy import Event


class State(str, Enum):
    """Where a request is in its life. Terminal states have no outgoing events."""

    WAITING = "waiting"
    ADMITTED = "admitted"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    DONE = "done"


# The one declaration. Reachable states and feasible n-grams for any n fall out
# by search, so a hand-written feasible set cannot reach 100 percent by having
# omitted the hard transitions. Checked both ways: observed must be legal, and
# every entry here must have been taken by a real run or moved below.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.WAITING, Event.ADMIT): State.ADMITTED,
    (State.ADMITTED, Event.EVICT): State.ADMITTED,
    (State.ADMITTED, Event.CACHE_HIT): State.PREFILLING,
    (State.ADMITTED, Event.CHUNK): State.PREFILLING,
    (State.ADMITTED, Event.DECODE): State.DECODING,
    (State.PREFILLING, Event.CHUNK): State.PREFILLING,
    (State.PREFILLING, Event.DECODE): State.DECODING,
    (State.PREFILLING, Event.PREEMPT_RC): State.PREEMPTED,
    (State.PREFILLING, Event.EVICT): State.PREFILLING,
    (State.DECODING, Event.DECODE): State.DECODING,
    (State.DECODING, Event.PREEMPT_RC): State.PREEMPTED,
    (State.DECODING, Event.FINISH): State.DONE,
    (State.PREEMPTED, Event.RESUME): State.ADMITTED,
}

UNREACHABLE_BY_DESIGN: dict[tuple[State, Event], str] = {
    (State.ADMITTED, Event.PREEMPT_RC): (
        "preemption fires only for a running request with kv_len != 0 and at "
        "least one generated token. A request still in ADMITTED was admitted "
        "this step, and _admit sets kv_len = 0; the only thing that raises it "
        "before any chunk is a cache hit, which emits CACHE_HIT and moves the "
        "request to PREFILLING first. So a request whose state is still ADMITTED "
        "has kv_len == 0 and the preemption loop skips it"
    ),
    (State.DECODING, Event.EVICT): (
        "a decoding request has been admitted and nothing reserves blocks again"
    ),
}

INITIAL = State.WAITING
TERMINAL = frozenset({State.DONE})


def reachable_states() -> set[State]:
    """Breadth-first from INITIAL over TRANSITIONS."""
    seen = {INITIAL}
    frontier = [INITIAL]
    while frontier:
        state = frontier.pop()
        for (source, _event), target in TRANSITIONS.items():
            if source == state and target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def events_from(state: State) -> set[Event]:
    return {event for (source, event) in TRANSITIONS if source == state}


def feasible_ngrams(n: int) -> set[tuple[Event, ...]]:
    """Every event sequence of length n realizable from a reachable state."""
    if n < 1:
        raise ValueError("n must be at least 1")

    paths: set[tuple[Event, ...]] = set()
    frontier = [(state, ()) for state in reachable_states()]
    while frontier:
        state, so_far = frontier.pop()
        if len(so_far) == n:
            paths.add(so_far)
            continue
        for event in events_from(state):
            frontier.append((TRANSITIONS[(state, event)], so_far + (event,)))
    return paths


def derivation_note() -> str:
    """How the denominator was arrived at, printed with the coverage number."""
    states = reachable_states()
    unreachable = set(State) - states
    return (
        f"derived from {len(TRANSITIONS)} declared transitions over "
        f"{len(states)} reachable states"
        + (f" ({len(unreachable)} unreachable: {sorted(s.value for s in unreachable)})"
           if unreachable else "")
        + f"; {len(Event)} events in the alphabet, "
        f"{len(list(product(State, Event)))} (state, event) pairs considered, "
        f"{len(TRANSITIONS)} legal"
    )
