"""The request lifecycle as a state machine, and the coverage denominator.

docs/02-technical-architecture.md section 8.2 asks for "All feasible 2-grams and
3-grams of per-request events, reported as a percentage of the feasible set."

The feasible set is *derived* here, not written down. Hand-listing feasible
2-grams lets a campaign reach 100 percent by having omitted the hard ones, and
neither the author nor a reader can tell from the number. What is declared here
is one thing, the transition relation: for each state, which events are legal and
where each leads. Everything else, the reachable states and the feasible n-grams
for any n, falls out by search.

The declaration is checked against reality rather than trusted. `Coverage`
asserts that every n-gram actually observed during a campaign is in the derived
feasible set, so a transition the table forgot cannot pass unnoticed: observing
it fails loudly. A denominator that can only be too *large* is a denominator that
cannot flatter the numerator.

The states are the ones the scheduler actually distinguishes, and each is named
by what the engine can do next from it rather than by an internal field.
"""

from __future__ import annotations

from enum import Enum
from itertools import product

from engine.sched.policy import Event


class State(str, Enum):
    """Where a request is in its life. Terminal states have no outgoing events."""

    WAITING = "waiting"  # submitted, not yet admitted
    ADMITTED = "admitted"  # admitted this step, cache hit not yet applied
    PREFILLING = "prefilling"  # prompt not fully in the pool
    DECODING = "decoding"  # prompt complete, emitting tokens
    PREEMPTED = "preempted"  # blocks released, tokens kept, awaiting re-admission
    DONE = "done"  # finished and released


# The one declaration. (state, event) -> next state. Everything else is derived.
#
# Read it as the engine's own rules: a waiting request can only be admitted or
# have a cache hit applied on admission; a prefilling request chunks until its
# prompt is in, then decodes; a decoding request decodes, is preempted, or
# finishes; a preempted request can only be re-admitted. Eviction is a
# pool-level event that a request observes without changing its own state.
# ADMITTED exists to model "a cache hit happens at most once, immediately after
# admission". Two earlier versions were wrong in opposite directions: putting
# CACHE_HIT on WAITING let cache_hit -> cache_hit into the denominator and left
# admit -> cache_hit out, and a self-loop on PREFILLING still admitted the
# repeat. Both would only have inflated the denominator, but a denominator that
# contains sequences the engine cannot produce reports coverage that can never
# reach 100 percent for reasons unrelated to exploration.
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

# Four transitions were removed after auditing this table against the engine
# rather than against intuition, and each removal names the rule that forbids it.
# The denominator shrank as a result, which is the direction that flatters a
# coverage number, so the reasoning is written down rather than asserted:
#
#   (PREFILLING, EVICT) and (DECODING, EVICT)
#     Blocks are reserved once, at admission, for the whole of
#     len(prompt) + max_new_tokens. Eviction only ever runs inside
#     _reserve_with_eviction, which only _admit calls, so a running request can
#     never observe an eviction. If allocation ever becomes incremental, which
#     is the more realistic design and where more interesting bugs live, these
#     come back and the denominator grows again.
#
# (ADMITTED, PREEMPT_RC) and (PREFILLING, PREEMPT_RC) were also removed, on the
# reasoning that preemption requires a generated token so a request can only be
# preempted while decoding. That was wrong, and the observed-versus-feasible
# check caught it within one campaign: after a resume, a request re-prefills its
# whole context in chunks *while already holding generated tokens*, so it is
# preemptable mid-prefill and the engine emits chunk -> preempt_rc. Both are
# restored.
#
# Worth recording rather than quietly fixing. The check has now caught three
# errors in this table, in both directions: CACHE_HIT placed on the wrong state
# inflating the denominator, EVICT transitions the engine cannot take inflating
# it further, and these two removals deflating it. A denominator nobody can check
# is worth less than no denominator at all.
#
# tests/test_lifecycle_coverage.py asserts the remaining two stay unreachable.
# (PREFILLING, EVICT) was also restored: eviction runs inside
# _reserve_with_eviction, which _admit calls *after* applying a cache hit, so the
# engine emits cache_hit -> evict. Only (DECODING, EVICT) is genuinely
# unreachable, because a request in DECODING has already been admitted and
# nothing reserves blocks again.
#
# Five errors in this table now. The fifth is different in kind from the other
# four and is the reason `Coverage.observe_transitions` exists.
#
# The n-gram check is one-directional: it asserts that every observed n-gram is
# legal, so the denominator can only ever be too large. Nothing asserted the
# converse, that every declared transition is reachable, and (ADMITTED,
# PREEMPT_RC) was not. It sat here inflating the denominator through every
# coverage number this project published, and no run could have contradicted it,
# because a transition that never fires produces no evidence of its own absence.
#
# It was found by witnessing instead: every declared transition must be taken by
# some real run, or moved here with an argument. Two others were unwitnessed in
# the first campaign, (PREFILLING, EVICT) and (PREFILLING, PREEMPT_RC), and a
# campaign built to target them witnessed both, so they are rare rather than
# dead. This one survived that campaign, and the engine argument above says why.
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
    """Every event sequence of length n realizable from a reachable state.

    Derived by walking the transition relation, so adding a state or an event to
    TRANSITIONS changes the denominator automatically and no second list needs
    editing.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    paths: set[tuple[Event, ...]] = set()
    # (state, events so far)
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
