"""The coverage denominator is derived, and it is checked against reality.

A hand-written feasible set can reach 100 percent by having omitted the hard
transitions, and nothing in the number would show it. Two properties make the
derived one trustworthy: it comes from the declared transition relation by
search, and any n-gram a real run produces that the relation calls impossible
fails loudly rather than being silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.sched.lifecycle import (  # noqa: E402
    INITIAL,
    TRANSITIONS,
    Event,
    State,
    feasible_ngrams,
    reachable_states,
)
from harness.fuzz.coverage import Coverage, UndeclaredTransition  # noqa: E402


def test_every_state_is_reachable_from_the_initial_state():
    assert reachable_states() == set(State)


def test_the_denominator_is_smaller_than_the_naive_product():
    """If it equalled the product, no derivation happened."""
    naive = len(Event) ** 2
    assert len(feasible_ngrams(2)) < naive


def test_ngrams_grow_with_n_and_are_all_realizable():
    two, three = feasible_ngrams(2), feasible_ngrams(3)
    assert len(three) > len(two)
    # Every 3-gram's leading pair must itself be a feasible 2-gram.
    for gram in three:
        assert gram[:2] in two


def test_an_undeclared_transition_is_an_error_not_a_silent_drop():
    coverage = Coverage()
    # FINISH is terminal, so nothing may follow it.
    with pytest.raises(UndeclaredTransition, match="impossible"):
        coverage.observe_events([Event.ADMIT, Event.FINISH, Event.DECODE])


def test_a_real_sequence_is_accepted():
    coverage = Coverage()
    coverage.observe_events(
        [Event.ADMIT, Event.CACHE_HIT, Event.CHUNK, Event.DECODE,
         Event.PREEMPT_RC, Event.RESUME, Event.DECODE, Event.FINISH]
    )
    seen, total = coverage.ngram_fraction(2)
    assert 0 < seen <= total


def test_cache_hit_happens_once_immediately_after_admission():
    """The engine applies a hit during admission, at most once, so the table
    must permit admit -> cache_hit and forbid cache_hit -> cache_hit."""
    assert (State.ADMITTED, Event.CACHE_HIT) in TRANSITIONS
    assert (State.WAITING, Event.CACHE_HIT) not in TRANSITIONS
    assert (Event.ADMIT, Event.CACHE_HIT) in feasible_ngrams(2)
    assert (Event.CACHE_HIT, Event.CACHE_HIT) not in feasible_ngrams(2)
    assert (Event.RESUME, Event.CACHE_HIT) in feasible_ngrams(2)


def test_terminal_state_has_no_outgoing_transitions():
    assert not [key for key in TRANSITIONS if key[0] is State.DONE]


def test_transitions_removed_by_design_are_really_unreachable():
    """The denominator shrank when these were removed, which is the direction
    that flatters a coverage number, so the engine rules behind each removal are
    asserted here rather than trusted."""
    import inspect

    from engine.sched.lifecycle import UNREACHABLE_BY_DESIGN
    from engine.sched.scheduler import Scheduler

    for key in UNREACHABLE_BY_DESIGN:
        assert key not in TRANSITIONS

    source = inspect.getsource(Scheduler)

    # Eviction runs only inside _reserve_with_eviction, and only _admit calls it.
    assert source.count("_reserve_with_eviction(") == 2, (
        "reserve-with-eviction is called from somewhere new; a running request "
        "may now be able to observe an eviction, and (PREFILLING, EVICT) and "
        "(DECODING, EVICT) belong back in the transition table"
    )

    # Preemption is skipped for a request with no generated token.
    assert "if request.kv_len == 0 or not request.generated:" in source, (
        "the guard that keeps preemption to decoding requests has changed; "
        "(ADMITTED, PREEMPT_RC) and (PREFILLING, PREEMPT_RC) may now be reachable"
    )
