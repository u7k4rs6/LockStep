"""The coverage denominator is derived, and it is checked against reality."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.sched.lifecycle import (  # noqa: E402
    INITIAL,
    UNREACHABLE_BY_DESIGN,
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
    for gram in three:
        assert gram[:2] in two


def test_an_undeclared_transition_is_an_error_not_a_silent_drop():
    coverage = Coverage()
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
    """The engine applies a hit during admission, at most once, so the table"""
    assert (State.ADMITTED, Event.CACHE_HIT) in TRANSITIONS
    assert (State.WAITING, Event.CACHE_HIT) not in TRANSITIONS
    assert (Event.ADMIT, Event.CACHE_HIT) in feasible_ngrams(2)
    assert (Event.CACHE_HIT, Event.CACHE_HIT) not in feasible_ngrams(2)
    assert (Event.RESUME, Event.CACHE_HIT) in feasible_ngrams(2)


def test_terminal_state_has_no_outgoing_transitions():
    assert not [key for key in TRANSITIONS if key[0] is State.DONE]


def test_transitions_removed_by_design_are_really_unreachable():
    """The denominator shrank when these were removed, which is the direction"""
    import inspect

    from engine.sched.lifecycle import UNREACHABLE_BY_DESIGN
    from engine.sched.scheduler import Scheduler

    for key in UNREACHABLE_BY_DESIGN:
        assert key not in TRANSITIONS

    source = inspect.getsource(Scheduler)

    assert source.count("_reserve_with_eviction(") == 2, (
        "reserve-with-eviction is called from somewhere new; a decoding request "
        "may now be able to observe an eviction, and (DECODING, EVICT) belongs "
        "back in the transition table"
    )


def test_a_resumed_request_can_be_preempted_mid_prefill():
    """chunk -> preempt_rc is reachable, and the table must say so."""
    assert (State.PREFILLING, Event.PREEMPT_RC) in TRANSITIONS
    assert (Event.CHUNK, Event.PREEMPT_RC) in feasible_ngrams(2)

    assert (Event.RESUME, Event.PREEMPT_RC) not in feasible_ngrams(2)
    assert (State.ADMITTED, Event.PREEMPT_RC) in UNREACHABLE_BY_DESIGN


def test_a_cache_hit_can_be_followed_by_an_eviction():
    """Eviction runs inside _admit, after the hit is applied, so the engine"""
    assert (State.PREFILLING, Event.EVICT) in TRANSITIONS
    assert (Event.CACHE_HIT, Event.EVICT) in feasible_ngrams(2)


def test_every_declared_transition_has_a_witness_or_an_argument():
    """Declared must equal reachable, and neither direction may be assumed."""
    from engine.sched.lifecycle import TRANSITIONS, UNREACHABLE_BY_DESIGN

    overlap = set(TRANSITIONS) & set(UNREACHABLE_BY_DESIGN)
    assert not overlap, f"declared both legal and unreachable: {overlap}"

    for key, reason in UNREACHABLE_BY_DESIGN.items():
        assert len(reason) > 40, (
            f"{key} is excluded without an argument; a one-word reason is how a "
            "reachable transition gets quietly dropped to flatter a percentage"
        )
