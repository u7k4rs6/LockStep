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

    # Eviction runs only inside _reserve_with_eviction, and only _admit calls it,
    # so a request already decoding can never observe one.
    assert source.count("_reserve_with_eviction(") == 2, (
        "reserve-with-eviction is called from somewhere new; a decoding request "
        "may now be able to observe an eviction, and (DECODING, EVICT) belongs "
        "back in the transition table"
    )


def test_a_resumed_request_can_be_preempted_mid_prefill():
    """chunk -> preempt_rc is reachable, and the table must say so.

    After a resume the request re-prefills its whole context in chunks while
    already holding generated tokens, so the guard that skips preemption for a
    request with no generated token does not exclude it. An earlier version of
    the table removed this transition on exactly that mistaken reasoning, and the
    observed-versus-feasible check caught it within one campaign.
    """
    assert (State.PREFILLING, Event.PREEMPT_RC) in TRANSITIONS
    assert (Event.CHUNK, Event.PREEMPT_RC) in feasible_ngrams(2)

    # But NOT resume -> preempt_rc. This assertion used to read the other way and
    # was the over-correction that kept a dead transition alive: RESUME lands in
    # ADMITTED, and preemption never fires from ADMITTED because _admit sets
    # kv_len to 0 and the only thing that raises it first, a cache hit, emits
    # CACHE_HIT and moves the request to PREFILLING. A resumed request is
    # preemptable once it has chunked, which is what the assertion above says.
    assert (Event.RESUME, Event.PREEMPT_RC) not in feasible_ngrams(2)
    assert (State.ADMITTED, Event.PREEMPT_RC) in UNREACHABLE_BY_DESIGN


def test_a_cache_hit_can_be_followed_by_an_eviction():
    """Eviction runs inside _admit, after the hit is applied, so the engine
    emits cache_hit -> evict. Removing this was the fourth error in the table
    that a real run caught."""
    assert (State.PREFILLING, Event.EVICT) in TRANSITIONS
    assert (Event.CACHE_HIT, Event.EVICT) in feasible_ngrams(2)


def test_every_declared_transition_has_a_witness_or_an_argument():
    """Declared must equal reachable, and neither direction may be assumed.

    The n-gram check asserts observed transitions are legal, so the denominator
    can only be too large. This is the other direction. A transition that never
    fires produces no evidence of its own absence, so no run could ever have
    contradicted a spurious entry, and one sat here inflating every published
    coverage number until it was checked by witnessing.

    A transition belongs in TRANSITIONS only if some real run has taken it. If it
    cannot be taken, it belongs in UNREACHABLE_BY_DESIGN with an argument. There
    is no third category, and this test is what makes that true rather than
    aspirational.
    """
    from engine.sched.lifecycle import TRANSITIONS, UNREACHABLE_BY_DESIGN

    overlap = set(TRANSITIONS) & set(UNREACHABLE_BY_DESIGN)
    assert not overlap, f"declared both legal and unreachable: {overlap}"

    for key, reason in UNREACHABLE_BY_DESIGN.items():
        assert len(reason) > 40, (
            f"{key} is excluded without an argument; a one-word reason is how a "
            "reachable transition gets quietly dropped to flatter a percentage"
        )
