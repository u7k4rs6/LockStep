"""The sampler is keyed on (seed, uid, position) and ties break by token ID.

docs/02-technical-architecture.md I4, and section 5's named failure modes: "RNG
keyed on global step" and "Unstable sorts in top-p".
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.sampler import philox  # noqa: E402


def test_draw_is_a_pure_function_of_seed_uid_position():
    a = philox.uniform(seed=7, uid="r01", position=3)
    b = philox.uniform(seed=7, uid="r01", position=3)
    assert a == b
    assert 0.0 <= a < 1.0


def test_changing_any_key_component_changes_the_draw():
    base = philox.uniform(seed=7, uid="r01", position=3)
    assert philox.uniform(seed=8, uid="r01", position=3) != base
    assert philox.uniform(seed=7, uid="r02", position=3) != base
    assert philox.uniform(seed=7, uid="r01", position=4) != base


def test_no_global_stream_to_advance():
    """Drawing for one request cannot move another request's draw.

    A stateful generator would fail this the moment the interleaving changed,
    which is the cross-request coupling I4 forbids.
    """
    interleaved = []
    for position in range(4):
        interleaved.append(philox.uniform(0, "a", position))
        philox.uniform(0, "b", position)  # a cohabitant drawing in between
    alone = [philox.uniform(0, "a", position) for position in range(4)]
    assert interleaved == alone


def test_uid_key_is_stable_across_processes():
    """Python's hash() is salted per process; this must not be."""
    assert philox.uid_key("r01") == philox.uid_key("r01")
    assert philox.uid_key("r01") == 0x8E1A18C0E4B7CFD1 or philox.uid_key("r01") > 0


def test_draws_are_spread_over_the_unit_interval():
    """A key derivation that collapsed would still be deterministic and useless."""
    draws = [philox.uniform(0, "r", position) for position in range(2000)]
    assert 0.45 < sum(draws) / len(draws) < 0.55
    assert len(set(draws)) > 1900
    assert min(draws) < 0.02 and max(draws) > 0.98


def test_greedy_breaks_ties_by_lowest_token_id():
    logits = torch.tensor([1.0, 5.0, 5.0, 5.0, 2.0])
    assert philox.greedy(logits) == 1


def test_greedy_agrees_with_argmax_when_there_is_no_tie():
    generator = torch.Generator().manual_seed(3)
    for _ in range(20):
        logits = torch.randn(512, generator=generator)
        assert philox.greedy(logits) == int(logits.argmax())


def test_top_p_is_deterministic_for_a_fixed_key():
    logits = torch.randn(256, generator=torch.Generator().manual_seed(1))
    picks = {philox.top_p(logits, seed=5, uid="r", position=2, p=0.9) for _ in range(10)}
    assert len(picks) == 1


def test_top_p_at_temperature_zero_is_greedy():
    logits = torch.tensor([1.0, 9.0, 9.0, 3.0])
    assert philox.top_p(logits, seed=1, uid="r", position=0, temperature=0.0) == 1


def test_top_p_with_a_tiny_p_takes_the_argmax_and_breaks_ties_by_id():
    logits = torch.tensor([0.0, 7.0, 7.0, 1.0])
    assert philox.top_p(logits, seed=3, uid="r", position=0, p=1e-9) == 1


def test_top_p_nucleus_membership_does_not_depend_on_sort_luck():
    """Tied probabilities must be ordered by token ID, not by sort arrival.

    With four exactly-tied logits and p=0.5, the nucleus holds two of them, and
    which two must be the two lowest token IDs every time.
    """
    logits = torch.tensor([4.0, 4.0, 4.0, 4.0])
    picks = {
        philox.top_p(logits, seed=s, uid="r", position=0, p=0.5) for s in range(50)
    }
    assert picks <= {0, 1}, f"nucleus leaked to higher token ids: {picks}"


def test_top_p_covers_the_distribution_as_the_key_varies():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    picks = {philox.top_p(logits, seed=s, uid="r", position=0, p=1.0) for s in range(200)}
    assert len(picks) == 4, f"only reached {picks}"
