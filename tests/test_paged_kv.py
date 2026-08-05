"""Paged KV: the refcount ledger, copy-on-write, and allocation determinism.

The mutation operators in architecture doc 10.1 target this module by name, so
these tests are written to be the things those mutants have to get past, not to
walk the happy path. Each one names the operator it is the counterpart to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.kv.paged import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    SUPPORTED_BLOCK_SIZES,
    AuditFailure,
    OutOfBlocks,
    PagedKVCache,
)

BLOCK_SIZE = DEFAULT_BLOCK_SIZE


def pool(num_blocks=16, layers=2, poison=True, block_size=DEFAULT_BLOCK_SIZE):
    return PagedKVCache(
        num_blocks=num_blocks,
        num_layers=layers,
        num_kv_heads=2,
        head_dim=8,
        device="cpu",
        dtype=torch.float16,
        poison_on_free=poison,
        block_size=block_size,
    )


@pytest.mark.parametrize("block_size", SUPPORTED_BLOCK_SIZES)
def test_every_supported_block_size_divides_the_kv_tile(block_size):
    """A tile that straddled a block boundary would make the gather order depend
    on layout rather than on logical position."""
    from engine.kernels import registry

    assert registry.KV_TILE % block_size == 0


def test_an_unsupported_block_size_is_refused():
    with pytest.raises(ValueError, match="does not divide the KV tile"):
        pool(block_size=48)




def test_allocation_is_lowest_free_first_and_not_lifo():
    """Allocation order must be a function of the request sequence alone.

    A LIFO free list would hand back blocks in an order that depends on the order
    things were freed, which makes the trajectory hash over allocator state
    depend on history that the schedule does not record.
    """
    p = pool()
    p.create("a")
    p.reserve("a", BLOCK_SIZE * 4)
    assert p.sequences["a"].block_ids == [0, 1, 2, 3]

    p.create("b")
    p.reserve("b", BLOCK_SIZE * 2)
    assert p.sequences["b"].block_ids == [4, 5]

    p.release("a")  # frees 0..3
    p.create("c")
    p.reserve("c", BLOCK_SIZE * 2)
    assert p.sequences["c"].block_ids == [0, 1], "free pool is not lowest-first"


def test_audit_catches_a_missing_decrement_on_free():
    p = pool()
    p.create("a")
    p.reserve("a", BLOCK_SIZE)
    block = p.sequences["a"].block_ids[0]
    p.sequences.pop("a")  # released without decrementing
    with pytest.raises(AuditFailure):
        p.audit()
    assert p.refcount[block] == 1


def test_audit_catches_two_logical_blocks_mapping_to_one_physical():
    """Block-table injectivity, architecture doc 10.2."""
    p = pool()
    p.create("a")
    p.reserve("a", BLOCK_SIZE * 2)
    p.sequences["a"].block_ids[1] = p.sequences["a"].block_ids[0]
    with pytest.raises(AuditFailure, match="two logical blocks"):
        p.audit()



def test_freed_blocks_are_poisoned():
    """Architecture doc 10.2: any read of freed memory perturbs bits
    deterministically rather than returning a plausible stale value."""
    p = pool()
    p.create("a")
    p.reserve("a", BLOCK_SIZE)
    block = p.sequences["a"].block_ids[0]
    p.k[0][block].fill_(1.0)
    p.release("a")
    assert torch.isnan(p.k[0][block]).all()


def test_exhaustion_is_an_error_not_a_silent_overwrite():
    p = pool(num_blocks=2)
    p.create("a")
    with pytest.raises(OutOfBlocks):
        p.reserve("a", BLOCK_SIZE * 3)



def test_state_digest_is_stable_and_sensitive():
    p = pool()
    p.create("a")
    p.reserve("a", BLOCK_SIZE)
    before = p.state_digest()
    assert before == p.state_digest(), "digest is not stable for identical state"
    p.create("b")
    p.reserve("b", BLOCK_SIZE)
    assert p.state_digest() != before, "digest missed a new sequence"
