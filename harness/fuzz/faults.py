"""Seeded faults, taken from the mutation operators in architecture doc 10.1.

Seeding one convenient bug validates nothing: it shows the fuzzer can find the
bug it was built around. These are the operators the mutation campaign will use
in week 6, applied now as seeded faults, so the fuzzer is validated against the
faults it will later be scored on and the misses are known early.

Each fault patches one function and is undone on exit. The misses are the more
useful half of the result: a fault nothing detects is either equivalent, which
needs a written argument, or a hole in the harness.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from engine.cache import prefix
from engine.kv import paged


@dataclass(frozen=True)
class Fault:
    name: str
    operator: str          # the architecture doc 10.1 wording
    apply: object          # contextmanager factory


@contextlib.contextmanager
def _patch(target, attribute, replacement):
    original = getattr(target, attribute)
    setattr(target, attribute, replacement)
    try:
        yield
    finally:
        setattr(target, attribute, original)


# -- allocator and KV ---------------------------------------------------------


@contextlib.contextmanager
def refcount_double_increment_on_fork():
    original = paged.PagedKVCache.fork

    def mutant(self, parent_uid, child_uid):
        sequence = original(self, parent_uid, child_uid)
        for block in sequence.block_ids:
            self.refcount[block] += 1
        return sequence

    with _patch(paged.PagedKVCache, "fork", mutant):
        yield


@contextlib.contextmanager
def refcount_decrement_missing_on_free():
    original = paged.PagedKVCache.release

    def mutant(self, uid):
        sequence = self.sequences.pop(uid)
        # The decrement is simply not done.
        sequence.block_ids.clear()
        sequence.length = 0

    with _patch(paged.PagedKVCache, "release", mutant):
        yield


@contextlib.contextmanager
def free_block_still_in_cache_index():
    """Free a block the cache index still names."""
    original = prefix.PrefixCache.evict

    def mutant(self, physical_block, pool):
        # Release the reference without removing the entry.
        pool.unpin(physical_block)
        self.stats["evictions"] += 1
        return True

    with _patch(prefix.PrefixCache, "evict", mutant):
        yield


@contextlib.contextmanager
def cow_copies_without_bumping_refcount():
    original = paged.PagedKVCache.ensure_writable

    def mutant(self, uid, logical_block):
        sequence = self.sequences[uid]
        old = sequence.block_ids[logical_block]
        if self.refcount[old] == 1:
            return old
        new = self._take_block()
        for layer in range(self.num_layers):
            self.k[layer][new].copy_(self.k[layer][old])
            self.v[layer][new].copy_(self.v[layer][old])
        sequence.block_ids[logical_block] = new
        # The old block's refcount is not decremented.
        self.stats["cow_copies"] += 1
        return new

    with _patch(paged.PagedKVCache, "ensure_writable", mutant):
        yield


@contextlib.contextmanager
def cow_bumps_refcount_without_copying():
    original = paged.PagedKVCache.ensure_writable

    def mutant(self, uid, logical_block):
        sequence = self.sequences[uid]
        old = sequence.block_ids[logical_block]
        if self.refcount[old] == 1:
            return old
        new = self._take_block()
        # The bytes are not copied.
        sequence.block_ids[logical_block] = new
        self.refcount[old] -= 1
        self.stats["cow_copies"] += 1
        return new

    with _patch(paged.PagedKVCache, "ensure_writable", mutant):
        yield


@contextlib.contextmanager
def stale_block_table_read_after_reclamation():
    """Reserve hands back a block that is still in the free pool."""
    original = paged.PagedKVCache.reserve

    def mutant(self, uid, total_tokens):
        sequence = self.sequences[uid]
        needed = -(-total_tokens // self.block_size)
        while sequence.logical_blocks() < needed:
            if self._free:
                block = min(self._free)
                self.refcount[block] += 1
                sequence.block_ids.append(block)
                self.stats["allocated"] += 1
            else:
                sequence.block_ids.append(self._take_block())

    with _patch(paged.PagedKVCache, "reserve", mutant):
        yield


# -- scheduler ----------------------------------------------------------------


@contextlib.contextmanager
def eviction_set_includes_a_running_sequence():
    original = prefix.PrefixCache.evictable_blocks

    def mutant(self, pool):
        return sorted(entry.physical_block for entry in self.entries.values())

    with _patch(prefix.PrefixCache, "evictable_blocks", mutant):
        yield


@contextlib.contextmanager
def cache_match_length_rounded_up_past_a_block():
    original = prefix.PrefixCache.lookup

    def mutant(self, tokens):
        hit_tokens, blocks = original(self, tokens)
        if blocks:
            hit_tokens += self.block_size  # claims one block more than it has
        return hit_tokens, blocks

    with _patch(prefix.PrefixCache, "lookup", mutant):
        yield


@contextlib.contextmanager
def chunk_boundary_off_by_one():
    """`<` where `<=` is correct, so one token is processed twice."""
    from engine.sched import scheduler as sched

    original = sched.Scheduler.step

    def mutant(self):
        for request in self.running:
            if request.kv_len > 0 and request.kv_len < len(request.context()):
                request.kv_len -= 1  # reprocess the last token
                break
        return original(self)

    with _patch(sched.Scheduler, "step", mutant):
        yield


@contextlib.contextmanager
def recompute_off_by_one_token_count():
    from engine.sched import scheduler as sched

    original = sched.Scheduler._preempt

    def mutant(self, request):
        original(self, request)
        request.kv_len = 1  # recompute starts one token in

    with _patch(sched.Scheduler, "_preempt", mutant):
        yield


FAULTS: tuple[Fault, ...] = (
    Fault("refcount_double_increment_on_fork",
          "refcount increment twice on fork", refcount_double_increment_on_fork),
    Fault("refcount_decrement_missing_on_free",
          "refcount decrement missing on free", refcount_decrement_missing_on_free),
    Fault("free_block_still_in_cache_index",
          "free a block still referenced by the cache index",
          free_block_still_in_cache_index),
    Fault("cow_copies_without_bumping_refcount",
          "COW fork copies without bumping refcount",
          cow_copies_without_bumping_refcount),
    Fault("cow_bumps_refcount_without_copying",
          "COW fork bumps refcount without copying",
          cow_bumps_refcount_without_copying),
    Fault("stale_block_table_read_after_reclamation",
          "stale block-table read after reclamation",
          stale_block_table_read_after_reclamation),
    Fault("eviction_set_includes_a_running_sequence",
          "eviction eligible-set includes a running sequence",
          eviction_set_includes_a_running_sequence),
    Fault("cache_match_length_rounded_up_past_a_block",
          "cache match length rounded up past a block boundary",
          cache_match_length_rounded_up_past_a_block),
    Fault("chunk_boundary_off_by_one",
          "chunk boundary uses < where <= is correct",
          chunk_boundary_off_by_one),
    Fault("recompute_off_by_one_token_count",
          "recompute re-prefills with an off-by-one token count",
          recompute_off_by_one_token_count),
)
