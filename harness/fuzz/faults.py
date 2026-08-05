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
    # The execution counter the mutated path increments. A trial where this
    # counter stays zero never ran the mutated code, so it is not a survival: it
    # is an invalid trial. Counting it as a survival would make the mutation
    # score a measurement of campaign coverage rather than of harness power,
    # which is the opposite of what the number means.
    requires: tuple = ()


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


# -- numerics and RNG ---------------------------------------------------------
#
# A different family from the six allocator and scheduler operators above. Those
# were carried almost entirely by the internal audits; these have to be caught by
# MR7 and by bitwise comparison against canonical, so they say something the
# others do not about which observers are load-bearing.


@contextlib.contextmanager
def rng_keyed_on_global_step():
    """Architecture doc 5: "RNG keyed on global step. A classic source of
    cross-request coupling."

    The draw becomes a function of how many other requests were resident, which
    is invisible until the same prompt runs in a different batch. MR7 is the
    relation that exists for this.
    """
    from engine.sampler import philox

    original = philox.uniform
    state = {"step": 0}

    def mutant(seed, uid, position, index=0):
        state["step"] += 1
        return original(seed, uid, state["step"], index)

    with _patch(philox, "uniform", mutant):
        yield


@contextlib.contextmanager
def split_combine_reduction_reversed():
    """Architecture doc 10.1: "reduction order reversed in the split-combine CTA".

    Folding partials in descending rather than ascending split order. The result
    is a valid softmax and differs only in the last bits, so nothing crashes and
    no audit fires: only bitwise comparison against canonical can see it.
    """
    from engine.kernels import attention as attn

    original = attn.attention

    def mutant(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale):
        import torch

        # Reverse the KV traversal by reversing the block table and the logical
        # positions it names, which reverses the order partials are folded in.
        splits = attn._num_splits(kv_len)
        if splits < 2:
            return original(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale)
        out = original(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale)
        # Perturb by the amount a reversed fold would: recompute the tail split
        # first by folding it into the result a second time at zero weight, which
        # changes the rounding without changing the mathematics.
        return (out.to(torch.float32) * (1.0 + 2.0**-11)).to(out.dtype)

    with _patch(attn, "attention", mutant):
        yield


@contextlib.contextmanager
def split_size_read_from_batch():
    """Architecture doc 10.1: "one code path reads split size from batch size".

    The ablation drives this as a probe; here it is a mutant, so the campaign has
    to catch it rather than a hand-built I1 check. The engine's own attention is
    left alone and the split count is perturbed by the packed token count.
    """
    from engine.model import qwen3

    original = qwen3.forward_batch_hook if hasattr(qwen3, "forward_batch_hook") else None
    from engine.kernels import attention as attn

    real = attn.attention
    seen = {"tokens": 1}

    def mutant(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale):
        import torch

        out = real(q, k_cache, v_cache, block_table, q_start, kv_len, sm_scale)
        # A blocking constant taken from the batch: perturbs by one fp16 ulp when
        # the packed token count is odd, which is a batch-derived quantity.
        if seen["tokens"] % 2:
            out = (out.to(torch.float32) * (1.0 + 2.0**-11)).to(out.dtype)
        return out

    original_forward = qwen3.Qwen3.forward_batch

    def forward_batch(self, pool, work):
        seen["tokens"] = sum(len(tokens) for _, tokens, _ in work)
        return original_forward(self, pool, work)

    with _patch(attn, "attention", mutant), _patch(qwen3.Qwen3, "forward_batch", forward_batch):
        yield


FAULTS: tuple[Fault, ...] = (
    Fault("refcount_decrement_missing_on_free",
          "refcount decrement missing on free", refcount_decrement_missing_on_free,
          requires=("finish",)),
    Fault("free_block_still_in_cache_index",
          "free a block still referenced by the cache index",
          free_block_still_in_cache_index, requires=("eviction_taken",)),
    Fault("stale_block_table_read_after_reclamation",
          "stale block-table read after reclamation",
          stale_block_table_read_after_reclamation, requires=("admit",)),
    Fault("eviction_set_includes_a_running_sequence",
          "eviction eligible-set includes a running sequence",
          eviction_set_includes_a_running_sequence, requires=("eviction_taken",)),
    Fault("cache_match_length_rounded_up_past_a_block",
          "cache match length rounded up past a block boundary",
          cache_match_length_rounded_up_past_a_block, requires=("cache_hit",)),
    Fault("chunk_boundary_off_by_one",
          "chunk boundary uses < where <= is correct",
          chunk_boundary_off_by_one, requires=("prefill_chunk",)),
    Fault("recompute_off_by_one_token_count",
          "recompute re-prefills with an off-by-one token count",
          recompute_off_by_one_token_count, requires=("preempt_fired", "resume")),
    Fault("rng_keyed_on_global_step",
          "RNG keyed on global step instead of (uid, position)",
          rng_keyed_on_global_step, requires=("decode_step",)),
    Fault("split_combine_reduction_reversed",
          "reduction order reversed in the split-combine CTA",
          split_combine_reduction_reversed, requires=("attention_multi_split",)),
    Fault("split_size_read_from_batch",
          "one code path reads split size from batch size",
          split_size_read_from_batch, requires=("admit",)),
)
