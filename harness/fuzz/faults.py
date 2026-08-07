"""Seeded faults, taken from the mutation operators in architecture doc 10.1."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import triton
import triton.language as tl

from engine.cache import prefix
from engine.kernels.attention import NEG_SENTINEL as _NEG_SENTINEL
from engine.kv import paged


@dataclass(frozen=True)
class Fault:
    name: str
    operator: str
    apply: object
    requires: tuple = ()
    fidelity_observable: bool = False


SENTINELS: dict[str, int] = {}


def trip(name: str) -> None:
    SENTINELS[name] = SENTINELS.get(name, 0) + 1


@contextlib.contextmanager
def _patch(target, attribute, replacement):
    original = getattr(target, attribute)
    setattr(target, attribute, replacement)
    try:
        yield
    finally:
        setattr(target, attribute, original)


@contextlib.contextmanager
def refcount_decrement_missing_on_free():
    original = paged.PagedKVCache.release

    def mutant(self, uid):
        sequence = self.sequences.pop(uid)
        trip("refcount_decrement_missing_on_free")
        sequence.block_ids.clear()
        sequence.length = 0

    with _patch(paged.PagedKVCache, "release", mutant):
        yield


@contextlib.contextmanager
def free_block_still_in_cache_index():
    """Free a block the cache index still names."""
    original = prefix.PrefixCache.evict

    def mutant(self, physical_block, pool):
        trip("free_block_still_in_cache_index")
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
        trip("stale_block_table_read_after_reclamation")
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


@contextlib.contextmanager
def eviction_set_includes_a_running_sequence():
    original = prefix.PrefixCache.evictable_blocks

    def mutant(self, pool):
        trip("eviction_set_includes_a_running_sequence")
        return sorted(entry.physical_block for entry in self.entries.values())

    with _patch(prefix.PrefixCache, "evictable_blocks", mutant):
        yield


@contextlib.contextmanager
def cache_match_length_rounded_up_past_a_block():
    original = prefix.PrefixCache.lookup

    def mutant(self, tokens):
        trip("cache_match_length_rounded_up_past_a_block")
        hit_tokens, blocks = original(self, tokens)
        if blocks:
            hit_tokens += self.block_size
        return hit_tokens, blocks

    with _patch(prefix.PrefixCache, "lookup", mutant):
        yield


@contextlib.contextmanager
def chunk_boundary_off_by_one():
    """`<` where `<=` is correct, so one token is processed twice."""
    from engine.sched import scheduler as sched

    original = sched.Scheduler.step

    def mutant(self):
        trip("chunk_boundary_off_by_one")
        for request in self.running:
            if request.kv_len > 0 and request.kv_len < len(request.context()):
                request.kv_len -= 1
                break
        return original(self)

    with _patch(sched.Scheduler, "step", mutant):
        yield


@contextlib.contextmanager
def recompute_off_by_one_token_count():
    """Architecture doc 10.1: "recompute re-prefills with an off-by-one token count"."""
    from engine.sched import scheduler as sched

    original = sched.Scheduler._admit

    def mutant(self, request, state):
        original(self, request, state)
        if request.preempt_count:
            trip("recompute_off_by_one_token_count")
            request.kv_len += 1
            self.pool.sequences[request.uid].length = request.kv_len

    with _patch(sched.Scheduler, "_admit", mutant):
        yield


@contextlib.contextmanager
def rng_keyed_on_global_step():
    """Architecture doc 5: "RNG keyed on global step. A classic source of"""
    from engine.sampler import philox

    original = philox.uniform
    state = {"step": 0}

    def mutant(seed, uid, position, index=0):
        trip("rng_keyed_on_global_step")
        state["step"] += 1
        return original(seed, uid, state["step"], index)

    with _patch(philox, "uniform", mutant):
        yield


@contextlib.contextmanager
def split_combine_reduction_reversed():
    """Architecture doc 10.1: "reduction order reversed in the split-combine CTA"."""
    from engine.kernels import attention as attn

    original_launch = attn._attn_combine_kernel

    @triton.jit
    def _reversed_combine(
        Acc, MPart, LPart, Out, q_len, num_splits,
        stride_at, stride_ah, stride_as, stride_ad,
        stride_mt, stride_mh, stride_ms,
        stride_ot, stride_oh, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_valid = offs_m < q_len

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_run = tl.full((BLOCK_M,), _NEG_SENTINEL, dtype=tl.float32)
        l_run = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for s in range(num_splits - 1, -1, -1):
            part_offset = offs_m * stride_mt + pid_h * stride_mh + s * stride_ms
            m_s = tl.load(MPart + part_offset, mask=m_valid, other=_NEG_SENTINEL)
            l_s = tl.load(LPart + part_offset, mask=m_valid, other=0.0)
            a_s = tl.load(
                Acc + offs_m[:, None] * stride_at + pid_h * stride_ah
                + s * stride_as + offs_d[None, :] * stride_ad,
                mask=m_valid[:, None], other=0.0,
            )
            m_new = tl.maximum(m_run, m_s)
            alpha = tl.exp(m_run - m_new)
            beta = tl.exp(m_s - m_new)
            acc = acc * alpha[:, None] + a_s * beta[:, None]
            l_run = l_run * alpha + l_s * beta
            m_run = m_new

        out = tl.where(l_run[:, None] > 0.0, acc / l_run[:, None], 0.0)
        tl.store(
            Out + offs_m[:, None] * stride_ot + pid_h * stride_oh
            + offs_d[None, :] * stride_od,
            out.to(Out.dtype.element_ty), mask=m_valid[:, None],
        )

    class _Sentinelled:
        """Trips the sentinel on launch, since `trip` cannot run inside a kernel."""

        def __getitem__(self, grid):
            inner = _reversed_combine[grid]

            def launch(*args, **kwargs):
                trip("split_combine_reduction_reversed")
                return inner(*args, **kwargs)

            return launch

    assert original_launch is not None
    with _patch(attn, "_attn_combine_kernel", _Sentinelled()):
        yield


@contextlib.contextmanager
def split_size_read_from_batch():
    """Architecture doc 10.1: "one code path reads split size from batch size"."""
    from engine.kernels import attention as attn
    from engine.model import qwen3
    from harness.mr import ablation

    original_forward = qwen3.Qwen3.forward_batch

    def forward_batch(self, pool, work):
        trip("split_size_read_from_batch")
        ablation._BATCH_TOKENS = sum(len(tokens) for _, tokens, _ in work)
        return original_forward(self, pool, work)

    with _patch(attn, "attention", ablation.batch_derived_split_attention), \
         _patch(qwen3, "attention", ablation.batch_derived_split_attention), \
         _patch(qwen3.Qwen3, "forward_batch", forward_batch):
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
          split_combine_reduction_reversed, requires=("attention_multi_split",),
          fidelity_observable=True),
    Fault("split_size_read_from_batch",
          "one code path reads split size from batch size",
          split_size_read_from_batch, requires=("admit",)),
)
