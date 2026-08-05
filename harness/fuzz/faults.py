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

import triton
import triton.language as tl

from engine.cache import prefix
# Triton resolves a kernel's free names from the *defining* module's globals, so
# the reversed-combine mutant below cannot close over a local alias. Bound here
# from the real constant rather than retyped, so the two cannot drift.
from engine.kernels.attention import NEG_SENTINEL as _NEG_SENTINEL
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
    # Some faults are invisible to I1 through I4 by construction and only the
    # fidelity relation can see them. Reversing the split-combine fold is the
    # clear case: the split count is a function of the request's own KV length,
    # identical in canonical and batched execution, so both sides are perturbed
    # the same way and no invariance relation can distinguish them. F1 compares
    # against fp64 and does not care about batching at all.
    fidelity_observable: bool = False


# Incremented by a mutant's own body when it actually executes. The execution
# counters prove the *path* ran; they cannot prove the *patch* took effect, and
# the difference is not theoretical: a fault that patched only the defining
# module left engine/model/qwen3.py bound to the name it had imported, so the
# mutation never ran while the counter gate still passed and the trial was scored
# as a survivor. Every mutant now trips a sentinel, checked separately.
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


# -- allocator and KV ---------------------------------------------------------


@contextlib.contextmanager
def refcount_decrement_missing_on_free():
    original = paged.PagedKVCache.release

    def mutant(self, uid):
        sequence = self.sequences.pop(uid)
        trip("refcount_decrement_missing_on_free")
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
        trip("free_block_still_in_cache_index")
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


# -- scheduler ----------------------------------------------------------------


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
        trip("chunk_boundary_off_by_one")
        for request in self.running:
            if request.kv_len > 0 and request.kv_len < len(request.context()):
                request.kv_len -= 1  # reprocess the last token
                break
        return original(self)

    with _patch(sched.Scheduler, "step", mutant):
        yield


@contextlib.contextmanager
def recompute_off_by_one_token_count():
    """Architecture doc 10.1: "recompute re-prefills with an off-by-one token count".

    The first version of this operator patched `_preempt` and set `kv_len = 1`
    after it. That was a **dead store**, and it was published for weeks as a
    proven-equivalent mutant with an equivalence argument describing a mechanism
    that does not exist. `_preempt` sets `kv_len = 0` and moves the request to
    the waiting queue; the next read of `kv_len` is in `_admit`, which
    unconditionally resets it to 0 before anything else looks at it. Nothing in
    between reads the field. So the fault never reached engine state, no observer
    could have seen it, and calling that an equivalent mutant confused "cannot be
    distinguished" with "was never injected".

    The methodological point generalizes, and is why this comment is long. The
    sentinel fired 16 times on the dead version: the *patch* executed. The
    *fault* was clobbered one line later. Patch-executed is not fault-injected,
    exactly as counter-fired is not patch-executed. Each check catches the layer
    below it and is blind to the layer above.

    So the injection point is now where the value survives: after `_admit` has
    finished setting up a resumed request, so the re-prefill genuinely starts one
    token in and position `kv_len` never gets its KV written. Attention then
    reads whatever occupies that slot, which under `poison_on_free` is NaN and
    otherwise is a stale block.
    """
    from engine.sched import scheduler as sched

    original = sched.Scheduler._admit

    def mutant(self, request, state):
        original(self, request, state)
        # Only on the resume path: this operator models recompute, and a fresh
        # admission is not a recompute. `preempt_count` is what distinguishes
        # them, and it is the same predicate `step` uses to emit RESUME.
        if request.preempt_count:
            trip("recompute_off_by_one_token_count")
            request.kv_len += 1
            self.pool.sequences[request.uid].length = request.kv_len

    with _patch(sched.Scheduler, "_admit", mutant):
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
        trip("rng_keyed_on_global_step")
        state["step"] += 1
        return original(seed, uid, state["step"], index)

    with _patch(philox, "uniform", mutant):
        yield


@contextlib.contextmanager
def split_combine_reduction_reversed():
    """Architecture doc 10.1: "reduction order reversed in the split-combine CTA".

    An earlier version of this operator was a proxy: it scaled the output by
    (1 + 2**-11) whenever the split count was at least two, on the theory that a
    reversed fold moves the last bits by about that much. The campaign killed it
    by bitwise divergence and the kill was worthless, because the *firing
    condition* was schedule-dependent. A request prefilled in chunks reaches a
    given position through a different sequence of kv_len values than the same
    request prefilled whole, so the proxy fired on a different set of calls in
    the two runs and the relations saw a difference that a real reversed fold
    would never produce. That is a mutant killed for the wrong reason, which
    inflates a mutation score exactly as badly as a mutant that never ran.

    This is the real thing: the combine kernel with its fold loop running
    descending instead of ascending, and nothing else changed. The fold order for
    a given kv_len is then identical no matter how the request was scheduled,
    which is what makes this operator worth having. It is the one perturbation in
    this set that the invariance relations *should* miss.
    """
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

        # The single mutated line. Everything below it is the clean kernel.
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

    # `attention` resolves `_attn_combine_kernel` from this module's globals at
    # call time, so there is only one binding to patch here. That is not true of
    # every operator in this file, which is why the sentinel exists.
    assert original_launch is not None
    with _patch(attn, "_attn_combine_kernel", _Sentinelled()):
        yield


@contextlib.contextmanager
def split_size_read_from_batch():
    """Architecture doc 10.1: "one code path reads split size from batch size".

    This reuses the ablation's `batch_derived_split_attention`, which is a
    faithful online-softmax fold whose only defect is that its blocking constant
    comes from the packed token count, and which the ablation table already shows
    turns I1 red. An earlier version of this operator applied a scalar to the
    attention output gated on token-count parity, which perturbed canonical and
    batched execution identically often enough to survive. That was a badly
    constructed mutant rather than a harness gap, and reusing the probe that is
    known to work converts an invalid trial into a real one.
    """
    from engine.kernels import attention as attn
    from engine.model import qwen3
    from harness.mr import ablation

    original_forward = qwen3.Qwen3.forward_batch

    def forward_batch(self, pool, work):
        trip("split_size_read_from_batch")
        # The probe reads this module-level value, which is the batch-derived
        # quantity the operator is about.
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
