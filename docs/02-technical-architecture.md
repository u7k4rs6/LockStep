# Lockstep: Technical Architecture

Version 0.1, August 2026. Read alongside `01-PRD.md`.

## 1. Target environment

Pinned and recorded in `env.lock`, emitted into every result artifact. No claim in this project is portable across this tuple.

- GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM, sm_89
- CPU: Intel Core i5, 13th gen. No AVX-512 on consumer 13th gen; the fp64 reference is a cached one-time pass, not a hot path
- OS: Linux, zsh. `python3` only, no bare `python` on PATH. uv for env management
- Stack: PyTorch, Triton, single GPU, single process
- Model: Qwen3-0.6B fp16 primary. 28 layers, 8 KV heads, head_dim 128, so roughly 112 KB of KV per token in fp16. Qwen3-1.7B only if the KV budget survives

Optional: a few rented A100 or H100 hours, used only for the final throughput table.

## 2. Definitions

**Request** `r = (prompt x, sampling params s, seed k, uid)`.

**Canonical execution** `C(r)`: batch size 1, single uninterrupted prefill, cold cache, no speculation. This is the reference for all invariance claims. It is not an external oracle; invariance is an internal-consistency property and needs no third party.

**Schedule** `sigma`: the sequence of scheduler decisions over a workload `W`. Alphabet per request: `admit`, `chunk(a,b)`, `decode`, `preempt_rc`, `resume`, `evict(blk)`, `cache_hit(len)`, `finish`. Plus a global memory-pressure state.

**Bit-identical**: identical emitted token IDs and identical raw fp16 logit bytes at every sampled position. Debug builds additionally hash per-layer hidden states for divergence localization.

## 3. Invariants

These are the README's claims. They must stay precise enough to be falsified.

| ID | Statement |
|---|---|
| I1 | Batch invariance. For any set of cohabitant requests, `r`'s output is bit-identical to `C(r)` |
| I2 | Schedule invariance. For any valid `sigma`, including preemption-with-recompute at any decode step, any chunk partition of prefill, eviction under pressure, and prefix-cache hit versus miss, `r`'s output is bit-identical to `C(r)` |
| I3 | Replay determinism. Identical `(W, sigma, seeds)` twice yields an identical trajectory hash over all engine state |
| I4 | RNG isolation. `r`'s sampled tokens are a function only of `(k, r.uid, position)` and `r`'s logits. Adding or removing any other request never changes `r`'s draws |
| F1 | Fidelity, tolerance-based, deliberately separate from invariance. Batch-1 logits versus an fp64 CPU reference within stated bounds |

I5 and I6 (speculative decoding equivalence) are out of scope per the PRD and are retained here only as vocabulary for interviews.

**The sufficient condition, one sentence, memorize it:** every floating-point reduction's blocking constants and traversal order are functions only of compile-time constants and the request's own (position, KV length); no atomics on accumulators; no batch-derived quantity (batch size, batch-max sequence length, total token count) reaches any kernel config or split decision; RNG per I4; autotuning frozen.

## 4. Why each invariant holds, and where it breaks

### 4.1 Preemption

Swap is trivially bit-preserving: a memcpy of KV pages. Recompute is the hard case. The original KV for generated tokens was produced by sequential decode steps; recompute reproduces it as a single prefill pass. So recompute correctness reduces to decode-versus-prefill equivalence per token, which is exactly chunk invariance. If chunk invariance holds, arbitrary preemption points are free.

### 4.2 Chunked prefill

Breaks precisely when a kernel specializes K-dimension blocking or split selection by M (the query-row count) or by any batch-wide maximum. Rows are independent, so M-tiling may vary freely; the K-loop and the KV traversal may not.

Minimal design that preserves it:
- One attention kernel family. Constant KV tile (128). Fixed split size (512), never a fixed split count. Partials combined by a single CTA in ascending split order, accumulated in fp32
- Triton GEMMs with fixed `BLOCK_K` and no split-K, including the roughly 151k-vocab lm_head, which is the most tempting place to reach for split-K and the worst place to do it
- RMSNorm and softmax as one CTA per row
- RoPE elementwise

This yields invariance under arbitrary chunk boundaries, which is strictly stronger than what recompute needs and stronger than what shipped engines test, since they test fixed chunk configurations.

### 4.3 Prefix cache

A cache hit returns stored bytes. Hit-versus-miss is therefore bit-identical if and only if chunk invariance holds, because the cached bits equal what recomputation would produce. Without chunk invariance, caching changes numerics. This is the exact class of defect to hunt in external engines: SGLang ships a radix-cache consistency test mode and still has an open block-boundary corruption bug in deterministic mode.

## 5. Common failure modes to design against

Named here so they become CI checks rather than debugging sessions:

- **Split-K.** Not something to "fix". cuBLAS split-K combines via atomics or a heuristically chosen reduce pass. The design is no split-K at all
- **Fixed split count.** Reintroduces batch dependence if derived from batch-max sequence length. Fixed split size only
- **Triton autotuning.** Timing-based, so the same binary can pick different configs across runs. Every config is pinned in a committed registry. CI greps for `tl.atomic_` in any kernel under claim and for autotune decorators
- **Host syncs in the decode loop.** Not a correctness issue but the main throughput killer at this model size, which is launch-overhead-bound rather than occupancy-bound
- **RNG keyed on global step.** A classic source of cross-request coupling. Counter-based Philox keyed on `(seed, uid, position)` instead
- **Unstable sorts** in top-p. Tie-break by token ID

## 6. Component map

```
lockstep/
  engine/
    model/          Qwen3 forward pass, fp16, invariant ops only
    kernels/        Triton: attention, gemm, rmsnorm, softmax, rope
      registry.py   Pinned configs. No autotune. Committed
    kv/             Block table, refcounts, COW fork, allocator
    cache/          Exact-prefix index, block-aligned, hash-keyed
    sched/
      policy.py     Decision interface. THE seam the fuzzer drives
      default.py    Production heuristic policy
      replay.py     Replays a recorded sigma exactly
    sampler/        Philox counter-based, stable tie-break
    audit/          Internal invariant checks, always on in debug
  harness/
    sim/            Deterministic simulation driver
    fuzz/           Generators, swarm configs, coverage tracking
    minimize/       ddmin over requests, events, tokens
    mr/             Metamorphic relations
    mutate/         Mutation operators, campaign runner, scoring
  certify/          Black-box MR suite for OpenAI-compatible endpoints
  bench/            Throughput and cost-of-invariance scripts
  report/           HTML report generator (see frontend spec)
```

The single most important architectural decision: **`sched/policy.py` is the seam**. Every scheduler decision (admit, chunk boundary, preempt or not, which block to evict, whether to honor a cache hit) is a call into a policy object. Production uses a heuristic policy; the fuzzer supplies an adversarial one; replay supplies a recorded one. Because execution is a pure function of `(W, sigma, seeds)`, ddmin is exact rather than best-effort. This is the entire differentiator versus live-server trace fuzzing. Do not let scheduling logic leak outside this seam.

## 7. Fidelity reference (F1)

Not an oracle for invariance, only for "the model is actually right".

- fp64 CPU forward pass of Qwen3-0.6B, a one-time cached run over a few thousand positions, results on disk
- Reported metrics: max absolute error, per-position KL, greedy-token match rate
- Near-tie positions (fp64 top1-to-top2 gap below a stated threshold) are excluded from the match rate and counted separately. Not excluding them is how people accidentally publish a misleading match rate
- ULP bounds are meaningless across dtypes. Use absolute and relative error against fp64, and reserve ULP language for same-dtype comparisons
- HF transformers is used as a sanity check only, never as ground truth, because it is not shape-invariant

## 8. Fuzzer design

### 8.1 Schedule space

The alphabet in section 2, constrained by feasibility rules (cannot resume a request that was never preempted, cannot evict a block with a nonzero refcount held by a running sequence, and so on). Global state adds memory pressure levels.

### 8.2 Coverage metric

Three components, all reported. Coverage is what proves exploration rather than sampling the easy middle.

1. **Lifecycle n-grams.** All feasible 2-grams and 3-grams of per-request events, reported as a percentage of the feasible set
2. **Boundary predicates**, each required at value, value minus 1, and value plus 1:
   - chunk end versus block size
   - cache-hit length versus block size (the live SGLang bug is literally `prefix_len == block_size`; make this the poster child)
   - split boundary versus KV length
   - batch size transitions across {1, 2, 3, 4, 8, 16, 31, 32}
   - free-block count in {0, 1, low}
3. **Preemption depth** per request, up to 3

### 8.3 Generation strategy

Swarm testing (Groce et al., ISSTA 2012): each campaign randomly disables or boosts feature subsets, which is what escapes the easy middle of the distribution. PCT (Burckhardt et al., ASPLOS 2010) supplies the vocabulary for probabilistic guarantees on hitting depth-d schedule bugs.

### 8.4 Minimization

ddmin (Zeller and Hildebrandt, IEEE TSE 2002) applied in three staged passes: requests, then schedule events, then prompt tokens. Output is a runnable script under 15 lines. Every published finding ships in this form.

## 9. Metamorphic relations

| ID | Relation |
|---|---|
| MR1 | Batch composition. Any set of cohabitants leaves `r`'s bits unchanged |
| MR2 | Chunk partition. Any partition of prefill leaves bits unchanged |
| MR3 | Preempt and resume. Preemption at any decode step, recompute mode, leaves bits unchanged |
| MR4 | Cache cold versus warm. Hit and miss produce identical bits |
| MR5 | Padding and occupancy. Adding a request that only changes tile occupancy leaves others' bits fixed |
| MR6 | Replay. Identical inputs twice, identical trajectory hash |
| MR7 | RNG isolation. Deleting request A leaves request B's tokens unchanged |
| MR8 | Temperature-0 tie-break. Argmax stable under logit-preserving permutations of the batch |
| MR9 | Mask no-op, fidelity-side. A fully-masked pad region changes nothing within tolerance |

MR1 through MR8 run both in-process and black-box. MR9 is in-process only.

## 10. Mutation testing

### 10.1 Operators

Allocator and KV:
- refcount increment twice on fork
- refcount decrement missing on free
- free a block still referenced by the cache index
- COW fork copies without bumping refcount
- COW fork bumps refcount without copying
- swap-in restores blocks to permuted slots without fixing the block table
- stale block-table read after reclamation

Scheduler:
- eviction eligible-set includes a running sequence
- recompute re-prefills with an off-by-one token count
- chunk boundary uses `<` where `<=` is correct, so one token is processed twice
- cache match length rounded up past a block boundary
- preemption chosen at a different point than recorded

Numerics and RNG:
- RNG keyed on global step instead of `(uid, position)`
- one code path reads split size from batch size
- reduction order reversed in the split-combine CTA

### 10.2 Observers

Output bits alone leave too many mutants equivalent. Internal audits run every step in debug builds:
- refcount ledger balances globally
- block-table injectivity (no two live sequences map to the same block without a COW record)
- freed-block poisoning, so any read of freed memory perturbs bits deterministically

### 10.3 Handling survivors

No hand-waving. Every survivor gets a written classification committed to the repo:
- **Proven equivalent**: an argument for why the mutation cannot change observable behavior. Typically a mutated heuristic that changes `sigma`, where `sigma` is universally quantified over anyway
- **Harness gap**: the mutation is observable but the suite missed it. Close the gap, or document why closing it is out of scope

Published mutation score excludes proven-equivalents and states the count. Cite Just et al. (FSE 2014) on mutant validity so the methodology reads literate rather than improvised.

## 11. Performance expectations

Honest numbers, from shipped systems: the Thinking Machines and vLLM demonstration ran roughly 62 percent slower in deterministic mode (26s to 42s for 1,000 sequences). SGLang acknowledges deterministic mode is significantly slower and targets reducing the gap to under 20 percent.

On sm_89 with a 0.6B dense model, expect 20 to 50 percent end-to-end overhead versus a non-invariant path in the same engine, mostly from losing split-K on skinny decode GEMMs and from conservative attention splits at small batch. Most of that is recoverable with CUDA graphs, elimination of host syncs in the decode loop, and tile tuning, because this regime is launch-overhead and bandwidth bound rather than occupancy bound.

Do not publish absolute tokens per second as a selling point. Publish ratios.

## 12. CI gates

1. Bitwise invariance suite (MR1 through MR8) on a fixed seed corpus, every commit
2. Nightly fuzz campaign with a coverage floor; regression if coverage drops
3. Throughput regression gate with a tolerance band
4. Static checks: no `tl.atomic_` in kernels under claim, no autotune decorators, kernel config registry unchanged without an accompanying claims-table review
5. `env.lock` emitted into every artifact; a mismatch invalidates published claims

## 13. Primary sources

- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (Sept 2025) and `thinking-machines-lab/batch_invariant_ops`
- vLLM: `docs.vllm.ai/en/latest/features/batch_invariance/`, tracking issue #27433, `tests/v1/determinism/`, PoC PR #24583
- SGLang: LMSYS blog "Towards Deterministic Inference in SGLang and Reproducible RL Training" (Sept 2025), `docs/advanced_features/deterministic_inference.md`, `sglang.test.test_deterministic`, issue #22819 (block-boundary corruption)
- LLM-42, arXiv 2601.17768. The over-constrained argument you must be able to rebut
- GRIEF, arXiv 2605.11202. Trace fuzzing of live serving systems. Your prior-art paragraph
- Feng Yao et al., "On the Rollout-Training Mismatch in Modern RL Systems"; TIM diagnosis arXiv 2605.14220; Miles "Truly On Policy" writeup in Awesome-ML-SYS-Tutorial
- "Give Me FP32 or Give Me Death?", arXiv 2506.09501
- Kwon et al., PagedAttention (SOSP 2023). Yu et al., Orca (OSDI 2022). Zheng et al., SGLang/RadixAttention (NeurIPS 2024). Dao et al., Flash-Decoding (PyTorch blog, Oct 2023)
- Groce et al., "Swarm Testing" (ISSTA 2012). Zeller and Hildebrandt, ddmin (IEEE TSE 2002). Burckhardt et al., PCT (ASPLOS 2010). Just et al., "Are Mutants a Valid Substitute for Real Faults?" (FSE 2014)
- Will Wilson, "Testing Distributed Systems with Deterministic Simulation" (FoundationDB, Strange Loop 2014). `tokio-rs/loom` for the decision-point-driving pattern in miniature
