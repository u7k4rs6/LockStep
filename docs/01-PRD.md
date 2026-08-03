# Lockstep: Product Requirements Document

Version 0.1, August 2026. Owner: Utkarsh Bahuguna.

## 1. One-line description

A batch-invariant LLM inference engine whose determinism is certified by deterministic-simulation fuzzing, with the harness's own bug-finding power measured by mutation testing, then pointed at vLLM and SGLang to certify their deterministic modes at boundary conditions.

## 2. Problem

Continuous batching makes a model's output depend on what else is in the batch. Reduction split sizes and kernel selection vary with batch shape, so the same prompt at batch size 1 and batch size 32 can produce different logits and eventually different tokens. This breaks eval reproducibility and it silently makes RL post-training off-policy, because rollout logprobs and training logprobs diverge for identical weights.

The kernel-level diagnosis is solved and shipped. Thinking Machines published it in September 2025 with `batch_invariant_ops`. vLLM ships `VLLM_BATCH_INVARIANT=1`. SGLang ships `--enable-deterministic-inference` with claimed compatibility across chunked prefill, CUDA graphs, and radix cache.

The unsolved part is verification. Both engines claim a compatibility surface wider than their test surface. SGLang currently has an open KV cache corruption bug in deterministic mode triggered when a request's `prefix_len` exactly equals the KV block size. That is a scheduler-layer boundary condition, not a kernel bug, and it survived a shipped determinism test suite. Nobody has measured whether those suites would catch an injected scheduler or allocator fault, and nobody has certified the boundary conditions systematically.

## 3. What this product is, and is not

**Is:** a correctness reference engine plus a certification harness. The engine exists so the harness has something whose internals can be driven and mutated. The harness is the deliverable.

**Is not:** a vLLM competitor, a throughput project, a serving product, or a novel-kernel project. Any framing that implies otherwise is a positioning failure and must be corrected in the README.

## 4. Users

1. **Primary, near-term: hiring managers and staff engineers on inference and reliability teams.** They read the README claims table and the three headline numbers in under 60 seconds. Success is that they can tell within that window that the author understands numerics, schedulers, and testing methodology.
2. **Secondary, real: engineers running RL post-training or reproducible evals** who need to know whether their engine's deterministic mode actually holds under their workload shape. They consume the black-box certifier, not the engine.
3. **Tertiary: vLLM and SGLang maintainers**, who consume filed issues with minimized repros.

## 5. Wedge decision

Eval reproducibility is the adoption story. RL mismatch is the motivation story.

Rationale: the RL mismatch space already has owners (truncated importance sampling, masked importance sampling, slime/Miles "Truly On Policy" mode). Competing there means competing with training-framework teams. Eval reproducibility has no incumbent tool, needs no training stack to demonstrate, and the certifier runs against any OpenAI-compatible endpoint. Lead the README with eval reproducibility, cite the RL literature as the reason the problem is expensive.

## 6. Success criteria

**Tier 1, the project landed:** at least one divergence found in vLLM or SGLang deterministic mode, minimized to a readable repro, filed upstream, and acknowledged by a maintainer.

**Tier 2, the project is a strong artifact:** all three headline numbers published with committed reproduction scripts, and the invariance CI gate green on every commit.

**Tier 3, the project is a competent artifact:** engine passes its own invariance suite under fuzzed schedules, mutation score published with a survivor census.

Below Tier 3 the project is not worth publishing under this framing and should be renamed to a learning-exercise README.

## 7. Headline numbers

Every number below must ship with a script in `bench/` that reproduces it and a committed workload trace.

1. **Invariance under adversarial scheduling.** N fuzzed schedules across batch sizes 1 to 32, with preemption, arbitrary chunk boundaries, and cache churn. Reported with lifecycle 2-gram coverage percentage, boundary-predicate checklist, and 100 percent bitwise equality against canonical execution. Measured as SHA of concatenated fp16 logit bytes per request versus the oracle SHA.
2. **Harness power.** M of N injected allocator and scheduler faults detected, median time-to-detection in seconds, K survivors split into proven-equivalent (with written arguments in-repo) and harness gaps (with the gap closed or documented). Detection is defined as any bitwise divergence or internal-audit assertion firing.
3. **Cost of determinism, honestly benchmarked.** Invariant mode versus this engine's own fast mode, then as a fraction of vLLM `VLLM_BATCH_INVARIANT=1` and SGLang deterministic mode. Same GPU, same model, same committed trace, median of 5 runs. Comparing against their deterministic modes rather than only vanilla is what makes the number undismissable.

Supporting artifact, not a headline: a per-token absolute delta-logprob table for vanilla vLLM and SGLang scoring identical tokens batched versus unbatched. This is the quantity the entire importance-sampling literature exists to correct for.

## 8. Scope

### 8.1 Must-have

| Item | Definition of done |
|---|---|
| Engine core | Qwen3-0.6B fp16 loads, greedy and top-p decode, paged KV with block table and refcounts |
| Batch-invariant kernels | Fixed split-size attention, no split-K GEMM, single-CTA RMSNorm and softmax, frozen Triton configs |
| Scheduler | Continuous batching, chunked prefill at arbitrary boundaries, preemption by recompute, eviction under pressure |
| Prefix cache | Exact-prefix, block-aligned, dict keyed on token hash, refcounted blocks with copy-on-write |
| Deterministic sampler | Counter-based Philox keyed on (seed, request uid, position); stable tie-break by token ID |
| Canonical execution mode | Batch size 1, uninterrupted prefill, cold cache, no speculation |
| Simulation fuzzer | Scheduler decision points behind a policy interface; (workload, schedule, seeds) fully determines execution |
| Minimizer | ddmin over requests, then schedule events, then prompt tokens; output is a runnable repro under 15 lines |
| Metamorphic suite | The nine relations in the architecture doc, runnable in-process and black-box |
| Mutation campaign | 30 to 50 curated mutants, scored, survivors triaged |
| CI gates | Bitwise invariance, throughput regression, static checks forbidding atomics on accumulators and forbidding autotune |
| Black-box certifier | Same MR suite against any OpenAI-compatible endpoint, boundary-focused workload generator |

### 8.2 Nice-to-have, in cut order

Swap-based preemption. Radix tree replacing the exact-prefix dict. Qwen3-1.7B. CUDA graphs (promote to must-have only if throughput is embarrassing enough to invite dismissal).

### 8.3 Explicitly out of scope

Speculative decoding, entirely. MoE models. Multi-GPU. Quantization. Serving polish (auth, rate limits, multi-tenancy). Cross-hardware determinism claims of any kind.

The speculative decoding cut is deliberate and permanent. If it re-enters scope after week 4, the plan has failed. The correct interview answer about spec decode is the distinction between exact greedy equivalence and distributional equivalence under sampling, plus the binomial acceptance test design. Being able to state that is worth more than a half-built implementation.

## 9. Claims discipline

The README claims table is the product surface. Every claim must be falsifiable and environment-scoped.

Forbidden words in any published artifact: "proof" applied to anything statistical, "guaranteed" without an environment scope, "catches all" applied to mutation results, "fully deterministic" without naming the pinned environment.

Required framing: every bitwise claim carries the pinned environment tuple (GPU model, driver, CUDA, Triton, torch versions) from `env.lock`. Every statistical claim carries alpha, power, effect size, and sample count.

Prior art must be cited generously and specifically in the README: Thinking Machines for the diagnosis, vLLM and SGLang for the shipped implementations, GRIEF for trace-based fuzzing of live serving systems, LLM-42 for the argument that batch invariance is over-constrained. The differentiator sentence, which must appear near the top: GRIEF fuzzes live servers with wall-clock traces, so repro is probabilistic; Lockstep's execution is a pure function of (workload, schedule, seeds), so every finding minimizes to an exact replay.

## 10. Timeline and exit criteria

Six to eight weeks part-time. Every week exits on something runnable and observable, not on "implemented X".

| Week | Exit criterion |
|---|---|
| 1 | Qwen3-0.6B greedy batch-1 decode; script prints token match versus HF plus logit absolute-error histogram against a cached fp64 CPU reference |
| 2 | Paged KV, naive continuous batching, fixed-config GEMM and RMSNorm, deterministic sampler. Replay MR green. Ablation table naming which op still breaks batch invariance |
| 3 | Fixed-split chunk-invariant attention. Batch invariance green across batch sizes 1, 2, 4, 8, 16, 31, 32 with identical logit bytes. Make-or-break week |
| 4 | Chunked prefill at arbitrary boundaries, recompute preemption, prefix cache. Full MR suite green including preempt-at-every-step sweep over a 64-token generation |
| 5 | Simulation fuzzer with policy injection, swarm configs, ddmin. Coverage report plus a seeded bug auto-minimized to a printed repro |
| 6 | Mutation campaign complete. Score table, median time-to-detection, survivor census with equivalence arguments committed |
| 7 | Performance pass. Throughput table across five configurations with committed trace |
| 8 | Black-box certification runs against vLLM and SGLang deterministic endpoints. Report published. Any surviving divergence filed upstream |

Slippage absorbs into weeks 3 and 6 first, then into the section 8.2 cut order.

## 11. Risks and kill criteria

| Risk | Detection signal | Action |
|---|---|---|
| Attention chunk-invariance fights Triton codegen on sm_89 | Chunking MR still red after 7 focused days | Downgrade to batch invariance only, fixed chunk grid. State the downgrade explicitly in the README. Redirect the saved week into the certifier. Never soften the claim silently |
| Engine-building starves the harness | End of week 4 and the fuzzer is not started | Freeze engine features permanently. Harness runs on whatever exists |
| Throughput invites dismissal as a toy | Below 15 percent of vLLM default after the CUDA-graph pass | Reframe explicitly as a correctness-reference engine. Publish only the cost-of-invariance ratio and the cross-engine comparison, not absolute tokens per second |
| GRIEF overlap makes the harness look derivative | Cannot state the simulation-versus-live-trace distinction in two sentences | If mutation scores do not separate this harness from a replay of their approach, shrink the harness and let the certification findings carry the project |
| External deterministic modes pass everything | Certifier finds nothing after boundary-focused campaigns | Publish as negative certification with coverage stated. The mutation table must then carry the artifact. Budget week 8 accordingly |
| 8 GB VRAM wall | KV budget under 20k tokens at target load | Qwen3-0.6B for everything. Rent one A100 evening only for the final table |
| Speculative decoding temptation | Reading EAGLE papers in week 5 | It was cut in section 8.3. Stop |

The first three are the three most likely ways this ends up mediocre: harness starvation, silent claim-softening, and performance ridicule.

## 12. Non-goals for the README

Do not claim novelty on batch-invariant kernels. Do not claim to have discovered the nondeterminism problem. Do not present throughput as a selling point. Do not compare only against vanilla vLLM, which would be a strawman a reviewer will immediately name.
