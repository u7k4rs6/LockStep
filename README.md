# Lockstep

A batch-invariant LLM inference engine whose determinism is **certified** by
deterministic-simulation fuzzing, with the harness's own bug-finding power
measured by mutation testing, then pointed at **vLLM** and **SGLang** to certify
their deterministic modes at boundary conditions.

> GRIEF fuzzes live servers with wall-clock traces, so repro is probabilistic;
> Lockstep's execution is a pure function of (workload, schedule, seeds), so every
> finding minimizes to an exact replay.

The engine is a fixture. **The harness is the product.** Batch-invariant kernels
are already shipped upstream by Thinking Machines, vLLM, and SGLang, so nothing
here claims novelty on kernels. What is unbuilt is the verification layer.

---

## Claims

Every claim is scoped to the environment tuple in `env.lock`, embedded in every
result artifact. A claim without one is invalid by construction.

| ID | Statement | How verified | Status | Scope |
|---|---|---|---|---|
| **I1** | Batch invariance. For any set of cohabitant requests, `r`'s output is bit-identical to canonical execution `C(r)` | MR1 and MR5, bitwise on fp16 logit bytes, batch sizes {1, 2, 3, 4, 8, 16, 31, 32} | holds | sm_89, block sizes 8 to 128 |
| **I2** | Schedule invariance. For any valid schedule, including preemption at any decode step, any chunk partition, eviction under pressure, and cache hit versus miss | MR2, MR3, MR4, plus decode-versus-prefill KV equality checked per layer at the tensor level | holds | as above |
| **I3** | Replay determinism. Identical `(W, sigma, seeds)` twice yields an identical trajectory hash over all engine state | MR6, 8 workload shapes, and a cross-process replay under differing `PYTHONHASHSEED` | holds | as above |
| **I4** | RNG isolation. `r`'s tokens are a function only of `(seed, uid, position)` and `r`'s logits | MR7, 11 perturbations: each request removed in turn, one added, order reversed, each re-run alone | holds | as above |
| **F1** | Fidelity. Batch-1 logits against an fp64 CPU reference, within the bounds in `docs/02-technical-architecture.md` 7.1 | `bench/fidelity.py`, exact KL over the full 151936-token vocabulary at 2756 positions | passes 7 of 7 bounds | corpus `sha256:59759d5b…` |

The trajectory hash covers emitted tokens, raw fp16 logit bytes, the packed work
list, the allocator ledger, and the prefix cache index. Output bits alone would
leave allocator faults invisible.

## The three numbers

| # | Claim | Value | Reproduce |
|---|---|---|---|
| 1 | Invariance under adversarial scheduling | see `harness/mr/run.py`; 55 of 55 relation runs bitwise identical across 5 block sizes | `python3 -m harness.mr.run` |
| 2 | Harness power | 6 of 7 seeded faults killed, 1 proven-equivalent, 0 not-exercised; median time to detection 26.5 s | `python3 -m harness.fuzz.campaign --seeded-faults` |
| 3 | Cost of determinism | **not yet measured** | `python3 -m bench.throughput` |

Claim 3 is a ratio against this engine's own fast mode and against vLLM and
SGLang deterministic modes. It is not written down yet, and an unwritten number
is stated as unwritten rather than estimated.

## What this does not claim

- **No novelty on batch-invariant kernels.** Thinking Machines published the
  diagnosis; vLLM and SGLang shipped implementations. This reimplements them so
  the harness has internals it can drive and mutate.
- **No cross-hardware guarantees.** Every bitwise claim is scoped to one
  environment tuple. Nothing here is known to hold on another GPU, driver, CUDA,
  Triton, or torch version.
- **No throughput superiority.** This is a correctness-reference engine. It has
  no CUDA graphs and makes no attempt to be fast.
- **No claim about untested models or configurations.** Qwen3-0.6B fp16 only.
- **Copy-on-write is absent, and that is a design consequence rather than a
  feature.** Prefix sharing here is whole-block only, so a granted hit always
  covers an exact multiple of the block size and a sequence's first write lands
  in a block it reserved itself. `assert_exclusive` checks that on every write.
  **This holds only because there is no fork API.** vLLM and SGLang need
  copy-on-write because they fork for parallel sampling and beam search; adding
  either here would require it back.
- **The mutation score is 7 operators, not 30 to 50.** Three were deleted along
  with the code they targeted, once that code turned out to be unreachable.
  Scoring against unreachable operators measures nothing.
- **Coverage is a percentage of a derived denominator**, not of an exhaustive
  one. The denominator comes from a declared transition relation; it is checked
  against reality but it is still a model.

## The thesis, demonstrated on its author

Lockstep exists because engines claim a compatibility surface wider than their
test surface. That is not a hypothesis about other people's code. It happened
five times in this repository, and every instance was caught by a run
contradicting a declaration rather than by review:

| # | Declared | Actually |
|---|---|---|
| 1 | A fidelity corpus exercising the fixed-split attention kernel | No prompt exceeded 512 tokens, so the split-combine fold never executed in any fidelity number |
| 2 | A VRAM table across five chunk sizes | Prompts were shorter than the largest chunk, so that row was byte-identical to unchunked and chunked nothing |
| 3 | Eviction under memory pressure, tested | Admission counted only free blocks, so a request serviceable by evicting was refused and the eviction path was unreachable |
| 4 | A copy-on-write boundary case passing | Nothing in the engine called `fork`; the test exercised dead code |
| 5 | A coverage denominator derived from the lifecycle | The transition table was wrong four times, in both directions, inflating and deflating the denominator |

Each was found the same way: an execution counter, an assertion, or an
observed-versus-feasible check contradicted something that had been written down
and believed. That is the entire argument for the tool, and it is why the engine
now carries 25 execution counters that every relation asserts against, and why a
relation which passes without firing its own mechanism is treated as a failure
rather than a pass.

The deterministic modes in vLLM and SGLang are tested by people at least as
careful as the author of this repository. The question is not whether they were
careless. It is whether anyone has checked which of their declared surface
actually executes.

## Prior art

- [Thinking Machines, *Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and [`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops), for the diagnosis this project takes as given.
- [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/), tracking issue #27433, `tests/v1/determinism/`.
- [SGLang deterministic inference](https://docs.sglang.ai/advanced_features/deterministic_inference.html) and issue #22819, the open block-boundary corruption in deterministic mode at `prefix_len == block_size`. That shape is a first-class boundary predicate here.
- GRIEF (arXiv 2605.11202), trace fuzzing of live serving systems. The differentiator sentence above is about this.
- LLM-42 (arXiv 2601.17768), the argument that batch invariance is over-constrained, which this project must be able to rebut rather than ignore.
- Groce et al., *Swarm Testing* (ISSTA 2012). Zeller and Hildebrandt, ddmin (IEEE TSE 2002). Just et al., *Are Mutants a Valid Substitute for Real Faults?* (FSE 2014).

## Install

Requires an NVIDIA GPU. Developed on sm_89 with 8 GB; nothing is claimed
elsewhere.

```sh
uv sync
python3 scripts/download_weights.py          # pinned by revision SHA, checksummed
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python3 scripts/build_fp64_reference.py    # about 90 seconds
```

Reproduce claim 1 from scratch:

```sh
python3 -m harness.mr.run --block-sizes 8 16 32 64 128
```
