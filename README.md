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

## Three observers, and what each one cannot see

The relations above are not one detector with one blind spot. They are three,
and the reason there are three is that a mutation testing run found a fault the
first two both missed.

| Observer | Compares | Against | Tolerance | Blind to |
|---|---|---|---|---|
| **I1 to I4** | the engine | itself, under a perturbed schedule | none, bitwise | any perturbation uniform across schedules |
| **F1** | the engine | an fp64 CPU reference | max abs logit error 0.5 | anything inside the tolerance |
| **golden bytes** | the engine | a committed baseline digest | none, bitwise | any change made before the baseline was written |

The first two blind spots overlap exactly where a real fault lives. Reverse the
fold loop in the split-combine CTA, descending instead of ascending, and every
metamorphic relation still passes. Not because the relations are weak, but
because the reversal is not a function of the schedule: the split count follows
from `kv_len` alone, so a position reached through four prefill chunks folds the
same partials in the same reversed order as the same position reached in one
pass. The engine agrees with itself perfectly, and I1 to I4 only ever ask whether
it agrees with itself.

F1 does not catch it either, and the precise reason matters more than the blunt
one. F1's headline statistic is the maximum absolute logit error against fp64,
and that statistic does not move:

| engine | max abs logit error vs fp64 | within the 0.5 bound |
|---|---|---|
| clean | 6.669992e-02 | yes, 7.5x headroom |
| reversed fold | 6.669992e-02 | yes |

600-token prompt, batch 1, two splits, sentinel confirming the patched kernel ran
28 times during the measurement. The two are the same number to every digit, so
**no threshold on that statistic separates the two engines**. Not "the bound is
too loose": no setting of the bound distinguishes them, because the quantity
being thresholded is identical.

That is not the same as saying the engines are indistinguishable. A per-position
comparison restricted to the positions the fault can reach does separate them,
and it was measured:

| positions | splits | max abs logit delta | positions changed |
|---|---|---|---|
| 0 to 511 | one | exactly 0 | 0 of 512 |
| 512 to 599 | two | 3.906250e-02 | 85 of 88 |

The difference is real, it is confined exactly to the multi-split positions, and
it is about 2.5 fp16 ulp at these logit magnitudes, smaller than the 6.669992e-02
quantization error F1 is already measuring. F1 is tolerance-based by design and
compares against a reference rather than against a prior run, so a perturbation
this size is inside what it is built to absorb. Tightening it until this fired
would make it fire on legitimate rebuilds, trading a working detector for a
false one.

So the gap is not that F1 is badly calibrated. It is that "differs from fp64 by
more than t" and "differs from what this engine produced yesterday" are different
questions, and only the second one has an exact answer.

So the mutant survives both, and both are behaving correctly. What was missing
was an observer that compares against something outside the running process:
`harness/fuzz/golden.py` commits a sha256 over raw fp16 logit bytes for a fixed
four-prompt corpus and compares exactly. Under the reversed fold it reports the
two prompts of 600 and 520 tokens as differing and the two of 48 and 17 as
identical, which is the corpus doing what it was built for. Two prompts cross the
512-token split boundary and two do not, so the fold path is exercised and the
comparison is not vacuous, and the split that falls exactly along the boundary
between changed and unchanged digests is evidence that the observer is seeing the
mechanism rather than noise.

**The first version of this operator was killed for the wrong reason.** It was a
proxy: scale the output by `1 + 2**-11` whenever the split count was at least
two, on the theory that a reversed fold moves the last bits by about that much.
The campaign killed it by bitwise divergence and the kill was worthless, because
the *firing condition* was schedule-dependent in a way the modelled fault is not.
A request prefilled in chunks reaches a position through a different sequence of
`kv_len` values than the same request prefilled whole, so the proxy fired on a
different set of calls in the two runs and the relations saw a difference no real
reversed fold would produce. A mutant killed for the wrong reason inflates a
mutation score exactly as badly as a mutant that never ran. The operator is now
the actual kernel with one loop bound reversed.

Golden bytes have the blind spot the other two do not: they cannot see a fault
that was present when the baseline was written. They are a regression detector,
not a correctness detector, and the baseline's authority comes entirely from the
`env.lock` committed beside it. Regenerating it on different hardware is not a
repair, it is a new claim.

## The three numbers

| # | Claim | Value | Reproduce |
|---|---|---|---|
| 1 | Invariance under adversarial scheduling | see `harness/mr/run.py`; 55 of 55 relation runs bitwise identical across 5 block sizes | `python3 -m harness.mr.run` |
| 2 | Harness power | 9 of 10 seeded faults killed, 1 proven-equivalent, 0 not-exercised; median time to detection 10.4 s | `python3 -m harness.fuzz.campaign --seeded-faults` |
| 3 | Cost of determinism | 1.06x within lockstep, against vLLM's own 1.45x; lockstep invariant is 3.3x vLLM batch-invariant wall time | `python3 -m bench.throughput` |

Claim 3, in full, on a committed 8-request 1232-token trace, median of 5:

| configuration | wall time vs lockstep fast |
|---|---|
| lockstep, fast mode | 1.00x |
| lockstep, invariant | 1.06x |
| vLLM default | 0.22x |
| vLLM `VLLM_BATCH_INVARIANT=1` | 0.32x |
| SGLang deterministic | not measured |

Two readings, and the second is the honest one.

**Cost of determinism inside each engine**, each measured against its own fast
path, which is unaffected by this engine being slower overall: **lockstep 1.06x,
vLLM 1.45x**. Lockstep's figure is small because its fast path differs in exactly
one way, letting torch pick the GEMM; the attention kernel and scheduler are
identical in both. It is the cost of the GEMM constraint, not of the whole
design, and it should be read that way.

**Absolute standing**: lockstep invariant is **3.3x the wall time of vLLM
batch-invariant** on this trace. No CUDA graphs, no host-sync elimination, eager
only. That is the number a reader should use, and it is why this is called a
correctness-reference engine rather than a fast one.

SGLang is not installed in the external environment, so its row is not measured
rather than estimated or omitted.

Claim 2, before and after the third observer was added. The operator set and the
campaign are otherwise identical, and the difference is one mutant:

| | killed | survived | not exercised |
|---|---|---|---|
| invariance relations and F1 only | 8 of 10 | 2 | 0 |
| with golden bytes | **9 of 10** | 1 | 0 |

The two survivors in the first row are the reversed split-combine fold, which is
a real fault no observer then present could see, and the recompute off-by-one,
which is a proven-equivalent mutant: it re-prefills a token that is immediately
overwritten with the identical value, so no observer can distinguish it and none
should. The second row has only the equivalent mutant left. Every one of the ten
reports a nonzero sentinel, so no trial in either row is a mutant that failed to
execute.

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
- **The mutation score is 10 operators, not 30 to 50.** Three earlier ones were
  deleted along with the code they targeted, once that code turned out to be
  unreachable; three were added later to cover the RNG keying, the split-combine
  fold order, and the batch-derived split size. Scoring against unreachable
  operators measures nothing, and ten is a thin denominator even so.
- **Coverage is a percentage of a derived denominator**, not of an exhaustive
  one. The denominator comes from a declared transition relation; it is checked
  against reality but it is still a model.
- **Execution-counter gating proves the path ran, not that the patch took
  effect.** These are different claims and the counters cannot tell them apart.
  A mutation operator that rebinds a name in the module defining it leaves an
  importing caller bound to the original, so the mutated code never executes
  while every counter the gate checks still fires. That happened here, to the
  reversed split-combine operator, and it was scored as a survivor until a
  sentinel contradicted it. Each operator now increments a counter from inside
  the mutant body, checked separately from the execution counters, and a trial
  whose sentinel reads zero is reported as `not-exercised` rather than as a
  survivor. The sentinel closes this particular hole; it does not prove no
  operator has a subtler version of it, and a mutation score is only ever as
  sound as the evidence that its mutants ran.
- **The golden baseline is a regression detector, not an oracle.** It cannot see
  a fault that predates it, and its authority is entirely the `env.lock` beside
  it.

## The thesis, demonstrated on its author

Lockstep exists because engines claim a compatibility surface wider than their
test surface. That is not a hypothesis about other people's code. It happened
eight times in this repository. Seven were caught by a run contradicting a
declaration rather than by review. The seventh was not, and could not have been,
for reasons worth stating:

| # | Declared | Actually |
|---|---|---|
| 1 | A fidelity corpus exercising the fixed-split attention kernel | No prompt exceeded 512 tokens, so the split-combine fold never executed in any fidelity number |
| 2 | A VRAM table across five chunk sizes | Prompts were shorter than the largest chunk, so that row was byte-identical to unchunked and chunked nothing |
| 3 | Eviction under memory pressure, tested | Admission counted only free blocks, so a request serviceable by evicting was refused and the eviction path was unreachable |
| 4 | A copy-on-write boundary case passing | Nothing in the engine called `fork`; the test exercised dead code |
| 5 | A coverage denominator derived from the lifecycle | The transition table was wrong four times, in both directions, inflating and deflating the denominator |
| 6 | A mutation trial gated on the mutated path executing | The gate passed and the patch never ran; the fault rebound a name in the module that defined it while the caller held its own import, and the trial was scored as a survivor |
| 7 | A mutant killed, counted toward the score | The path ran, the observer fired, and the kill was still wrong: the operator was a proxy whose firing condition varied with the schedule, so the relations caught the proxy rather than the fault it modelled |
| 8 | A nine-command CLI, and a reproduce line naming one of them | No `lockstep` executable existed. Every entry point was `python3 -m`, the reproduce line named a command nobody could run and a file under a gitignored directory, and the tests asserted the string rather than the thing it named |

The eighth was found while fixing the seventh, and it is the most literal
instance of the sentence this project opens with. A declared surface wider than
the tested one is exactly what Lockstep exists to find in other people's engines,
and the CLI section of the frontend spec had been declaring nine commands at a
repository that shipped none of them. The tests covering it checked that the
divergence report printed the right string, which it did, correctly, for a
command that did not exist.

The seventh is the one that took longest to see, because it looked like success.
A kill is normally self-justifying: the observer fired, so the harness works. But
the operator scaled the output whenever the split count was at least two, and the
split count varies with the schedule, so the relations were detecting the
operator rather than the fault. Path executed, observer fired, measurement still
wrong. A gate cannot catch that; only reading the operator against the fault it
claims to model can.

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

## Evidence, and replaying it

`evidence/` is committed and holds the artifacts the claims above actually cite,
each with its `env.lock`. `results/` stays gitignored: it is bulk campaign output,
hundreds of files superseded on every run, and no published number points at one.
Promotion is deliberate, one artifact at a time, with
`python3 -m report.publish results/<date>/<artifact>.json`.

| file | backs |
|---|---|
| `evidence/verify-0004.json` | claim 1, 55 of 55 relation runs across 5 block sizes |
| `evidence/fuzz-0003.json` | claim 2, the ten-operator mutation campaign over 192 cases |
| `evidence/throughput-0002.json` | claim 3, the cost-of-determinism table |
| `evidence/fidelity-0003.json` | F1, exact KL over the full vocabulary |
| `evidence/certify-0001.json` | the vLLM certification, 7 of 7 boundary cases clean |
| `evidence/case-0003.json` | the eviction finding the fuzzer found, minimized and 1-minimal |
| `evidence/case-witness.json` | a replay-determinism witness, see below |

The differentiator sentence at the top of this file claims every finding
minimizes to an exact replay. That is checkable rather than asserted:

```sh
python3 -m harness.replay evidence/case-witness.json
```

The witness is a clean case, not a bug repro, carrying the trajectory hash over
emitted tokens, raw fp16 logit bytes, the packed work list, the allocator ledger,
and the prefix cache index. Replaying it in a fresh process re-runs the exact
`(W, sigma, seeds)` triple and compares hashes. It is deliberately not trivial:
two prompts cross the 512-token split boundary, prefill is chunked, one request
is preempted mid-decode, and a 544-token prefix is shared, because a witness that
exercised none of the machinery would pass whatever the engine did.

`evidence/case-0003.json` is the other kind: a real finding, minimized to one
request and proven 1-minimal in 788 checks. It is the oversized-request wedge,
where a 61-token request with 4 new tokens against a 2-block pool at block size 32
needed 3 blocks, could never be admitted, and sat in the queue until the scheduler
reported memory pressure that named free blocks the request could never have used.

**It is pinned to the commit that closed it.** Fixed by
[`b592d7c`](https://github.com/u7k4rs6/LockStep/commit/b592d7c), "Bound the
eviction loop, and refuse a request that can never fit the pool"; the last
revision before that fix is
[`1269c01`](https://github.com/u7k4rs6/LockStep/commit/1269c01). Replaying it at
HEAD reports the finding as closed and names the commit, rather than reporting a
bare "did not reproduce" that a reader cannot distinguish from a broken file.

One thing about that pin has to be said plainly, because the obvious reading of
it is wrong. **No single checkout reproduces this finding**, and not because
anything is missing: the harness that produces and replays cases,
`harness/sim/driver.py` and `engine/sched/lifecycle.py`, landed in `d8465e8`,
which is *after* the fix. The fuzzer found this bug from an uncommitted working
tree, the fix was committed, and the fuzzer was committed after it. So checking
out `1269c01` gets the pre-fix scheduler and no way to drive it.

What was verified, rather than inferred from commit dates: removing only the
oversized-request guard from `Scheduler.submit` at HEAD, 8 lines, makes this
artifact's minimized case raise `OutOfBlocks` with the recorded condition, one
request waiting against 2 free blocks with nothing running to free them.
Restoring the guard, the same case is refused at submit and never wedges. That is
in the artifact's `provenance` block along with both SHAs, and it is a one-hunk
change a reader can apply.

Artifacts written since carry `engine_revision`, so newer findings pin themselves
and need none of this.
