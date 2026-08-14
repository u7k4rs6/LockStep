<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/hero-dark.svg">
  <img alt="Lockstep: a batch-invariant inference engine and the harness that tries to prove it wrong. Two traces of one workload run coincident and then fork at 44 co-resident byte-identical requests, where logprobs disagree, which became vllm-project/vllm issue 51187. 65 of 65 relation runs bitwise, 10 of 10 seeded faults killed, 19 of 25 lifecycle 2-grams reached, one finding filed upstream." src="docs/img/hero-light.svg" width="840">
</picture>

[![ci](https://github.com/u7k4rs6/LockStep/actions/workflows/ci.yml/badge.svg)](https://github.com/u7k4rs6/LockStep/actions/workflows/ci.yml)

If you run evals or RL post-training and your numbers move between runs, this
tells you whether the inference engine is the reason. The certifier works
against any OpenAI-compatible endpoint, so you can point it at your own
deployment without adopting anything else here.

Batch-invariant kernels are already shipped by Thinking Machines, vLLM and
SGLang. Nothing here claims novelty on kernels. What was unbuilt is the
verification layer, so **the engine is a fixture and the harness is the
product**.

```sh
uv run ./lockstep replay evidence/case-witness.json      # no GPU, no server, seconds after a clone
uv run ./lockstep certify --cache-mode warm --repeats 5  # needs a GPU and a local vLLM
```

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/architecture-dark.svg">
  <img alt="Workload, schedule and seeds feed the engine, which runs the workload under the given schedule and again under a canonical batch-1 schedule. Oracles compare the two bitwise. An internal white-box loop runs through ddmin, a witness artifact and the report; an external black-box loop reads logprobs over HTTP from vLLM or SGLang." src="docs/img/architecture-light.svg" width="840">
</picture>

<details>
<summary>Three things in that picture are load-bearing</summary>

**Workload and schedule are separate inputs.** A trace records one interleaving
that happened. Lockstep takes the schedule as an argument, so the same workload
can be replayed under a different one, and a schedule can be shrunk by the
minimizer without touching the workload. That is what makes minimization exact
rather than confirmed.

**Canonical execution is the same engine**, re-run under a batch-1 uninterrupted
schedule. Not a second implementation, not a reference model. Every invariance
relation is the engine measured against itself, which is why a shared bug
cancels: both sides move together and the comparison still holds. That blind
spot is what F1 against fp64 and the committed golden bytes are for.

**There are two loops, not one chain.** The internal loop is white box and sees
the whole trajectory hash, so it can minimize a failure to a 1-minimal case and
replay it by hash. The external certifier sees nothing but logprobs over HTTP
against an engine it did not build, which is strictly weaker, and is the reason
a clean certification means no divergence at that observable rather than bitwise
identity.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/three-numbers-dark.svg">
  <img alt="65 of 65 relation runs bitwise identical, 10 of 10 seeded faults killed with none equivalent, and lockstep invariant at 5.2 to 5.7 times the wall time of vLLM batch-invariant in eager mode." src="docs/img/three-numbers-light.svg" width="840">
</picture>

Reproduce with `make claim1`, `make claim2`, `make claim3`. All three need a GPU.

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/claims-dark.svg">
  <img alt="Five claims with what verifies each. I1 batch invariance, I2 schedule invariance, I3 replay determinism and I4 RNG isolation all hold. F1 fidelity passes 7 of 7 bounds against an fp64 CPU reference." src="docs/img/claims-light.svg" width="840">
</picture>

<details>
<summary>Scope, and one qualification about what env.lock actually describes</summary>

All thirteen committed artifacts were audited and twelve carry byte-identical
tuples: `torch 2.6.0+cu124`, `triton 3.2.0`, `cu12.4`, driver `595.84`, `sm_89`
on an RTX 4060 Laptop. The thirteenth, `evidence/upstream-finding.json`, carries
no tuple and should not: it is a provenance index, not a result.

**For a `certify` run, `env.lock` describes the certifier's process, not the
engine it certified.** vLLM runs in a separate virtual environment on
`torch 2.11.0+cu130`, so those artifacts carry the tuple of the process making
the requests while the subject ran on a different one. Read against a certify
artifact, `env.lock` scopes the observer rather than the observed. That is
row 17 of the thesis table.

Certify runs now capture the subject's tuple as well, read from the server's
startup output and from the interpreter the certifier launched. The six
artifacts backing vllm#51187 predate the field and are deliberately not re-run;
`scripts/audit_artifacts.py` prints which artifacts carry it rather than letting
an absent field read as agreement.

The trajectory hash covers emitted tokens, raw fp16 logit bytes, the packed work
list, the allocator ledger, and the prefix cache index. Output bits alone would
leave allocator faults invisible.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/observers-dark.svg">
  <img alt="Four observers. I1 to I4 compare the engine against itself under a perturbed schedule, F1 against an fp64 reference, golden bytes against a committed baseline, and PATH-EQ compares the observed path against the path the scheduler serves. The first three shared a blind spot because all read model.forward while every served request goes through forward_batch." src="docs/img/observers-light.svg" width="840">
</picture>

<details>
<summary>The fault that took a third observer, and why no threshold on F1 could have caught it</summary>

Reverse the fold loop in the split-combine CTA and every metamorphic relation
still passes. Not because the relations are weak: the reversal is not a function
of the schedule, so a position reached through four prefill chunks folds the
same partials in the same reversed order as the same position reached in one
pass. The engine agrees with itself perfectly, and I1 to I4 only ever ask
whether it agrees with itself.

F1 does not catch it either, and the precise reason matters. Its headline
statistic is max absolute logit error against fp64, and that statistic reads
`6.669992e-02` for both the clean engine and the reversed fold, to every digit.
**No setting of the bound distinguishes them**, because the quantity being
thresholded is identical.

A per-position comparison does separate them: positions 0 to 511 at one split
are unchanged, positions 512 to 599 at two splits move by `3.906250e-02`, about
2.5 fp16 ulp, which is smaller than the quantization error F1 is already
measuring. So the gap is not that F1 is badly calibrated. "Differs from fp64 by
more than t" and "differs from what this engine produced yesterday" are
different questions, and only the second has an exact answer.

`harness/fuzz/golden.py` commits a sha256 over raw fp16 logit bytes for a fixed
four-prompt corpus. Under the reversed fold the two prompts of 600 and 520
tokens differ and the two of 48 and 17 do not, which is the split boundary
falling exactly where the mechanism says it should.

**The first version of this operator was killed for the wrong reason.** It was a
proxy that scaled output whenever the split count was at least two, and its
firing condition was schedule-dependent in a way the modelled fault is not, so
the relations caught the proxy rather than the fault. A mutant killed for the
wrong reason inflates a mutation score exactly as badly as one that never ran.
The operator is now the actual kernel with one loop bound reversed.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/mutation-dark.svg">
  <img alt="Ten mutation operators. Nine killed by the invariance relations and F1 alone, with the reversed split-combine fold surviving as O10. Ten killed once golden bytes are added as an observer outside the process." src="docs/img/mutation-light.svg" width="840">
</picture>

<details>
<summary>There are no equivalent mutants, and the previous claim that there was one was wrong in the flattering direction</summary>

The published score read "9 of 10 killed, 1 proven-equivalent" for weeks, with a
written argument that the recompute off-by-one operator re-prefilled a token
immediately overwritten with an identical value. That argument described a
mechanism that is not in the code. The operator set `kv_len` after `_preempt`;
`_admit` resets it to 0 on re-admission and nothing reads it in between. It was
a dead store. The fault never reached engine state, so no observer could have
distinguished it, and calling that equivalence confused "cannot be
distinguished" with "was never injected".

Injected where the value survives, after `_admit` on the resume path, the same
operator dies in 1.3 seconds by bitwise divergence with the sentinel confirming
six executions.

That correction is why sentinel discipline has a second layer. The dead
version's sentinel fired 16 times, so the patch executed; the fault was
clobbered one line later. Patch-executed is not fault-injected, exactly as
counter-fired is not patch-executed. Each check catches the layer below it and
is blind to the layer above.

Time to detection moved from 10.4 s to 17.9 s between two runs of identical code
on identical inputs, and individual faults roughly doubled. That is thermal
behaviour on a laptop GPU, not detection power, and should be read as an order
of magnitude.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/throughput-dark.svg">
  <img alt="Wall time for six configurations, run A solid and run B faded, and the 15 percent kill criterion drawn as a scale: the eager comparison at 14.6 and 16.6 percent straddles the bar, the graphed comparison at 10.6 and 11.5 percent sits below it." src="docs/img/throughput-light.svg" width="840">
</picture>

<details>
<summary>This number took four benchmark designs, and three of them were biased</summary>

Each design was a correct fix for the previous one's bias and each introduced a
new bias of its own.

**Blocked.** All samples of one configuration, then the next, so drift mapped
straight onto the ratio. Two runs of identical code disagreed by 1.75x on every
eager configuration, which was enough to publish, and then retract, a claim that
CUDA graphs gave vLLM's batch-invariant mode no benefit.

**Interleaved.** Shuffled under a seed so drift hits both arms. But that put
5 GB of this benchmark's own torch context in front of every external sample:
with `empty_cache()` a following vLLM sample ran about twice as slow, and
without it vLLM failed to launch at all.

**Isolated.** Every measurement in its own process, the parent holding no device
memory, with a 5000 MiB free-VRAM floor asserted per sample. That put Triton's
JIT inside the timed region and produced the impossible reading that invariant
mode was **faster** than the mode it constrains, at 0.95x.

**Isolated with warmup.** One untimed pass per process. The ratio is 1.10x and
every spread is at or below 1.24x.

The third design's impossible number is why this is legible at all. A bias that
produces a plausible figure is one you publish; a bias that produces "invariant
faster than fast" is one you cannot.

**On the kill criterion.** `docs/kickoff/01-PRD.md` section 11 set it before
anything was built. Across every design this figure has read 16.1, 16.2, 14.2,
14.6 and 16.6 percent against eager. An earlier version of this section said the
corrections had moved it decisively below the bar; the next run gave 16.6, so
that was a claim built on two samples of a quantity that varies by more than its
distance to the threshold. Two qualifications, neither an excuse: the criterion
reads "after the CUDA-graph pass" and this engine never received one, and the
prescribed response, publishing ratios rather than absolute tokens per second,
was already in force. What was missing was the disclosure.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/coverage-dark.svg">
  <img alt="19 of 25 two-grams and 43 of 79 three-grams reached, split between the swarm campaign and the eviction campaign, with probe-only cells outlined and never-reached cells hatched. The denominator was corrected from 27 and 84." src="docs/img/coverage-light.svg" width="840">
</picture>

<details>
<summary>Why the eviction campaign counts as exploration and the probes do not, decided on evidence after the obvious argument turned out to be false</summary>

**The rule:** a campaign phase is credited as exploration to the extent it
reaches n-grams *outside* the subsystem it was written to target. A phase that
only hits what it was built to hit is a probe wearing a campaign's name.

The eviction campaign was nearly excluded on an argument that turned out to be
false. The swarm campaign misses ten 2-grams, six of which involve `evict`, and
the eviction campaign adds exactly four. Four added against four non-eviction
misses is the kind of tidy correspondence that reads as confirmation.

Reading the identities refutes it. The four added are `cache_hit->evict`,
`evict->chunk`, `cache_hit->chunk` and `resume->cache_hit`: **two of the four
involve no eviction**. At 3-grams it is six of fifteen. The phase drives heavy
prefix sharing under allocation pressure, which reaches cache and resume
interleavings the uniform swarm generator rarely produces. So it is credited.

**This could not have been settled from committed data.** The eviction phase
kept its own `Coverage` object and never wrote it into the artifact, so nothing
published carried the numbers needed to decide the project's own headline. It
had to be re-measured, and the artifact now carries `eviction_campaign.coverage`.

Transitions: 12 of 13 reached by the standard campaign unaided, 13 of 13 once a
targeted probe is added, 2 declared unreachable with a written argument. Only
`(PREFILLING, PREEMPT_RC)` needs the probe. Boundary predicates reach 17 of 20.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/certification-dark.svg">
  <img alt="Seven boundary cases against vLLM batch-invariant mode: five clean, and two diverged at 44 and 45 co-resident with logprob deltas of 4.685e-02 and 3.906e-02 and no token divergence. The max-num-seqs control restores reproducibility at 8." src="docs/img/certification-light.svg" width="840">
</picture>

Filed as [vllm-project/vllm#51187](https://github.com/vllm-project/vllm/issues/51187),
cross-referenced on the batch-invariance tracker
[#27433](https://github.com/vllm-project/vllm/issues/27433#issuecomment-5195555951).
**Everything here is scoped exactly as that issue is scoped. If this section ever
says more than the issue does, the issue is right and this is wrong.**

<details>
<summary>The first result was withdrawn, what was retracted after that, and what could not be determined from outside the engine</summary>

**Withdrawn.** An earlier version reported vLLM's deterministic mode clean across
seven boundary cases. The certifier submitted every request sequentially and
blocking, so nothing ever cohabited: cases named "co-batched" and "batch 31" ran
at effective batch 1, and there was no positive control, so the certifier had
never been shown able to fail. It now submits concurrently, measures co-residency
from `vllm:num_requests_running`, refuses to score a case whose witness never
exceeded one running request, and compares **every pair** of repeats.

**The finding.** Byte-identical requests, byte-identical order, warm cache,
identical-workload priming before repeat 0, one unchanged server, five repeats.
At 44 co-resident: 3 distinct outputs of 5, max delta 4.685e-02. At 45: 2 of 5,
3.906e-02. No token divergence. Differing pairs are later-versus-later, so this
is not a first-batch effect. Intermittent at about 1 lifetime in 3, measured
twice independently, once at a clean committed revision.

**Ruled out.** CUDA graphs, prefix caching, KV pressure, chunked prefill by
budget, request order, warmup. Negative controls: sequential submission clean
7 of 7, default mode fails the same observable 7 of 7, 36 co-resident clean
21 of 21.

**Retracted.** The small-workload divergences at 15 to 16 co-resident are
withdrawn. Priming with byte-identical traffic before repeat 0 eliminates them:
they were substantially a first-batch artifact. Only the two large workloads
carry a finding, and only those are in the filed issue.

**Not determined.** The mechanism, which cannot be seen from outside the engine
and is not guessed at here or in the issue. Whether there is a threshold or a
probability rising with resident batch size, since intermediate widths were
sampled one lifetime per point. **Only the `--max-num-seqs` comparison is a
controlled single-variable result.** Everything else here is observational.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/thesis-dark.svg">
  <img alt="Eighteen tiles for seventeen findings, placed by finding index and laned by who found them: eight from this project's own machinery, five from an outside audit, one from an anomaly, two from readers not looking for defects, and two from this project auditing itself later. Findings 9 to 15 came from outside the repository." src="docs/img/thesis-light.svg" width="840">
</picture>

Lockstep exists because engines claim a compatibility surface wider than their
test surface. That is not a hypothesis about other people's code. It happened
seventeen times in this repository, and the distribution above is the most useful
thing in the table: **most were caught by a run contradicting a declaration, and
the ones that were not are worth more than their fixes.**

<details>
<summary>All seventeen, and why four of them no check here could have caught</summary>

| # | Declared | Actually |
|---|---|---|
| 1 | A fidelity corpus exercising the fixed-split attention kernel | No prompt exceeded 512 tokens, so the split-combine fold never executed in any fidelity number |
| 2 | A VRAM table across five chunk sizes | Prompts were shorter than the largest chunk, so that row was byte-identical to unchunked |
| 3 | Eviction under memory pressure, tested | Admission counted only free blocks, so a request serviceable by evicting was refused and the path was unreachable |
| 4 | A copy-on-write boundary case passing | Nothing called `fork`; the test exercised dead code |
| 5 | A coverage denominator derived from the lifecycle | The transition table was wrong four times, in both directions |
| 6 | A mutation trial gated on the mutated path executing | The gate passed and the patch never ran; the fault rebound a name in the module that defined it while the caller held its own import |
| 7 | A mutant killed, counted toward the score | The operator was a proxy whose firing condition varied with the schedule, so the relations caught the proxy rather than the fault |
| 8 | A nine-command CLI, and a reproduce line naming one | No `lockstep` executable existed. The tests asserted the string rather than the thing it named |
| 9 | vLLM certified across seven boundary cases, two named for co-batching | Every request was submitted sequentially. The cases ran at effective batch 1, and there was no positive control |
| 10 | A proven-equivalent mutant, with an equivalence argument | A dead store. The published proof described a mechanism absent from the code |
| 11 | A concurrency cap in the security config | `max_concurrency: 1` sat unread from week 8, and enforcing it later revealed it also made batching impossible |
| 12 | One edit applied to two files | It matched in one and silently no-op'd in the other, so an artifact shipped without the fields proving its own batch witness fired |
| 12b | A coverage denominator checked against reality | The check ran in one direction only. `(ADMITTED, PREEMPT_RC)` was unreachable and inflated every published coverage number |
| 13 | An observable comparing repeated runs for identical output | Every comparison was repeat 0 against a later repeat, so nondeterminism and a differing first batch were indistinguishable |
| 14 | Abstraction sized for what the design needs | Four definitions with no caller, and `Comparison.notes` computed every comparison and never read, one of which reports a narrowed observable |
| 15 | Five CI gates, declared in the architecture doc | No `.github/` directory existed, so none of them ran |
| 16 | An evidence table mapping every figure to its artifact | Two rows named deleted files and one carried 55 of 55 against an actual 65 of 65, across 12 commits |
| 17 | Every claim scoped to the tuple in `env.lock` | `envlock.py` fingerprinted the harness process. Every certify artifact printed torch 2.6.0 beside a claim about a server on torch 2.11.0 |

**Four could not have been caught by any check this repository had.** Number 9
was invisible because the certifier's own passes were the evidence it was
working, and a green result is the hardest thing to doubt. Number 11 is the same
shape one layer down, in the file that exists to constrain the certifier. Number
12 is the tooling lying about its own success, which no test can catch because
the test would have to distrust the edit that wrote it. Number 13 was made
visible not by a check but by an anomaly: divergence magnitudes reproducing to
four significant figures across independent processes, which is not what
nondeterminism looks like.

**Row 17 cost something outside this repository.** Clean-revision discipline was
applied to the harness rigorously enough that a whole artifact exists to pin a
harness SHA, while the subject was a released wheel under no revision discipline
at all. So nobody asked whether upstream main had moved, a finding was filed
against a version whose fix had already merged, and the mechanism was supplied by
an outside reproducer. It was then verified here independently and pushed past
the report that prompted it, with a pre-registration committed before anything
ran and a correction to the granularity of the claim.

Each was found the same way: an execution counter, an assertion, or an
observed-versus-feasible check contradicted something written down and believed.
That is the entire argument for the tool, and it is why the engine carries 25
execution counters that every relation asserts against, and why a relation which
passes without firing its own mechanism is treated as a failure.

The deterministic modes in vLLM and SGLang are tested by people at least as
careful as the author of this repository. The question is not whether they were
careless. It is whether anyone has checked which of their declared surface
actually executes.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/limits-dark.svg">
  <img alt="Eight limits: no novelty on the kernels, no cross-hardware guarantee, no throughput superiority, one model only, no copy-on-write, ten mutation operators rather than fifty, coverage against a derived model, and execution counters proving only that the path ran." src="docs/img/limits-light.svg" width="840">
</picture>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/evidence-dark.svg">
  <img alt="Nine committed artifacts under evidence/ and what each one backs, from the relation runs and the fuzz campaign to the three runs the upstream filing rests on. A replay command recomputes a committed trajectory hash on a fresh clone with no GPU." src="docs/img/evidence-light.svg" width="840">
</picture>

<details>
<summary>What replay prints, and why one artifact is pinned to the commit that fixed it</summary>

```console
$ uv run ./lockstep replay evidence/case-witness.json

  recorded          trajectory 82ee4d7d1d99fe766991aa1a8258d1c8
  recorded env      sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0
  this env          sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0
  recorded engine   c835a7361031
  this engine       05621331aeb8
  minimality        reproduces=True 1-minimal=False checks=2

  OK: trajectory hash 82ee4d7d1d99fe76 matches the artifact
```

`1-minimal=False` is correct: this artifact is a witness rather than a finding,
so there is no failing property for ddmin to shrink against. The two environment
lines agreeing is the point of printing both, and the two engine revisions
disagreeing is the stronger result, since the hash held across the change.

The `uv run` prefix is load-bearing. `./lockstep` starts `#!/usr/bin/env python3`,
the system interpreter, and on a machine with a different torch the replay will
run against it and say so on the `this env` line.

The witness is deliberately not trivial: two prompts cross the 512-token split
boundary, prefill is chunked, one request is preempted mid-decode, and a
544-token prefix is shared.

`evidence/case-0003.json` is the other kind: a real finding, minimized to one
request and proven 1-minimal in 788 checks. **No single checkout reproduces it**,
and not because anything is missing: the harness that produces and replays cases
landed *after* the fix. What was verified rather than inferred from commit dates
is that removing the oversized-request guard from `Scheduler.submit` at HEAD, 8
lines, makes the minimized case raise `OutOfBlocks` with the recorded condition.
That is in the artifact's `provenance` block with both SHAs, as a one-hunk change
a reader can apply.

</details>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/gates-dark.svg">
  <img alt="Eight checks automated on every push, four manual gates that need a GPU, and the note that for most of this project none of the declared CI gates ran because no .github directory existed." src="docs/img/gates-light.svg" width="840">
</picture>

Three things are checkable in under a minute on any machine, with no CUDA and no
model download:

```sh
python3 scripts/verify_no_gpu.py
```

It recomputes the coverage denominators from the declared transition relation and
asserts they are 25 and 79, prints a sha256 for every artifact in `evidence/`
alongside the engine revision that produced it, and rebuilds
`evidence/case-0003.json` to show the finding arithmetically. The rest is the
substance and it is not checkable cheaply. Pinned dependencies do not fix that;
naming the split is the honest alternative.

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/prior-art-dark.svg">
  <img alt="Thinking Machines, vLLM and SGLang, GRIEF, LLM-42, MarginGate, and the swarm testing, ddmin and mutation testing literature. What differs here is that the schedule is supplied rather than observed, so minimisation is exact rather than confirmed." src="docs/img/prior-art-light.svg" width="840">
</picture>

<details>
<summary>Links, and what each source actually says</summary>

- [Thinking Machines, *Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and [`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops), for the diagnosis this project takes as given.
- [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/), `VLLM_BATCH_INVARIANT=1`, beta as of July 2026, tracker [#27433](https://github.com/vllm-project/vllm/issues/27433). Its attention operators support `PIECEWISE` but not `FULL`, which is why throughput is measured both eager and graphed.
- [SGLang deterministic inference](https://docs.sglang.ai/advanced_features/deterministic_inference.html) and [#22819](https://github.com/sgl-project/sglang/issues/22819), KV corruption at a radix cache block boundary where the request whose `prefix_len` equals the block size is corrupted while `prefix_len == 0` requests in the same batch are not. Both halves of that shape are first-class here.
- [GRIEF](https://arxiv.org/abs/2605.11202), greybox fuzzing of LLM serving systems, whose oracle applies controlled replay with log-probability checks. 15 vulnerabilities, 10 confirmed, 2 CVEs. The differentiator above is about the replay mechanism, not about GRIEF's effectiveness.
- [LLM-42](https://arxiv.org/abs/2601.17768), the argument that batch invariance is over-constrained. This project does not rebut it; it concedes it and measures the price, which is what the throughput panel is.
- [MarginGate](https://arxiv.org/abs/2605.30218), sparse margin-triggered verification, adjacent to F1's near-tie threshold: both key on the top-1-to-top-2 margin.
- Groce et al., *Swarm Testing* (ISSTA 2012). Zeller and Hildebrandt, ddmin (IEEE TSE 2002). Just et al., *Are Mutants a Valid Substitute for Real Faults?* (FSE 2014).

</details>

---

## Install

Requires an NVIDIA GPU. Developed on sm_89 with 8 GB; nothing is claimed
elsewhere. `make setup` does all of this and `make help` lists every target.

```sh
uv sync
python3 scripts/download_weights.py          # pinned by revision SHA, checksummed
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python3 scripts/build_fp64_reference.py    # about 90 seconds
make check                                   # everything the CPU gate runs, locally
```

Figures regenerate with `python3 docs/img/build_figures.py`. The visual summary of
a run is `results/report.html`, one self-contained 84 KB file with no network
dependencies, built with `python3 -m report.html`.
