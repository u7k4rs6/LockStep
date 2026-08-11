You reproduced this on your own hardware and then handed over the mechanism,
which is further than I could get from outside the engine. What was missing was
the step connecting your operator result to repeat-to-repeat variance at a fixed
workload. I have now instrumented that step and measured it per token.

**Your operator table falls out of the two launchers exactly.** For
`fused_add_rms_norm` the block is `min(hidden_size, max_block_size)`, so both of
your hidden sizes move when `max_block_size` flips and you get 32/32. For the
non-fused `rms_norm` it is `min(hidden_size / vec_size, max_block_size)` with
`vec_size = gcd(16 / sizeof(scalar_t), hidden_size)`, which is 8 at FP16: 512/8
is 64 and never reaches the 256 clamp, while 4096/8 is 512 and does. One of your
two hidden sizes is affected and the other is not, which is your 16/32, and the
split falls exactly where the source says it should.

Two independent reasons that second op cannot matter for the end-to-end runs.
Under `VLLM_BATCH_INVARIANT=1`, `rms_norm_batch_invariant` returns straight into
`ops.fused_add_rms_norm` whenever a residual is present and sends only the
residual-free path to Triton, so the 32/32 op carries 56 of the 57
hidden-size-1024 RMSNorm launches per forward pass here. Separately, and without
appealing to dispatch at all, 1024/8 is 128, so `min(128, 1024)` and
`min(128, 256)` are both 128 and the non-fused op could not flip at this hidden
size even if the engine did reach it.

**The measurement.** Two patches to a 0.26.0 venv: `num_tokens` at every RMSNorm
launch, and `scheduler_output.num_scheduled_tokens` per step together with a hash
of each request's prompt, since vLLM assigns a fresh `req_id` per submission and
the same request otherwise cannot be followed from one repeat to the next. 16
server lifetimes of 5 byte-identical repeats each: 10 default, 2 at
`--max-num-seqs 8`, 4 with the FlashInfer sampler disabled.

Two repeats return identical logprobs if and only if every token was reduced at
the same RMSNorm block width in both: 1120 pairwise comparisons across all 16
lifetimes, no exceptions either way, so it holds across all three configurations
and not just the default. Read at case granularity, all 112 case-lifetimes
diverged exactly when some repeat put 256 or more tokens into a launch.

**Why a fixed workload varies at all.** The 44 and 45 co-resident cases leave 270
and 274 uncached prefill tokens. The scheduler sometimes puts nearly all of them
into one launch and sometimes splits them, so a byte-identical workload lands on
both sides of 256 within a single server lifetime.

**The title's "at ~44 co-resident sequences" is causally wrong, and that is my
error.** The governing quantity is tokens scheduled into one launch, whatever
their origin, which is what the kernel actually branches on. It is not sequence
count, and it is not uncached prefill either:

| workload | co-resident | uncached | max tokens in a launch | ever crossed 256 |
|---|---:|---:|---:|:--:|
| cache hit covering the full prompt | 15 | 117 | 147 | no |
| the other four small boundary cases | 15 to 16 | 130 to 140 | 128 to 136 | no |
| batch 31 | 44 | 270 | 268 | yes |
| batch 32 | 45 | 274 | 272 | yes |
| any of the above, `--max-num-seqs 8` | 8 | unchanged | 85 | no |

The cache-hit row inverts the uncached ordering because its two requests are
fully cached and decode while the fillers prefill, so decode tokens fold into the
same launches. A filler contributes about 9 uncached tokens and a prefix-sharing
request about 4.9, so sequence count is not even monotonic in the governing
quantity: two requests of 128 fresh tokens each would cross at a co-residency of
2, and `--max-num-seqs 8` is clean because it caps a launch at 85 tokens rather
than because 8 sequences are few.

Every crossing is a prefill or mixed step, and decode steps sit at 44 or 45, so
the perturbation starts in prefill and reaches the decode logprobs through the KV
cache. That is why the emitted token ids never move.

**`VLLM_USE_FLASHINFER_SAMPLER=0` does not fix it.** All 4 of the sampler-off
lifetimes above diverged, with a maximum launch of 272 tokens, so it neither
prevents the divergence nor moves packings below 256. A further 4 lifetimes at a
different flush granularity of my own tracing behaved the same way, 3 of those 4
diverging. Your caution that the all-greedy path bypasses the sampling call looks
right.

Our magnitudes match to every digit reported, 4.685e-02 and 3.906e-02 for the 44
and 45 cases, on your 4090 against my 4060 laptop and CUDA 12.9 against 13.0. Two
fixed reduction orders give a deterministic delta, which is what that agreement
means and what noise would not produce.

I said I would re-run on nightly and I have not. Between your 5/5 and 35/35 and
the measurement here, I do not think a fourth confirmation adds anything.

Broken in 0.26.0, fixed by #48391 (b6cbba8), and that fix shipped in v0.27.0 on
2026-08-10. Please close as fixed by #48391. One line in the determinism docs may
still be worth it for anyone pinned to 0.26.0 or earlier: this reaches any
`VLLM_BATCH_INVARIANT=1` user whose per-step scheduled token count varies across
256, at any hidden size strictly greater than 256. At exactly 256 both branches
give `min(256, ...) = 256` and the model is unaffected. It is a scheduling
condition rather than a property of the model or the request pattern, which is
why it presented as a sequence-count threshold from outside.
