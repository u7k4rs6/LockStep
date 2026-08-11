I instrumented the engine, so this is now per token rather than by correlation.

Two patches to a 0.26.0 venv: `num_tokens` at every RMSNorm launch, and
`scheduler_output.num_scheduled_tokens` per step together with a hash of each
request's prompt, since vLLM assigns a fresh `req_id` per submission and the same
request otherwise cannot be followed from one repeat to the next. 16 server
lifetimes of 5 byte-identical repeats each: 10 default, 2 at `--max-num-seqs 8`,
4 with the FlashInfer sampler disabled.

**Result.** Two repeats return identical logprobs if and only if every token was
reduced at the same RMSNorm block width in both. 1120 pairwise comparisons across
all 16 lifetimes, no exceptions in either direction, so it holds across all three
configurations rather than only the default one. At case level, a case diverged
exactly when some repeat put 256 or more tokens into a launch, in all 112
case-lifetimes.

**Why a fixed workload varies at all.** The 44 and 45 co-resident cases leave 270
and 274 uncached prefill tokens. The scheduler sometimes puts nearly all of them
into one launch and sometimes splits them, so a byte-identical workload lands on
both sides of 256 within a single server lifetime. That is the step between your
operator result, which is batch-size dependence, and this issue, which is
repeat-to-repeat variance at a fixed workload.

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

The cache-hit row has the lowest uncached total of any case and the highest max
launch of the small ones, because its two requests are fully cached and decode
while the fillers prefill, so decode tokens fold into the same launches. In this
workload a filler contributes about 9 uncached tokens and a prefix-sharing
request about 4.9, so sequence count is not even monotonic in the governing
quantity. Two requests of 128 fresh tokens each would cross at a co-residency
of 2.

So `--max-num-seqs 8` is clean because it caps a launch at 85 tokens, not because
8 sequences are few. Every crossing is a prefill or mixed step, and decode steps
sit at 44 or 45, so the perturbation starts in prefill and reaches the decode
logprobs through the KV cache. That is why the emitted token ids never move.

**`VLLM_USE_FLASHINFER_SAMPLER=0` does not fix it.** 8 instrumented lifetimes,
4 at each of two flush granularities of my own tracing: 7 of the 8 diverged, with
a maximum launch of 272 tokens. It neither prevents the divergence nor moves
packings below 256. Your caution that the all-greedy path bypasses the sampling
call looks right.

**Your operator table is explained exactly by the launcher.** For
`fused_add_rms_norm` the block is `min(hidden_size, max_block_size)`, so both of
your hidden sizes move when `max_block_size` flips and you get 32/32. For the
non-fused `rms_norm` it is `min(hidden_size / vec_size, max_block_size)` with
`vec_size = gcd(16 / sizeof(scalar_t), hidden_size)`, which is 8 at FP16: 512/8
is 64 and never reaches the 256 clamp, while 4096/8 is 512 and does. One of your
two hidden sizes is affected and the other is not, which is your 16/32.

That also says which half of your table the engine can reach. Under
`VLLM_BATCH_INVARIANT=1`, `rms_norm_batch_invariant` returns straight into
`ops.fused_add_rms_norm` whenever a residual is present, and only the
residual-free path goes to Triton. So the 32/32 op is the one carrying 56 of the
57 hidden-size-1024 RMSNorm launches per forward pass here, and the 16/32 op is
largely not reached through the engine at all.

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
