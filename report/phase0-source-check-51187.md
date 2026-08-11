# Phase 0 — source-level check of vllm#48391 against vLLM 0.26.0

Method: read the source, not the thread. Two sources of truth:

- Upstream C++ at `v0.26.0` and at `b6cbba8`, fetched from
  `raw.githubusercontent.com` (both HTTP 200), diffed locally.
- The Python actually installed in the subject environment,
  `/home/utkuputku/lockstep-extenv/vllmdet/lib/python3.12/site-packages/vllm`
  (`vllm-0.26.0.dist-info`, torch 2.11.0, triton 3.6.0, flashinfer_python 0.6.14).

The GitHub MCP server returned `Bad credentials`, so nothing here came through it.

---

## Q1 — Does the pre-fix launch pick block size from `num_tokens`, with a discontinuity at 256?

**Yes, in both kernels, and the discontinuity is exactly at `num_tokens == 256`.**

`csrc/libtorch_stable/layernorm_kernels.cu` @ `v0.26.0`, `rms_norm` launcher (line 251-252):

```cpp
  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
```

Same file, `fused_add_rms_norm` launcher (line 322-328):

```cpp
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
```

`num_tokens = input.numel() / hidden_size`, i.e. the token count of that
individual kernel launch. The block width is
`min(hidden_size, max_block_size)`, so the selection only bites when
`hidden_size > 256`. Qwen3-0.6B has `hidden_size = 1024`, so the block goes
**1024 threads → 256 threads** as a launch crosses 256 tokens. The reduction is a
block reduce over that width, so the per-thread strided partials and the
tree order both change. Different summation order, different rounding.

Two secondary points worth having:

- The 256 in the predicate is a **token count**; the 256 in `dim3 block` is a
  **thread count**. They are unrelated quantities that happen to share a
  literal. Only the first is the trigger.
- Qwen3's q_norm/k_norm are residual-free, so they take the OTHER launcher,
  whose width is `min(hidden_size / calculated_vec_size, max_block_size)` with
  `calculated_vec_size = gcd(16 / sizeof(scalar_t), hidden_size)`. At
  `head_dim = 128` and fp16 that is `gcd(8, 128) = 8`, so the width is
  `min(16, ...) = 16` either way and those launches are never affected.

  Corrected after the fact. This originally read `min(128, ...) = 128`, using
  the fused-add launcher's expression for a call that does not use it. The
  conclusion was right and the derivation was wrong, which is only visible once
  the two expressions are known to differ, and they differ by exactly the
  `vec_size` division that explains the reproducer's 16/32.

## Q2 — Is the fix gated on batch-invariant mode, or general?

**Gated. Batch-invariant mode only; the default path is untouched.**

`b6cbba8` makes the identical edit at both launch sites:

```cpp
  const bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  const int max_block_size =
      batch_invariant_launch ? 1024 : ((num_tokens < 256) ? 1024 : 256);
```

`vllm_is_batch_invariant()` (`csrc/core/batch_invariant.hpp`) is a cached read of
`VLLM_BATCH_INVARIANT`. When the flag is off the expression collapses to the old
one, so throughput at large batch is unchanged for everyone not asking for
determinism. The rest of the diff is a one-line hoist: `fused_add_rms_norm`
already computed `batch_invariant_launch` further down, and the fix just moves
that declaration above the block-size line and reuses it.

That hoist is the most informative line in the diff. Pre-fix, the
`fused_add_rms_norm` launcher **already consulted batch-invariant mode** — to
suppress the width-8 vectorized variant, because vectorization changes reduction
order:

```cpp
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  ...
    if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
        !batch_invariant_launch) {
      LAUNCH_FUSED_ADD_RMS_NORM(8, true);
    } else {
      LAUNCH_FUSED_ADD_RMS_NORM(0, true);
    }
```

So this was not a kernel that had been left out of batch invariance. It was a
kernel deliberately made batch-invariant along one axis, with a second
order-changing knob sitting eleven lines above the guard, missed. That is a much
more credible bug than "nobody thought about it".

## Q3 (the crux) — Under 0.26.0 + `VLLM_BATCH_INVARIANT=1` + `--enforce-eager`, does the C++ kernel actually run?

**Yes for `fused_add_rms_norm`. No for plain `rms_norm`.** The concern that the
Triton kernel takes over the whole op is half right, and the half that is wrong
is the half that carries essentially all of the traffic.

The dispatch chain, each link read in the installed 0.26.0 tree:

**1. `--enforce-eager` ⇒ custom ops enabled.** `config/vllm.py:1251-1258`:

```python
        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")
```

`--enforce-eager` sets `mode = CompilationMode.NONE`, so the second branch runs
and `custom_ops` becomes `["all"]`. `CustomOp.default_on()` is then true,
`RMSNorm.enabled()` is true, and `dispatch_forward` (`model_executor/custom_op.py:186-207`)
returns `self.forward_cuda`. `forward_native` is only reached when the op is
disabled, which is the inductor default, not the eager one.

**2. `forward_cuda` routes to the batch-invariant entry point.**
`model_executor/layers/layernorm.py:96-115`:

```python
    def forward_cuda(self, x, residual=None):
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (...)
            pass_weight = (
                self.pass_weight_add if residual is not None else self.pass_weight
            )
            return rms_norm_batch_invariant(
                x, self.weight.data if pass_weight else None,
                self.variance_epsilon, residual=residual,
            )
        return self.forward_native(x, residual)
```

**3. `rms_norm_batch_invariant` splits, and the residual branch calls straight
back into the C++ kernel.** `model_executor/layers/batch_invariant.py:848-855`:

```python
    if residual is not None:
        assert input.shape == residual.shape, (...)
        import vllm._custom_ops as ops

        ops.fused_add_rms_norm(input, residual, weight, eps)
        return input, residual
```

Only past that early return does the Triton path begin, at line 873, with
`BLOCK_SIZE = 1024` hardcoded — genuinely invariant, and irrelevant here.

`_custom_ops.py:269-276` closes it:

```python
def fused_add_rms_norm(input, residual, weight, epsilon) -> None:
    # Note: this func is batch invariant
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
```

The comment asserting batch invariance sits directly on the call that was not
batch invariant. And `CMakeLists.txt` @ v0.26.0 line 404 compiles
`csrc/libtorch_stable/layernorm_kernels.cu` — the file diffed above is the file
that ships.

**How much traffic takes the affected branch.** `models/qwen3.py:230-246`: layer 0
enters with `residual is None` and takes the Triton branch once; every
subsequent norm passes a residual. For 28 layers that is **1 Triton call and 56
C++ `fused_add_rms_norm` calls per forward pass**, all at `hidden_size = 1024`,
all with a block width that flips 1024→256 when a launch crosses 256 tokens.

So the answer to the question as posed: the flag does **not** route both ops to
Triton in eager mode on 0.26.0. It routes the residual-free op to Triton and
leaves the residual op on the C++ kernel that #48391 fixes. **The proposed
mechanism is reachable, and nothing below has to change.**

## Two things this does not establish

Phase 0 was a reachability check and it passed. It did not test the bridging
claim, which is still unstated by anyone: that per-step scheduled token counts
vary across repeats of a byte-identical workload and straddle 256 at 44-45
co-resident but not at `--max-num-seqs 8`. A reachable mechanism that is never
triggered explains nothing, and 994 prompt tokens over 45 requests is not
obviously a workload that puts 256 tokens in one launch. That is Phase 2's job.

Separately, one discrepancy noticed in passing, not chased:
`evidence/upstream-finding.json` records the subject model as "Qwen3-0.6B fp16",
while `weights/Qwen3-0.6B/config.json` says `"torch_dtype": "bfloat16"`. It does
not affect the block-size argument, which is dtype-independent, but one of the
two is wrong and the server's resolved dtype should be read from a log rather
than assumed.
