"""Qwen3-0.6B forward pass on invariant kernels only.

Week 1 runs a single sequence, but the kernel shapes are the ones the
architecture doc requires rather than a simpler stand-in that would have to be
replaced: fixed-BLOCK_K GEMMs with no split-K including lm_head, one CTA per row
for RMSNorm, elementwise RoPE, and the fixed-split attention family. Paged KV,
batching, and the scheduler are week 2 and are deliberately absent; the KV cache
here is one contiguous tensor per layer.

Torch operations used in the forward pass, and why each is safe under I1:

  * `embed[token_ids]`: a gather. No arithmetic, no reduction.
  * `h + o` for residual: elementwise add over matching shapes.
  * `.view` / `.reshape` / `.contiguous`: layout only.

Every operation that reduces goes through engine/kernels. There is no torch
`matmul`, `softmax`, `mean`, or `sum` anywhere in this path, because those are
the ops whose kernel selection is shape-dependent and therefore batch-dependent.

Numerics deviations from HF `Qwen3ForCausalLM`, all deliberate, all in the
direction of the fp64 reference that F1 measures against:

  * RMSNorm keeps fp32 through the weight multiply; HF rounds to fp16 first.
  * cos/sin tables are built in fp64 on CPU and held in fp32; HF casts them to
    the activation dtype, so at fp16 it rotates by a 10-bit-mantissa angle.

HF is a sanity check, never ground truth, because it is not shape-invariant
(architecture doc section 7). These deviations are why the HF token-match number
in bench/fidelity.py is a sanity check rather than a target of 100 percent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.kernels import registry
from engine.kernels.attention import attention
from engine.kernels.gemm import linear
from engine.kernels.rmsnorm import rmsnorm
from engine.kernels.rope import apply_rope
from engine.kernels.swiglu import swiglu

ENGINE_DTYPE = torch.float16


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_file(cls, path: Path) -> "Qwen3Config":
        raw = json.loads(Path(path).read_text())
        if raw.get("model_type") != "qwen3":
            raise ValueError(f"expected a qwen3 config, got {raw.get('model_type')!r}")
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw["tie_word_embeddings"],
        )


class KVCache:
    """One contiguous K and V tensor per layer, for one sequence.

    Week 2 replaces this with a paged block table and refcounts. The interface
    kept here is deliberately minimal so that replacement is not a rewrite of the
    forward pass.
    """

    def __init__(self, cfg: Qwen3Config, max_len: int, device: torch.device):
        shape = (max_len, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [
            torch.zeros(shape, dtype=ENGINE_DTYPE, device=device)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v = [
            torch.zeros(shape, dtype=ENGINE_DTYPE, device=device)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.max_len = max_len
        self.length = 0

    def reset(self) -> None:
        self.length = 0


def _rope_tables(
    cfg: Qwen3Config, max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of shape [max_len, head_dim/2], built in fp64 on CPU.

    Held in fp32. The tables are a function of position and rope_theta alone, so
    they are computed once and are identical for every request; building them in
    fp64 removes the table itself as a source of error in the F1 measurement.
    """
    half = cfg.head_dim // 2
    exponent = torch.arange(0, half, dtype=torch.float64) * 2.0 / cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta**exponent)
    positions = torch.arange(max_len, dtype=torch.float64)
    freqs = positions[:, None] * inv_freq[None, :]
    return (
        freqs.cos().to(torch.float32).to(device),
        freqs.sin().to(torch.float32).to(device),
    )


class Qwen3:
    """The engine's model. Batch-1 in week 1; the kernels are not."""

    def __init__(self, weights_dir: Path, device: str = "cuda", max_len: int = 4096):
        weights_dir = Path(weights_dir)
        self.cfg = Qwen3Config.from_file(weights_dir / "config.json")
        self.device = torch.device(device)
        self.max_len = max_len

        assert self.cfg.head_dim == registry.HEAD_DIM, (
            f"registry pins BLOCK_D={registry.HEAD_DIM}, model has "
            f"head_dim={self.cfg.head_dim}"
        )

        self.weights = self._load(weights_dir / "model.safetensors")
        self.cos, self.sin = _rope_tables(self.cfg, max_len, self.device)
        self.sm_scale = 1.0 / math.sqrt(self.cfg.head_dim)

    # -- loading --------------------------------------------------------------

    def _load(self, path: Path) -> dict[str, torch.Tensor]:
        """Load safetensors and cast to the engine dtype.

        `safetensors` is a transitive dependency of transformers, which the
        security doc scopes to "loading and tokenizer only"; this is that use.
        No pickle format is ever opened: the pinned revision publishes
        safetensors and scripts/download_weights.py refuses a lockfile that names
        a .bin.

        The checkpoint is bf16. bf16 carries 7 mantissa bits and fp16 carries 10,
        so the cast is exact for every value inside fp16's normal range; values
        below it would flush, and `weight_cast_report` counts them so the number
        is measured rather than asserted.
        """
        from safetensors.torch import load_file

        raw = load_file(str(path), device="cpu")
        self.checkpoint_dtype = str(next(iter(raw.values())).dtype)

        # Measured while streaming, one tensor at a time, and the raw checkpoint
        # is released as we go. This machine has ~5 GB of headroom and the
        # checkpoint is 1.5 GB; keeping it resident to measure it later is how a
        # reference build ends up in swap.
        elements = 0
        inexact = 0
        max_abs = 0.0
        max_rel = 0.0

        weights: dict[str, torch.Tensor] = {}
        for name in list(raw.keys()):
            tensor = raw.pop(name)
            cast = tensor.to(ENGINE_DTYPE)

            exact = tensor.to(torch.float64)
            diff = (exact - cast.to(torch.float64)).abs()
            elements += diff.numel()
            inexact += int((diff > 0).sum())
            if diff.numel():
                max_abs = max(max_abs, float(diff.max()))
                denom = exact.abs().clamp_min(torch.finfo(torch.float64).tiny)
                max_rel = max(max_rel, float((diff / denom).max()))
            del tensor, exact, diff

            weights[name] = cast.to(self.device).contiguous()

        self._cast_report = {
            "checkpoint_dtype": self.checkpoint_dtype,
            "engine_dtype": str(ENGINE_DTYPE),
            "elements": elements,
            "elements_not_exactly_representable": inexact,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        }

        if self.cfg.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        return weights

    def weight_cast_report(self) -> dict[str, object]:
        """How much the checkpoint -> fp16 cast cost, measured across all weights.

        The fp64 reference is built from the *fp16* values this engine holds, so
        this error sits deliberately outside the F1 number: F1 measures the
        forward pass, not the weight cast. Reporting it separately is what keeps
        that choice honest rather than convenient.
        """
        return dict(self._cast_report)

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    # -- forward --------------------------------------------------------------

    def forward(
        self, token_ids: torch.Tensor, start_pos: int, cache: KVCache
    ) -> torch.Tensor:
        """Run `token_ids` at positions [start_pos, start_pos + len), return logits.

        Prefill passes start_pos=0 with the whole prompt; a decode step passes
        one token at the current length. Chunked prefill in week 4 passes an
        arbitrary interior offset, and I2 says the bits must not care.
        """
        cfg = self.cfg
        seq_len = token_ids.shape[0]
        kv_len = start_pos + seq_len
        assert kv_len <= cache.max_len, f"kv_len {kv_len} exceeds cache {cache.max_len}"

        h = self._w("model.embed_tokens.weight")[token_ids]  # gather, no arithmetic
        cos = self.cos[start_pos:kv_len]
        sin = self.sin[start_pos:kv_len]

        for layer in range(cfg.num_hidden_layers):
            p = f"model.layers.{layer}"

            normed = rmsnorm(
                h, self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps, "rmsnorm.hidden"
            )

            q = linear(normed, self._w(f"{p}.self_attn.q_proj.weight")).view(
                seq_len, cfg.num_attention_heads, cfg.head_dim
            )
            k = linear(normed, self._w(f"{p}.self_attn.k_proj.weight")).view(
                seq_len, cfg.num_key_value_heads, cfg.head_dim
            )
            v = linear(normed, self._w(f"{p}.self_attn.v_proj.weight")).view(
                seq_len, cfg.num_key_value_heads, cfg.head_dim
            )

            # Qwen3 normalizes per head before rotation, unlike Qwen2.
            q = rmsnorm(q, self._w(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")
            k = rmsnorm(k, self._w(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")

            q = apply_rope(q.contiguous(), cos, sin)
            k = apply_rope(k.contiguous(), cos, sin)

            cache.k[layer][start_pos:kv_len] = k
            cache.v[layer][start_pos:kv_len] = v

            attn = attention(
                q,
                cache.k[layer][:kv_len],
                cache.v[layer][:kv_len],
                q_start=start_pos,
                sm_scale=self.sm_scale,
            )

            h = h + linear(
                attn.reshape(seq_len, cfg.num_attention_heads * cfg.head_dim),
                self._w(f"{p}.self_attn.o_proj.weight"),
            )

            normed = rmsnorm(
                h,
                self._w(f"{p}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
                "rmsnorm.hidden",
            )
            gate = linear(normed, self._w(f"{p}.mlp.gate_proj.weight"))
            up = linear(normed, self._w(f"{p}.mlp.up_proj.weight"))
            h = h + linear(swiglu(gate, up), self._w(f"{p}.mlp.down_proj.weight"))

        cache.length = kv_len

        h = rmsnorm(h, self._w("model.norm.weight"), cfg.rms_norm_eps, "rmsnorm.hidden")
        return linear(h, self._w("lm_head.weight"), config="gemm.lm_head")

    # -- decode ---------------------------------------------------------------

    def generate_greedy(
        self, prompt_ids: list[int], max_new_tokens: int, eos_token_ids: set[int] | None = None
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Greedy batch-1 decode. Returns generated ids and the logits per step.

        Ties break by lowest token ID, matching the sampler rule in the PRD's
        must-have table so that greedy here and temperature-0 sampling later
        cannot disagree. `torch.argmax` already returns the first maximal index,
        which is that rule; it is asserted rather than assumed in the tests.
        """
        cache = KVCache(self.cfg, self.max_len, self.device)
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)

        logits = self.forward(ids, 0, cache)
        step_logits = [logits[-1].clone()]
        generated: list[int] = []
        eos = eos_token_ids or set()

        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(step_logits[-1].float()).item())
            generated.append(next_id)
            if next_id in eos:
                break
            token = torch.tensor([next_id], dtype=torch.long, device=self.device)
            logits = self.forward(token, cache.length, cache)
            step_logits.append(logits[-1].clone())

        return generated, step_logits
