"""Qwen3-0.6B forward pass on invariant kernels only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.kernels import registry
from engine.kv import paged
from engine.kernels.attention import attention
from engine.kernels.gemm import linear
from engine.kernels.rmsnorm import rmsnorm
from engine.kernels.rope import apply_rope
from engine.kernels.swiglu import swiglu

ENGINE_DTYPE = torch.float16

CAST_REPORT_CHUNK = 1 << 24


def write_kv(
    pool: "paged.PagedKVCache",
    uid: str,
    layer: int,
    start_pos: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Scatter freshly computed K and V into their paged slots."""
    table = pool.sequences[uid].block_ids
    written = 0
    total = k.shape[0]
    while written < total:
        position = start_pos + written
        logical, slot = divmod(position, pool.block_size)
        take = min(pool.block_size - slot, total - written)
        physical = table[logical]
        if layer == 0:
            pool.assert_exclusive(uid, physical)
        pool.k[layer][physical, slot : slot + take] = k[written : written + take]
        pool.v[layer][physical, slot : slot + take] = v[written : written + take]
        written += take


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
    """A single-sequence view onto a PagedKVCache, for batch-1 paths."""

    def __init__(
        self,
        cfg: Qwen3Config,
        max_len: int,
        device: torch.device,
        block_size: int = paged.DEFAULT_BLOCK_SIZE,
    ):
        blocks = -(-max_len // block_size)
        self.pool = paged.PagedKVCache(
            num_blocks=blocks,
            block_size=block_size,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=ENGINE_DTYPE,
        )
        self.uid = "seq"
        self.pool.create(self.uid)
        self.pool.reserve(self.uid, max_len)
        self.max_len = max_len
        self.length = 0

    @property
    def block_table(self) -> torch.Tensor:
        return self.pool.block_table(self.uid)

    def reset(self) -> None:
        self.length = 0


def _rope_tables(
    cfg: Qwen3Config, max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of shape [max_len, head_dim/2], built in fp64 on CPU."""
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


    def _load(self, path: Path) -> dict[str, torch.Tensor]:
        """Load safetensors and cast to the engine dtype."""
        from safetensors import safe_open

        handle = safe_open(str(path), framework="pt", device="cpu")
        names = list(handle.keys())
        self.checkpoint_dtype = str(handle.get_tensor(names[0]).dtype)

        tie = (
            self.cfg.tie_word_embeddings
            and "lm_head.weight" in names
            and "model.embed_tokens.weight" in names
        )
        if tie:
            if not torch.equal(
                handle.get_tensor("lm_head.weight"),
                handle.get_tensor("model.embed_tokens.weight"),
            ):
                raise ValueError(
                    "config sets tie_word_embeddings but lm_head.weight and "
                    "model.embed_tokens.weight differ in this checkpoint; refusing "
                    "to alias them"
                )
            names.remove("lm_head.weight")

        elements = 0
        inexact = 0
        max_abs = 0.0
        max_rel = 0.0
        tiny = torch.finfo(torch.float32).tiny

        weights: dict[str, torch.Tensor] = {}
        for name in names:
            tensor = handle.get_tensor(name)
            cast = tensor.to(ENGINE_DTYPE)
            elements += tensor.numel()

            flat_exact = tensor.reshape(-1)
            flat_cast = cast.reshape(-1)
            for start in range(0, flat_exact.numel(), CAST_REPORT_CHUNK):
                exact = flat_exact[start : start + CAST_REPORT_CHUNK].to(torch.float32)
                diff = (exact - flat_cast[start : start + CAST_REPORT_CHUNK].to(torch.float32)).abs()
                inexact += int((diff > 0).sum())
                max_abs = max(max_abs, float(diff.max()))
                max_rel = max(max_rel, float((diff / exact.abs().clamp_min(tiny)).max()))
                del exact, diff
            del tensor, flat_exact, flat_cast

            weights[name] = cast.to(self.device).contiguous()

        if self.cfg.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        self._cast_report = {
            "checkpoint_dtype": self.checkpoint_dtype,
            "engine_dtype": str(ENGINE_DTYPE),
            "distinct_elements": elements,
            "tied_lm_head": tie,
            "duplicate_elements_dropped": (
                weights["model.embed_tokens.weight"].numel() if tie else 0
            ),
            "elements_not_exactly_representable": inexact,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        }
        return weights

    def weight_cast_report(self) -> dict[str, object]:
        """How much the checkpoint -> fp16 cast cost, measured across all weights."""
        return dict(self._cast_report)

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]


    def forward(
        self, token_ids: torch.Tensor, start_pos: int, cache: KVCache
    ) -> torch.Tensor:
        """Run `token_ids` at positions [start_pos, start_pos + len), return logits."""
        cfg = self.cfg
        seq_len = token_ids.shape[0]
        kv_len = start_pos + seq_len
        assert kv_len <= cache.max_len, f"kv_len {kv_len} exceeds cache {cache.max_len}"

        h = self._w("model.embed_tokens.weight")[token_ids]
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

            q = rmsnorm(q, self._w(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")
            k = rmsnorm(k, self._w(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")

            q = apply_rope(q.contiguous(), cos, sin)
            k = apply_rope(k.contiguous(), cos, sin)

            write_kv(cache.pool, cache.uid, layer, start_pos, k, v)

            attn = attention(
                q,
                cache.pool.k[layer],
                cache.pool.v[layer],
                cache.block_table,
                q_start=start_pos,
                kv_len=kv_len,
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

    def forward_batch(
        self,
        pool: "paged.PagedKVCache",
        work: list[tuple[str, list[int], int]],
    ) -> dict[str, torch.Tensor]:
        """Run several sequences in one launch. Returns logits per uid."""
        cfg = self.cfg
        packed = torch.tensor(
            [t for _, tokens, _ in work for t in tokens], dtype=torch.long, device=self.device
        )
        starts = [start for _, _, start in work]
        lengths = [len(tokens) for _, tokens, _ in work]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        positions = torch.tensor(
            [start + i for (_, tokens, start) in work for i in range(len(tokens))],
            dtype=torch.long,
            device=self.device,
        )
        cos = self.cos[positions]
        sin = self.sin[positions]

        total = packed.shape[0]
        h = self._w("model.embed_tokens.weight")[packed]

        for layer in range(cfg.num_hidden_layers):
            p = f"model.layers.{layer}"

            normed = rmsnorm(
                h, self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps, "rmsnorm.hidden"
            )
            q = linear(normed, self._w(f"{p}.self_attn.q_proj.weight")).view(
                total, cfg.num_attention_heads, cfg.head_dim
            )
            k = linear(normed, self._w(f"{p}.self_attn.k_proj.weight")).view(
                total, cfg.num_key_value_heads, cfg.head_dim
            )
            v = linear(normed, self._w(f"{p}.self_attn.v_proj.weight")).view(
                total, cfg.num_key_value_heads, cfg.head_dim
            )

            q = rmsnorm(q, self._w(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")
            k = rmsnorm(k, self._w(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps, "rmsnorm.head")
            q = apply_rope(q.contiguous(), cos, sin)
            k = apply_rope(k.contiguous(), cos, sin)

            attn = torch.empty_like(q)
            for index, (uid, tokens, start) in enumerate(work):
                lo, hi = offsets[index], offsets[index + 1]
                kv_len = start + len(tokens)
                write_kv(pool, uid, layer, start, k[lo:hi], v[lo:hi])
                attn[lo:hi] = attention(
                    q[lo:hi],
                    pool.k[layer],
                    pool.v[layer],
                    pool.block_table(uid),
                    q_start=start,
                    kv_len=kv_len,
                    sm_scale=self.sm_scale,
                )

            h = h + linear(
                attn.reshape(total, cfg.num_attention_heads * cfg.head_dim),
                self._w(f"{p}.self_attn.o_proj.weight"),
            )
            normed = rmsnorm(
                h, self._w(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps,
                "rmsnorm.hidden",
            )
            gate = linear(normed, self._w(f"{p}.mlp.gate_proj.weight"))
            up = linear(normed, self._w(f"{p}.mlp.up_proj.weight"))
            h = h + linear(swiglu(gate, up), self._w(f"{p}.mlp.down_proj.weight"))

        h = rmsnorm(h, self._w("model.norm.weight"), cfg.rms_norm_eps, "rmsnorm.hidden")
        logits = linear(h, self._w("lm_head.weight"), config="gemm.lm_head")

        return {
            uid: logits[offsets[i] : offsets[i + 1]]
            for i, (uid, _, _) in enumerate(work)
        }


    def generate_greedy(
        self, prompt_ids: list[int], max_new_tokens: int, eos_token_ids: set[int] | None = None
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Greedy batch-1 decode. Returns generated ids and the logits per step."""
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
