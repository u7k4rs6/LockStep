"""The fp64 CPU reference forward pass."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from engine.model.qwen3 import ENGINE_DTYPE, Qwen3Config

LM_HEAD_CHUNK = 16384


def require_pinned_threads() -> dict[str, int]:
    """Refuse to run a reference whose reduction order is unpinned."""
    values = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    missing = [name for name, raw in values.items() if not (raw and raw.isdigit() and int(raw) >= 1)]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not pinned.\n"
            "MKL and OpenMP choose reduction trees by thread count, so the fp64\n"
            "reference would not be reproducible run to run. Export both before\n"
            "python starts, since MKL binds its pool at import:\n"
            "    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m bench.fidelity"
        )

    omp = int(values["OMP_NUM_THREADS"])
    mkl = int(values["MKL_NUM_THREADS"])
    torch.set_num_threads(omp)
    intraop = torch.get_num_threads()
    if intraop != omp:
        raise SystemExit(
            f"torch intraop pool is {intraop} but OMP_NUM_THREADS is {omp}; the "
            "reduction order would not match what env.lock records"
        )
    return {"omp_num_threads": omp, "mkl_num_threads": mkl, "torch_intraop_threads": intraop}


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Explicit max-subtract softmax over the last dimension, CPU only."""
    assert x.device.type == "cpu", (
        "the fp64 reference runs on CPU; torch.softmax fp64 on CUDA is wrong at "
        "some row widths, and moving ground truth to the GPU for speed would "
        "corrupt every fidelity number without failing anything"
    )
    m = x.max(dim=-1, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


class Fp64Reference:
    """A pure-CPU fp64 forward pass. Weights stay fp16; each op upcasts."""

    def __init__(self, weights_dir: Path):
        from safetensors import safe_open

        weights_dir = Path(weights_dir)
        self.cfg = Qwen3Config.from_file(weights_dir / "config.json")

        handle = safe_open(str(weights_dir / "model.safetensors"), framework="pt", device="cpu")
        names = list(handle.keys())

        if (
            self.cfg.tie_word_embeddings
            and "lm_head.weight" in names
            and "model.embed_tokens.weight" in names
        ):
            if not torch.equal(
                handle.get_tensor("lm_head.weight"),
                handle.get_tensor("model.embed_tokens.weight"),
            ):
                raise ValueError("tie_word_embeddings set but the two tensors differ")
            names.remove("lm_head.weight")

        self.w = {}
        for name in names:
            self.w[name] = handle.get_tensor(name).to(ENGINE_DTYPE)
        if self.cfg.tie_word_embeddings and "lm_head.weight" not in self.w:
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]
        self.sm_scale = 1.0 / math.sqrt(self.cfg.head_dim)

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ self.w[name].to(torch.float64).T

    def _rmsnorm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.cfg.rms_norm_eps) * self.w[name].to(
            torch.float64
        )

    def _rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.cfg.head_dim // 2
        exponent = torch.arange(0, half, dtype=torch.float64) * 2.0 / self.cfg.head_dim
        inv_freq = 1.0 / (self.cfg.rope_theta**exponent)
        freqs = torch.arange(seq_len, dtype=torch.float64)[:, None] * inv_freq[None, :]
        return freqs.cos(), freqs.sin()

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        lo, hi = x[..., :half], x[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1)

    def hidden(self, token_ids: list[int]) -> torch.Tensor:
        """Final normalized hidden states, [seq_len, hidden_size], fp64."""
        cfg = self.cfg
        seq_len = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long)

        h = self.w["model.embed_tokens.weight"][ids].to(torch.float64)
        cos, sin = self._rope_tables(seq_len)
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float64), diagonal=1
        )

        for layer in range(cfg.num_hidden_layers):
            p = f"model.layers.{layer}"

            normed = self._rmsnorm(h, f"{p}.input_layernorm.weight")
            q = self._linear(normed, f"{p}.self_attn.q_proj.weight").view(
                seq_len, cfg.num_attention_heads, cfg.head_dim
            )
            k = self._linear(normed, f"{p}.self_attn.k_proj.weight").view(
                seq_len, cfg.num_key_value_heads, cfg.head_dim
            )
            v = self._linear(normed, f"{p}.self_attn.v_proj.weight").view(
                seq_len, cfg.num_key_value_heads, cfg.head_dim
            )

            q = self._rmsnorm(q, f"{p}.self_attn.q_norm.weight")
            k = self._rmsnorm(k, f"{p}.self_attn.k_norm.weight")
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

            group = cfg.num_attention_heads // cfg.num_key_value_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

            scores = torch.einsum("qhd,khd->hqk", q, k) * self.sm_scale + causal
            attn = torch.einsum("hqk,khd->qhd", stable_softmax(scores), v)

            h = h + self._linear(
                attn.reshape(seq_len, cfg.num_attention_heads * cfg.head_dim),
                f"{p}.self_attn.o_proj.weight",
            )

            normed = self._rmsnorm(h, f"{p}.post_attention_layernorm.weight")
            gate = self._linear(normed, f"{p}.mlp.gate_proj.weight")
            up = self._linear(normed, f"{p}.mlp.up_proj.weight")
            h = h + self._linear(gate * torch.sigmoid(gate) * up, f"{p}.mlp.down_proj.weight")

        return self._rmsnorm(h, "model.norm.weight")

    def logits_for(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """Full-vocab fp64 logits for the given hidden rows."""
        weight = self.w["lm_head.weight"]
        out = torch.empty((hidden_rows.shape[0], weight.shape[0]), dtype=torch.float64)
        for start in range(0, weight.shape[0], LM_HEAD_CHUNK):
            stop = min(start + LM_HEAD_CHUNK, weight.shape[0])
            out[:, start:stop] = hidden_rows @ weight[start:stop].to(torch.float64).T
        return out
