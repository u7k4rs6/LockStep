"""The fp64 CPU reference forward pass.

docs/02-technical-architecture.md section 7: the fidelity reference, and never an
invariance oracle.

Three decisions worth stating, because each is a place a fidelity number can be
quietly inflated:

1. **The reference uses the fp16 weight values the engine holds, upcast to
   fp64.** Not the bf16 checkpoint. This isolates forward-pass numerics, which is
   what F1 claims to measure, from the weight cast, which `Qwen3` reports
   separately. Mixing them would let a clean cast flatter a noisy forward pass.

2. **The reference computes the mathematically correct attention**, a single fp64
   softmax over the whole causal row. It does not mirror the engine's split
   structure. The engine's splitting is a numerics artifact to be measured, not a
   definition to be matched.

3. **Thread counts are pinned and asserted.** MKL and OpenMP pick reduction trees
   by thread count, so an unpinned reference is itself nondeterministic. All
   three of OMP_NUM_THREADS, MKL_NUM_THREADS, and torch's intraop pool are
   checked, because the first two are backend-specific environment variables
   while `torch.get_num_threads()` is what actually governs the pool whichever
   backend is underneath.

The class exposes `hidden` and `logits_for` separately rather than one `prefill`
returning full logits. A 727-position prompt at fp64 full vocab is 884 MB, and
the caller needs the engine's rows alongside it; splitting lets the caller walk
the vocabulary in row chunks and keep the peak near 80 MB.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from engine.model.qwen3 import ENGINE_DTYPE, Qwen3Config

LM_HEAD_CHUNK = 16384  # vocab rows upcast at a time; 16384*1024*8 B = 134 MB


def require_pinned_threads() -> dict[str, int]:
    """Refuse to run a reference whose reduction order is unpinned.

    MKL_NUM_THREADS has to be exported before torch is imported to bind, so this
    checks the environment rather than trusting a late assignment, and then
    cross-checks torch's actual intraop pool against it.
    """
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
    """Explicit max-subtract softmax over the last dimension, CPU only.

    Not `torch.softmax`, and the device assertion is not paranoia: torch.softmax
    on CUDA in fp64 returns probabilities off by as much as 0.5 at row widths 513
    and 769 with two or more rows (tests/test_torch_softmax_hazard.py pins the
    repro). Ground truth silently computed on a broken path is the worst failure
    this project could have, so the reference refuses to leave the CPU.
    """
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
        from safetensors.torch import load_file

        weights_dir = Path(weights_dir)
        self.cfg = Qwen3Config.from_file(weights_dir / "config.json")
        raw = load_file(str(weights_dir / "model.safetensors"), device="cpu")

        # Same tied-embedding handling as the engine: the checkpoint ships a
        # bitwise-identical duplicate of the embedding as lm_head, and holding
        # both costs 311 MB for nothing.
        if (
            self.cfg.tie_word_embeddings
            and "lm_head.weight" in raw
            and "model.embed_tokens.weight" in raw
        ):
            if not torch.equal(raw["lm_head.weight"], raw["model.embed_tokens.weight"]):
                raise ValueError("tie_word_embeddings set but the two tensors differ")
            del raw["lm_head.weight"]

        self.w = {name: t.to(ENGINE_DTYPE) for name, t in raw.items()}
        del raw
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
        """Full-vocab fp64 logits for the given hidden rows.

        Chunked over the vocabulary so the fp64 upcast of the 151936 x 1024
        lm_head never needs 1.24 GB resident.
        """
        weight = self.w["lm_head.weight"]
        out = torch.empty((hidden_rows.shape[0], weight.shape[0]), dtype=torch.float64)
        for start in range(0, weight.shape[0], LM_HEAD_CHUNK):
            stop = min(start + LM_HEAD_CHUNK, weight.shape[0])
            out[:, start:stop] = hidden_rows @ weight[start:stop].to(torch.float64).T
        return out
