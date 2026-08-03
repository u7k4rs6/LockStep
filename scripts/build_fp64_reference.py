"""Build the fp64 CPU reference for F1.

docs/02-technical-architecture.md section 7: an fp64 CPU forward pass over a few
thousand positions, cached on disk, used as the fidelity reference and never as
an invariance oracle.

Three decisions worth stating, because each one is a place a fidelity number can
be quietly inflated:

1. **The reference uses the fp16 weight values the engine holds, upcast to
   fp64.** Not the bf16 checkpoint. This isolates forward-pass numerics, which is
   what F1 claims to measure, from the weight cast, which is reported separately
   by `Qwen3.weight_cast_report`. Mixing them would let a clean cast flatter a
   noisy forward pass or the reverse.

2. **The reference computes the mathematically correct attention**, a single
   fp64 softmax over the whole causal row. It does not mirror the engine's split
   structure. The engine's splitting is a numerics artifact to be measured, not a
   definition to be matched.

3. **`OMP_NUM_THREADS` is pinned and asserted.** MKL and OpenMP pick reduction
   trees by thread count, so an unpinned reference is itself nondeterministic. A
   nondeterministic ground truth in a determinism project is the first thing a
   reviewer would go looking for. The value is recorded in env.lock.

Two artifacts:

  * `compressed.pt` over every prefill position: top-2048 fp64 logits with their
    ids, plus full-vocab logsumexp, max, and the tail's logsumexp.
  * `exact.pt` under `--exact`: full-vocab fp64 logits for a stratified sample of
    positions, so bench/fidelity.py can report what the compression cost.

k is 2048 rather than 256 because the cost was measured rather than assumed. At
k=256 the compressed KL was missing 27.4 percent of the true full-vocabulary KL
on this corpus, and the measured sweep (256 -> 8192) showed the gap closing only
about 1.3x per doubling: 21.7 percent at 512, 12.3 at 2048, 5.2 at 8192.
Reaching one percent would need k in the tens of thousands, which is not
compression. So top-k KL is a lower bound at every practical k, and
bench/fidelity.py reports it as one, with exact KL over the audit subset as the
headline. k=2048 is the point where the standing artifact is still small
(68 MB) and the bound is reasonably tight.

Regeneration is single-digit minutes, so these are a convenience, not a
necessity, and they are gitignored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.corpus import PROMPTS, corpus_sha256  # noqa: E402
from engine.model.qwen3 import ENGINE_DTYPE, Qwen3Config  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
DEFAULT_OUT = REPO_ROOT / "reference"

TOP_K = 2048
LM_HEAD_CHUNK = 16384  # vocab rows upcast at a time; 16384*1024*8 B = 134 MB


def require_pinned_threads() -> int:
    """Refuse to build a reference whose reduction order is unpinned."""
    value = os.environ.get("OMP_NUM_THREADS")
    if not value or not value.isdigit() or int(value) < 1:
        raise SystemExit(
            "OMP_NUM_THREADS is not pinned.\n"
            "MKL and OpenMP choose reduction trees by thread count, so the fp64\n"
            "reference would not be reproducible run to run. Set it explicitly:\n"
            "    OMP_NUM_THREADS=8 python3 scripts/build_fp64_reference.py"
        )
    threads = int(value)
    torch.set_num_threads(threads)
    return threads


def _stable_softmax(x: torch.Tensor) -> torch.Tensor:
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

        self.cfg = Qwen3Config.from_file(weights_dir / "config.json")
        raw = load_file(str(weights_dir / "model.safetensors"), device="cpu")
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

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """Chunked over the vocabulary so the fp64 upcast never needs 1.24 GB."""
        weight = self.w["lm_head.weight"]
        out = torch.empty((h.shape[0], weight.shape[0]), dtype=torch.float64)
        for start in range(0, weight.shape[0], LM_HEAD_CHUNK):
            stop = min(start + LM_HEAD_CHUNK, weight.shape[0])
            out[:, start:stop] = h @ weight[start:stop].to(torch.float64).T
        return out

    def prefill(self, token_ids: list[int]) -> torch.Tensor:
        """Full-vocab fp64 logits for every position in the prompt."""
        cfg = self.cfg
        seq_len = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long)

        h = self.w["model.embed_tokens.weight"][ids].to(torch.float64)
        cos, sin = self._rope_tables(seq_len)

        causal = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float64)
        causal = torch.triu(causal, diagonal=1)

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

            # One exact fp64 softmax per causal row. Deliberately not the
            # engine's split-and-combine structure.
            scores = torch.einsum("qhd,khd->hqk", q, k) * self.sm_scale + causal
            attn = torch.einsum("hqk,khd->qhd", _stable_softmax(scores), v)

            h = h + self._linear(
                attn.reshape(seq_len, cfg.num_attention_heads * cfg.head_dim),
                f"{p}.self_attn.o_proj.weight",
            )

            normed = self._rmsnorm(h, f"{p}.post_attention_layernorm.weight")
            gate = self._linear(normed, f"{p}.mlp.gate_proj.weight")
            up = self._linear(normed, f"{p}.mlp.up_proj.weight")
            act = gate * torch.sigmoid(gate) * up
            h = h + self._linear(act, f"{p}.mlp.down_proj.weight")

        h = self._rmsnorm(h, "model.norm.weight")
        return self._lm_head(h)


def compress(logits: torch.Tensor, top_k: int = TOP_K) -> dict[str, torch.Tensor]:
    """Top-k plus the three full-vocab scalars that make the tail recoverable.

    `tail_logsumexp` is over the vocabulary minus the top-k, computed from the
    full row before it is discarded. With it, the exact normalizer is
    logaddexp(head_logsumexp, tail_logsumexp), so the compressed reference gives
    exact probabilities for every token it retains, and an exact total mass for
    the ones it does not. The only thing lost is the per-token split of the tail.
    """
    values, indices = torch.topk(logits, top_k, dim=-1, sorted=True)
    full_lse = torch.logsumexp(logits, dim=-1)

    masked = logits.clone()
    masked.scatter_(-1, indices, float("-inf"))
    tail_lse = torch.logsumexp(masked, dim=-1)

    return {
        "top_indices": indices.to(torch.int32),
        "top_logits": values,
        "full_logsumexp": full_lse,
        "full_max": logits.max(dim=-1).values,
        "tail_logsumexp": tail_lse,
    }


def entropy_from_full(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def stratified_sample(entropy: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Pick `count` positions spread across the entropy range, not uniformly.

    Ten equal-width bins over the observed entropy range, sampled round-robin so
    that flat positions, which are rare and are where the interesting behavior
    is, are represented in proportion to the number of bins rather than to their
    frequency.
    """
    n = entropy.numel()
    if count >= n:
        return torch.arange(n)

    generator = torch.Generator().manual_seed(seed)
    lo, hi = float(entropy.min()), float(entropy.max())
    width = (hi - lo) or 1.0
    bin_index = ((entropy - lo) / width * 10).clamp(0, 9).to(torch.long)

    buckets: list[list[int]] = []
    for b in range(10):
        members = torch.nonzero(bin_index == b, as_tuple=False).flatten()
        shuffled = members[torch.randperm(members.numel(), generator=generator)]
        buckets.append(shuffled.tolist())

    chosen: list[int] = []
    while len(chosen) < count and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < count:
                chosen.append(bucket.pop())
    return torch.tensor(sorted(chosen), dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Also write full-vocab fp64 logits for an audit subset.",
    )
    parser.add_argument("--exact-positions", type=int, default=400)
    parser.add_argument("--exact-seed", type=int, default=20260803)
    args = parser.parse_args()

    threads = require_pinned_threads()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.weights))
    model = Fp64Reference(args.weights)

    print(f"fp64 reference  corpus sha256:{corpus_sha256()[:16]}  threads={threads}")

    prompt_ids: list[list[int]] = []
    per_prompt: list[dict[str, torch.Tensor]] = []
    entropies: list[torch.Tensor] = []
    started = time.monotonic()

    # Pass one compresses and discards. Holding every prompt's full-vocab fp64
    # logits to slice an audit subset out of later would need 2756 * 151936 * 8
    # bytes, which is 3.35 GB and more than this machine has spare; the audit
    # subset is recomputed in pass two instead. Compute is the cheap resource
    # here, memory is not.
    for prompt in PROMPTS:
        ids = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"{prompt.uid} tokenized to nothing")
        logits = model.prefill(ids)

        prompt_ids.append(ids)
        per_prompt.append(compress(logits))
        entropies.append(entropy_from_full(logits))
        del logits

        elapsed = time.monotonic() - started
        print(f"  {prompt.uid}  {len(ids):>4} positions  {elapsed:7.1f}s", flush=True)

    entropy = torch.cat(entropies)
    total_positions = int(entropy.numel())

    compressed = {
        "corpus_sha256": corpus_sha256(),
        "top_k": TOP_K,
        "omp_num_threads": threads,
        "prompt_uids": [p.uid for p in PROMPTS],
        "prompt_token_ids": prompt_ids,
        "prompt_lengths": [len(ids) for ids in prompt_ids],
        "entropy": entropy,
        **{
            key: torch.cat([block[key] for block in per_prompt])
            for key in per_prompt[0]
        },
    }
    torch.save(compressed, args.out / "compressed.pt")
    size_mb = (args.out / "compressed.pt").stat().st_size / 1e6
    print(f"\ncompressed.pt  {total_positions} positions  top-{TOP_K}  {size_mb:.1f} MB")

    if args.exact:
        selected = stratified_sample(entropy, args.exact_positions, args.exact_seed)

        # Pass two: re-prefill only the prompts holding a selected position and
        # keep only those rows.
        wanted = set(selected.tolist())
        offsets = [0]
        for ids in prompt_ids:
            offsets.append(offsets[-1] + len(ids))

        order = {int(g): i for i, g in enumerate(selected.tolist())}
        full = torch.empty((selected.numel(), model.cfg.vocab_size), dtype=torch.float64)
        for index, (prompt, ids) in enumerate(zip(PROMPTS, prompt_ids)):
            start, stop = offsets[index], offsets[index + 1]
            local = sorted(g - start for g in wanted if start <= g < stop)
            if not local:
                continue
            logits = model.prefill(ids)
            for position in local:
                full[order[start + position]] = logits[position]
            del logits
            print(
                f"  exact  {prompt.uid}  {len(local):>3} of {len(ids)} positions  "
                f"{time.monotonic() - started:7.1f}s",
                flush=True,
            )

        torch.save(
            {
                "corpus_sha256": corpus_sha256(),
                "omp_num_threads": threads,
                "positions": selected,
                "logits": full,
                "seed": args.exact_seed,
            },
            args.out / "exact.pt",
        )
        size_mb = (args.out / "exact.pt").stat().st_size / 1e6
        print(
            f"exact.pt       {selected.numel()} positions  full vocab  {size_mb:.1f} MB"
            f"  (stratified over 10 entropy bins, seed {args.exact_seed})"
        )

    meta = {
        "corpus_sha256": corpus_sha256(),
        "total_positions": total_positions,
        "top_k": TOP_K,
        "omp_num_threads": threads,
        "entropy_nats": {
            "min": float(entropy.min()),
            "median": float(entropy.median()),
            "max": float(entropy.max()),
        },
        "build_seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built in {meta['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
