"""Build the small stable fp64 cache: top-2 logits per position."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.corpus import PROMPTS, corpus_sha256  # noqa: E402
from bench.fp64_reference import Fp64Reference, require_pinned_threads  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"
DEFAULT_OUT = REPO_ROOT / "reference"

ROW_CHUNK = 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    threads = require_pinned_threads()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.weights))
    model = Fp64Reference(args.weights)

    print(
        f"fp64 top-2 cache  corpus sha256:{corpus_sha256()[:16]}  "
        f"threads omp={threads['omp_num_threads']} mkl={threads['mkl_num_threads']}"
    )

    prompt_ids: list[list[int]] = []
    top_values: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    started = time.monotonic()

    for prompt in PROMPTS:
        ids = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"{prompt.uid} tokenized to nothing")
        hidden = model.hidden(ids)

        for start in range(0, len(ids), ROW_CHUNK):
            rows = model.logits_for(hidden[start : start + ROW_CHUNK])
            values, indices = torch.topk(rows, 2, dim=-1, sorted=True)
            top_values.append(values)
            top_indices.append(indices.to(torch.int32))

            log_probs = rows - torch.logsumexp(rows, dim=-1, keepdim=True)
            entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1))
            del rows, log_probs

        prompt_ids.append(ids)
        del hidden
        print(f"  {prompt.uid}  {len(ids):>4} positions  {time.monotonic() - started:7.1f}s",
              flush=True)

    entropy = torch.cat(entropies)
    cache = {
        "corpus_sha256": corpus_sha256(),
        "prompt_uids": [p.uid for p in PROMPTS],
        "prompt_token_ids": prompt_ids,
        "top2_logits": torch.cat(top_values),
        "top2_indices": torch.cat(top_indices),
        "entropy": entropy,
        **threads,
    }
    torch.save(cache, args.out / "top2.pt")

    size_kb = (args.out / "top2.pt").stat().st_size / 1e3
    print(f"\ntop2.pt  {int(entropy.numel())} positions  {size_kb:.1f} KB")

    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "corpus_sha256": corpus_sha256(),
                "total_positions": int(entropy.numel()),
                **threads,
                "entropy_nats": {
                    "min": float(entropy.min()),
                    "median": float(entropy.median()),
                    "max": float(entropy.max()),
                },
                "build_seconds": round(time.monotonic() - started, 1),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"built in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
