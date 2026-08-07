"""Fidelity as a mutation observer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.fp64_reference import Fp64Reference  # noqa: E402
from engine.model.qwen3 import KVCache  # noqa: E402

MAX_ABS_LOGIT_ERROR = 0.5


def spot_check(model, reference: Fp64Reference, prompt: list[int]) -> tuple[bool, float]:
    """Max absolute logit error against fp64 over one prompt."""
    cache = KVCache(model.cfg, len(prompt), model.device)
    ids = torch.tensor(prompt, dtype=torch.long, device=model.device)
    engine_logits = model.forward(ids, 0, cache).to(torch.float64).cpu()

    hidden = reference.hidden(prompt)
    worst = 0.0
    for start in range(0, len(prompt), 64):
        rows = reference.logits_for(hidden[start : start + 64])
        worst = max(worst, float((engine_logits[start : start + 64] - rows).abs().max()))
        del rows

    del engine_logits, hidden
    torch.cuda.empty_cache()
    return worst <= MAX_ABS_LOGIT_ERROR, worst


def build_prompt(vocab: int, tokens: int = 600, seed: int = 5150) -> list[int]:
    """Longer than the 512 split size, so the split-combine fold actually runs."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (tokens,), generator=generator).tolist()
