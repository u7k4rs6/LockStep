"""The third observer: exact comparison against a committed baseline."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine.kv import paged  # noqa: E402

BASELINE = Path(__file__).parent / "golden.json"

# Two prompts cross the 512 split boundary and two do not, so a fold fault has
# something to perturb and the single-split path is covered. Under the reversed
# fold the 600 and 520 digests move and the 48 and 17 do not, which is the
# observer seeing the mechanism rather than noise.
CORPUS_LENGTHS = (600, 520, 48, 17)
CORPUS_SEED = 90210


def corpus(vocab: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(CORPUS_SEED)
    return [
        torch.randint(0, vocab, (n,), generator=generator).tolist()
        for n in CORPUS_LENGTHS
    ]


def logit_digest(model, prompt: list[int]) -> str:
    """sha256 over the raw fp16 logit bytes for every position in the prompt."""
    # Through forward_batch, the path the scheduler serves from. The baseline
    # previously observed forward, the batch-1 path, so a defect confined to the
    # served implementation would have left the committed digests untouched.
    pool = paged.PagedKVCache(
        num_blocks=-(-len(prompt) // paged.DEFAULT_BLOCK_SIZE) + 2,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        device=model.device,
        dtype=torch.float16,
        block_size=paged.DEFAULT_BLOCK_SIZE,
    )
    uid = "golden"
    pool.create(uid)
    pool.reserve(uid, len(prompt))
    logits = model.forward_batch(pool, [(uid, list(prompt), 0)])[uid]
    raw = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    del logits, pool
    torch.cuda.empty_cache()
    return hashlib.sha256(raw).hexdigest()


def measure(model) -> dict[str, str]:
    return {
        f"prompt_{len(p)}": logit_digest(model, p) for p in corpus(model.cfg.vocab_size)
    }


def write_baseline(model, env_fingerprint: str) -> Path:
    BASELINE.write_text(json.dumps({
        "note": "Committed baseline for the golden-bytes observer. Regenerating "
                "this is a claims-affecting change: it asserts that a numerics "
                "difference is intended.",
        "env_fingerprint": env_fingerprint,
        "corpus_lengths": list(CORPUS_LENGTHS),
        "corpus_seed": CORPUS_SEED,
        "digests": measure(model),
    }, indent=2) + "\n")
    return BASELINE


def compare(model) -> tuple[bool, list[str]]:
    """(matches, differing prompt names). Exact, with no tolerance."""
    if not BASELINE.is_file():
        return True, []
    baseline = json.loads(BASELINE.read_text())["digests"]
    current = measure(model)
    differing = [
        name for name, digest in current.items()
        if name in baseline and baseline[name] != digest
    ]
    return not differing, differing
