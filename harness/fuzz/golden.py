"""The third observer: exact comparison against a committed baseline.

The other two cannot see everything, and the gap between them is precise:

  * **I1 to I4** compare the engine against *itself* across schedules. A
    perturbation that is uniform across schedules is invisible by construction,
    because both sides of every comparison move together.
  * **F1** compares against fp64 within a tolerance. The bound carries about
    sevenfold headroom over the clean engine's own error, which is enough to
    absorb the roughly one unit in the last place that a changed reduction order
    moves results by.
  * **Golden bytes**, here, compare against a committed baseline exactly. This is
    the observer that distinguishes "the engine changed" from "the engine is
    inconsistent" and from "the engine is inaccurate".

The reversed split-combine mutant survived the first two and is what this file
was built for. It is not a redundant check: the three answer different questions,
and only this one answers "is this the same engine that produced the published
numbers".

Measured against that mutant, the corpus behaves the way the corpus was designed
to: the 600 and 520 token digests change and the 48 and 17 token digests do not.
The boundary between changed and unchanged falls exactly on the 512-token split
threshold, which is evidence the observer is seeing the mechanism rather than
seeing noise. A corpus where all four changed, or none did, would be a weaker
result even if the verdict were the same.

The baseline is committed (`harness/fuzz/golden.json`, a few KB of hashes) rather
than regenerated, because a baseline the run recomputes is not a baseline. A
legitimate numerics change requires regenerating it deliberately, which is a
claims-affecting act and shows up in review as one.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine.model.qwen3 import KVCache  # noqa: E402

BASELINE = Path(__file__).parent / "golden.json"

# Short and fixed. Two prompts long enough to cross the 512 split boundary, so a
# fault in the split-combine fold has something to perturb, and two short ones so
# the single-split path is covered too.
CORPUS_LENGTHS = (600, 520, 48, 17)
CORPUS_SEED = 90210


def corpus(vocab: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(CORPUS_SEED)
    return [
        torch.randint(0, vocab, (n,), generator=generator).tolist()
        for n in CORPUS_LENGTHS
    ]


def logit_digest(model, prompt: list[int]) -> str:
    """sha256 over the raw fp16 logit bytes for every position in the prompt.

    Bytes, not a rounded decimal view: the architecture doc defines
    bit-identical as identical raw fp16 logit bytes, so that is what is hashed.
    """
    cache = KVCache(model.cfg, len(prompt), model.device)
    ids = torch.tensor(prompt, dtype=torch.long, device=model.device)
    logits = model.forward(ids, 0, cache)
    raw = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    del logits
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
