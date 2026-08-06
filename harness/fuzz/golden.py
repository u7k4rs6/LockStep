"""The third observer: exact comparison against a committed baseline.

The other two cannot see everything, and the gap between them is precise:

  * **I1 to I4** compare the engine against *itself* across schedules. A
    perturbation that is uniform across schedules is invisible by construction,
    because both sides of every comparison move together.
  * **F1** compares against fp64 within a tolerance. Against the reversed fold
    its statistic, the maximum absolute logit error, is identical to every digit
    (6.669992e-02 either way), so no threshold on it separates the two. The
    logits themselves do differ: measured per position, exactly 0 across the 512
    single-split positions and up to 3.906250e-02 across the multi-split ones,
    about 2.5 fp16 ulp, which is below the quantization error F1 already absorbs.
    That is F1 working as designed, not F1 miscalibrated.
  * **Golden bytes**, here, compare against a committed baseline exactly. This is
    the observer that distinguishes "the engine changed" from "the engine is
    inconsistent" and from "the engine is inaccurate".

The reversed split-combine mutant survived the first two and is what this file
was built for. It is not a redundant check: the three answer different questions.
"Does the engine agree with itself across schedules" and "is the engine within t
of fp64" both have correct negative answers here. Only "is this the same engine
that produced the published numbers" has an exact one.

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

from engine.kv import paged  # noqa: E402

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
    # Through `forward_batch`, the path the scheduler actually serves from, not
    # through `forward`. The baseline previously observed the batch-1 contiguous
    # path while every served request went through the packed one, so a defect
    # confined to the served implementation would have left the committed digests
    # untouched. `harness/mr/equivalence.py::path_equivalence` asserts the two
    # agree bitwise; this makes the baseline independent of that assertion
    # holding rather than resting on it.
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
