"""Fidelity as a mutation observer.

Invariance and correctness are different claims, and this file is where that
stops being a slogan. A mutation can leave I1 through I4 perfectly green and
still make the engine wrong, because those relations compare the engine against
*itself* under different schedules. Reversing the split-combine fold is the clean
example: the split count depends only on the request's own KV length, so
canonical and batched execution are perturbed identically and every invariance
relation agrees. Only a comparison against fp64 can see it.

That was the prediction. It was tested and it is wrong, which is worth more than
the score point it cost.

Measured on a 600-token prompt against fp64: the clean engine's max absolute
logit error is 6.670e-02 and the reversed-fold engine's is 8.158e-02. Both sit
inside the F1.5 bound of 0.5. So the reversed fold is invisible to F1 as well,
and for a reason that generalizes: reversing a reduction order perturbs results
by roughly one unit in the last place, the F1 bound carries about sevenfold
headroom over the clean engine's own error, and every invariance relation
compares the engine against itself under a perturbation that is uniform across
schedules.

Neither claim family can see it. That is a genuine harness gap, not a proven
equivalence, and naming the missing observer is the useful part: what would catch
it is a **bitwise regression baseline**, committed golden logit bytes for a fixed
corpus compared exactly rather than within a tolerance. That is a third observer,
different from both invariance and fidelity, and it is not built.

The check is a spot-check rather than the full 2756-position pass: a handful of
positions is enough to exceed the F1 bounds by orders of magnitude when a
reduction order has changed, and the full pass takes two minutes per trial.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench.fp64_reference import Fp64Reference  # noqa: E402
from engine.model.qwen3 import KVCache  # noqa: E402

# F1.5 from the architecture doc. A reduction-order change moves logits by far
# more than fp16 rounding, so this bound separates the two cleanly.
MAX_ABS_LOGIT_ERROR = 0.5


def spot_check(model, reference: Fp64Reference, prompt: list[int]) -> tuple[bool, float]:
    """Max absolute logit error against fp64 over one prompt.

    Returns (within_bounds, max_error). The prompt must be long enough to reach
    a second attention split, or a fault in the split-combine fold has nothing to
    perturb and the check reports clean for the wrong reason.
    """
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
