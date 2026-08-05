"""What "identical" means through a black-box API, stated before anything runs.

The internal relations in `harness/mr/` compare raw fp16 logit bytes. A hosted or
local OpenAI-compatible endpoint does not expose those, so **the certifier's
relation is strictly weaker than the internal one**, and a reader who sees
"bitwise" applied to both would assume the same comparison was made. It was not.

The observable, precisely:

  1. **Token identity.** The emitted token id sequences must be equal, element
     for element and in length.
  2. **Logprob identity at the precision the API exposes.** For every emitted
     position, the returned logprob of the chosen token, and the logprobs of the
     top `TOP_LOGPROBS` alternatives with their token ids, must be equal as
     IEEE-754 doubles after JSON round-trip.

That second clause is the whole reason this is worth doing at all. Token identity
alone is a coarse observable: fp16 perturbations well below the top-1-to-top-2
gap change no token at all, so an engine can be visibly nondeterministic in its
logits and token-identical for thousands of positions. Logprobs are the finest
grain any OpenAI-compatible API offers, and asking for the top alternatives
rather than only the chosen token widens the exposed surface by
`TOP_LOGPROBS + 1` values per position.

What it still cannot see: anything below the serialized precision of the API's
float encoding, and any divergence in a position whose logprob happens to round
to the same value. So a negative result from this certifier means "no divergence
at this observable", not "bitwise identical". The report says exactly that.

**Greedy only.** Every certification run is temperature 0. Per-request seeding
differs between engines, and no OpenAI-compatible API exposes the counter-based
keying this project uses internally, so under sampling a divergence would tell
you about seeding conventions rather than about determinism. That is a scope
limit, not an oversight, and it is stated in the report rather than left
implicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# How many alternatives to request per position. 5 is the common cap across
# OpenAI-compatible servers; asking for more is silently truncated by some, which
# would make the observable differ between engines without saying so.
TOP_LOGPROBS = 5


@dataclass
class Completion:
    """One response, reduced to the observable."""

    tokens: list[int]
    logprobs: list[float]                      # chosen token, per position
    alternatives: list[list[tuple[int, float]]]  # (token id, logprob), per position

    def positions(self) -> int:
        return len(self.tokens)


@dataclass
class Comparison:
    identical: bool
    first_token_divergence: int | None = None
    first_logprob_divergence: int | None = None
    max_logprob_delta: float = 0.0
    detail: str = ""
    notes: list[str] = field(default_factory=list)


def compare(a: Completion, b: Completion) -> Comparison:
    """Apply the observable. Returns where and by how much, not just whether."""
    notes: list[str] = []

    if len(a.tokens) != len(b.tokens):
        return Comparison(
            False,
            first_token_divergence=min(len(a.tokens), len(b.tokens)),
            detail=f"length {len(a.tokens)} vs {len(b.tokens)}",
        )

    first_token = next(
        (i for i, (x, y) in enumerate(zip(a.tokens, b.tokens)) if x != y), None
    )

    first_logprob = None
    worst = 0.0
    for index, (x, y) in enumerate(zip(a.logprobs, b.logprobs)):
        if x is None or y is None:
            notes.append(f"position {index}: an engine returned no logprob")
            continue
        if not _same(x, y):
            worst = max(worst, abs(x - y))
            if first_logprob is None:
                first_logprob = index

    for index, (xs, ys) in enumerate(zip(a.alternatives, b.alternatives)):
        if len(xs) != len(ys):
            notes.append(
                f"position {index}: {len(xs)} alternatives vs {len(ys)}; the "
                "engines expose different widths, so the observable is narrower "
                "than requested"
            )
            continue
        for (xt, xl), (yt, yl) in zip(xs, ys):
            if xt != yt or not _same(xl, yl):
                worst = max(worst, abs(xl - yl))
                if first_logprob is None:
                    first_logprob = index

    identical = first_token is None and first_logprob is None
    return Comparison(
        identical=identical,
        first_token_divergence=first_token,
        first_logprob_divergence=first_logprob,
        max_logprob_delta=worst,
        detail="" if identical else (
            f"first token divergence at {first_token}"
            if first_token is not None
            else f"logprobs differ from position {first_logprob}, max delta {worst:.3e}"
        ),
        notes=notes,
    )


def _same(x: float, y: float) -> bool:
    """Exact equality, with NaN handled. No tolerance.

    A tolerance here would be a decision about how much nondeterminism is
    acceptable, which is exactly the question being asked. If two runs of a
    deterministic mode differ at all at the exposed precision, that is the
    finding.
    """
    if x is None or y is None:
        return x is y
    if math.isnan(x) and math.isnan(y):
        return True
    return x == y


def describe() -> str:
    """Printed at the top of every certification report."""
    return (
        "observable: token ids identical, and the chosen-token logprob plus the "
        f"top {TOP_LOGPROBS} alternatives with their ids identical as doubles "
        "after JSON round-trip, at every emitted position. Greedy only "
        "(temperature 0). This is strictly weaker than the internal relations, "
        "which compare raw fp16 logit bytes; a negative result means no "
        "divergence at this observable, not bitwise identity."
    )
