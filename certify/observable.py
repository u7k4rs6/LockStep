"""What "identical" means through a black-box API, stated before anything runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TOP_LOGPROBS = 5


@dataclass
class Completion:
    """One response, reduced to the observable."""

    tokens: list[int]
    logprobs: list[float]
    alternatives: list[list[tuple[int, float]]]

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
    """Exact equality, with NaN handled. No tolerance."""
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
