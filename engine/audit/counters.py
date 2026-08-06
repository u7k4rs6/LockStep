"""Execution counters over every path under claim.

Three times now a path has looked tested and never executed: a corpus that never
reached a second attention split, a chunk-size row that never chunked, and an
admission check that made the eviction path unreachable. Each was caught by
noticing afterwards. This makes it a mechanism.

Every relation asserts not only its outcome but that the paths it exists to
exercise had nonzero counts. A relation that passes without firing its own
mechanism is a failure, not a pass.

The counters are also the fuzzer's coverage substrate, so the coverage report and
the vacuity guards read the same numbers. A coverage denominator derived from
somewhere else than the thing being counted is how a coverage number ends up
describing a denominator someone chose.

Counters are observation only. They are incremented after a decision is taken,
never read by one, and nothing in a kernel config path can see them: a counter
that fed back into a decision would be a batch-derived quantity by another name.
"""

from __future__ import annotations

from collections import Counter as _Counter
from dataclasses import dataclass, field

# Every path under claim, with the one-line statement of what firing it means.
# The fuzzer's coverage report enumerates this dict rather than a hand-written
# list, so a path added here is covered by both without a second edit.
PATHS: dict[str, str] = {
    # admission and lifecycle
    "admit": "a waiting request was admitted",
    "admit_refused": "the policy declined to admit a waiting request",
    "finish_eos":
        "a request finished because it emitted an EOS token, the half of the "
        "finish disjunction that no relation or campaign ever exercised: every "
        "caller left eos_token_ids empty, so `generated[-1] in self.eos` had "
        "never once been true",
    "finish_limit":
        "a request finished by reaching max_new_tokens",
    "finish": "a request reached its limit or an EOS and released its blocks",
    # prefill and chunking
    "prefill_chunk": "a partial prefill chunk was computed",
    "prefill_complete": "a prefill finished and the request began decoding",
    "decode_step": "a single-token decode step was computed",
    "chunk_boundary_mid_block": "a chunk ended part-way through a KV block",
    "chunk_boundary_mid_split": "a chunk ended part-way through an attention split",
    "chunk_boundary_on_block": "a chunk ended exactly on a block boundary",
    # attention shape
    "attention_multi_split": "attention ran with two or more splits",
    "attention_single_split": "attention ran within one split",
    # preemption
    "preempt_fired": "a running request was preempted for recompute",
    "preempt_depth_2": "a request was preempted for the second time",
    "preempt_depth_3_plus": "a request was preempted a third time or beyond",
    "resume": "a preempted request was re-admitted and recomputed",
    # prefix cache
    "cache_hit": "a prefix hit was found and honoured",
    "cache_hit_refused": "a prefix hit was found and the policy declined it",
    "cache_miss": "a lookup found no usable prefix",
    "cache_insert": "whole blocks were indexed after a request finished",
    "cache_full_prompt_trimmed": "a whole-prompt hit gave its last block back",
    # allocator
    "eviction_taken": "a cached block was reclaimed under pressure",
    "eviction_pass": "one pass of the reserve-with-eviction loop",
    "block_reclaimed_at_zero": "a block's last reference went away and it was freed",
    "out_of_blocks": "the pool could not satisfy a reservation",
}


@dataclass
class Counters:
    """Per-run execution counts. Created by the Scheduler, shared with the pool.

    Not a module global: a global would make one test's counts visible to the
    next and would not survive the cross-process replay, where two interpreters
    must produce identical trajectories.
    """

    counts: _Counter = field(default_factory=_Counter)

    def hit(self, path: str, amount: int = 1) -> None:
        if path not in PATHS:
            raise KeyError(
                f"{path!r} is not a declared path. Add it to PATHS with a "
                "one-line statement of what firing it means, so the coverage "
                "report enumerates it too."
            )
        self.counts[path] += amount

    def __getitem__(self, path: str) -> int:
        return self.counts.get(path, 0)

    def fired(self, *paths: str) -> bool:
        return all(self.counts.get(path, 0) > 0 for path in paths)

    def missing(self, *paths: str) -> list[str]:
        """Which of `paths` never fired. What a vacuity guard reports."""
        return [path for path in paths if self.counts.get(path, 0) == 0]

    def merge(self, other: "Counters") -> None:
        self.counts.update(other.counts)

    def as_dict(self) -> dict[str, int]:
        return {path: self.counts.get(path, 0) for path in PATHS if self.counts.get(path, 0)}

    def summary(self) -> str:
        live = self.as_dict()
        if not live:
            return "no paths fired"
        width = max(len(path) for path in live)
        return "\n".join(f"  {path:<{width}}  {count}" for path, count in sorted(live.items()))


class VacuousRun(AssertionError):
    """A relation passed without firing the paths it exists to exercise."""


def require_fired(counters: Counters, *paths: str, what: str = "this relation") -> None:
    """Assert the mechanism actually ran. The generalization of the hand checks.

    MR3 reported "7 of 10 actually preempted" and MR4 reported "12 of 12
    registering real hits" by hand. This is that, for every relation.
    """
    missing = counters.missing(*paths)
    if missing:
        raise VacuousRun(
            f"{what} passed without firing: {', '.join(missing)}. "
            "A relation that never executes its own mechanism is not evidence. "
            f"Paths that did fire: {sorted(counters.as_dict())}"
        )
