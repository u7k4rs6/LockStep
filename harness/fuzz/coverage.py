"""Coverage: lifecycle n-grams, boundary predicates, preemption depth."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.sched.lifecycle import Event, derivation_note, feasible_ngrams


class UndeclaredTransition(AssertionError):
    """A run produced an n-gram the state machine says is impossible."""


BOUNDARY_PREDICATES = {
    "chunk end vs block size": ("below", "exact", "above"),
    "cache hit length vs block size": ("below", "exact", "above"),
    "split boundary vs kv length": ("below", "exact", "above"),
    "free block count": ("zero", "one", "low"),
    "batch size transition": ("1", "2", "3", "4", "8", "16", "31", "32"),
}


@dataclass
class Coverage:
    """Accumulates coverage across a campaign."""

    ngram_n: tuple[int, ...] = (2, 3)
    observed_ngrams: dict[int, set] = field(default_factory=dict)
    boundary_hits: dict[str, set] = field(default_factory=dict)
    preempt_depths: dict[int, int] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    witnessed_transitions: set = field(default_factory=set)

    def __post_init__(self) -> None:
        for n in self.ngram_n:
            self.observed_ngrams.setdefault(n, set())
        for name in BOUNDARY_PREDICATES:
            self.boundary_hits.setdefault(name, set())


    def observe_transitions(self, events: list[Event]) -> None:
        """Record which declared (state, event) transitions actually fired.

        The n-gram check runs one way only: observed must be legal, so the
        denominator can only ever be too large. Nothing asserted the converse
        until this existed, and a transition that never fires produces no
        evidence of its own absence, so `(ADMITTED, PREEMPT_RC)` sat in the table
        inflating every published coverage number from its first day.
        """
        from engine.sched.lifecycle import INITIAL, TRANSITIONS

        state = INITIAL
        for event in events:
            key = (state, event)
            if key not in TRANSITIONS:
                return
            self.witnessed_transitions.add(key)
            state = TRANSITIONS[key]

    def unwitnessed_transitions(self) -> list[tuple]:
        """Declared transitions no run has ever taken."""
        from engine.sched.lifecycle import TRANSITIONS

        return sorted(
            (s.value, e.value) for (s, e) in TRANSITIONS
            if (s, e) not in self.witnessed_transitions
        )

    def observe_events(self, events: list[Event]) -> None:
        """Record one request's event sequence, and its n-grams."""
        self.observe_transitions(events)
        for event in events:
            self.event_counts[event.value] = self.event_counts.get(event.value, 0) + 1
        for n in self.ngram_n:
            feasible = feasible_ngrams(n)
            for index in range(len(events) - n + 1):
                gram = tuple(events[index : index + n])
                if gram not in feasible:
                    raise UndeclaredTransition(
                        f"observed the {n}-gram {[e.value for e in gram]}, which the "
                        "lifecycle transition relation says is impossible. Either the "
                        "engine took an illegal step, or engine/sched/lifecycle.py is "
                        "missing a transition and every coverage number computed "
                        "against it is over a denominator that is too small."
                    )
                self.observed_ngrams[n].add(gram)

    def observe_boundary(self, predicate: str, value: str) -> None:
        if predicate not in BOUNDARY_PREDICATES:
            raise KeyError(f"{predicate!r} is not a declared boundary predicate")
        self.boundary_hits[predicate].add(value)

    def observe_preempt_depth(self, depth: int) -> None:
        self.preempt_depths[depth] = self.preempt_depths.get(depth, 0) + 1


    def ngram_fraction(self, n: int) -> tuple[int, int]:
        return len(self.observed_ngrams[n]), len(feasible_ngrams(n))

    def missing_ngrams(self, n: int) -> list[tuple[str, ...]]:
        missing = feasible_ngrams(n) - self.observed_ngrams[n]
        return sorted(tuple(e.value for e in gram) for gram in missing)

    def boundary_fraction(self) -> tuple[int, int]:
        hit = sum(len(self.boundary_hits[name] & set(values))
                  for name, values in BOUNDARY_PREDICATES.items())
        total = sum(len(values) for values in BOUNDARY_PREDICATES.values())
        return hit, total

    def report(self) -> str:
        lines = ["coverage", f"  denominator: {derivation_note()}", ""]

        for n in self.ngram_n:
            seen, total = self.ngram_fraction(n)
            lines.append(f"  lifecycle {n}-grams        {seen}/{total}")
            missing = self.missing_ngrams(n)
            if missing:
                shown = missing[:6]
                lines.append(f"    never reached          "
                             + ", ".join("->".join(g) for g in shown)
                             + (f", and {len(missing) - len(shown)} more"
                                if len(missing) > len(shown) else ""))

        hit, total = self.boundary_fraction()
        lines.append("")
        lines.append(f"  boundary predicates      {hit}/{total}")
        for name, values in BOUNDARY_PREDICATES.items():
            seen = self.boundary_hits[name]
            marks = " ".join(
                f"[{'x' if value in seen else ' '}]{value}" for value in values
            )
            lines.append(f"    {name:<32} {marks}")

        unwitnessed = self.unwitnessed_transitions()
        from engine.sched.lifecycle import TRANSITIONS
        lines.append("")
        lines.append(f"  declared transitions     "
                     f"{len(TRANSITIONS) - len(unwitnessed)}/{len(TRANSITIONS)} "
                     f"witnessed by this campaign unaided")
        for state, event in unwitnessed:
            lines.append(f"    not reached here       ({state}, {event})"
                         "  <- see the witness table; a targeted probe proves "
                         "reachability, it does not count as exploration")

        lines.append("")
        depths = self.preempt_depths
        top = max(depths) if depths else 0
        lines.append(f"  preemption depth         max {top}")
        for depth in range(0, max(top, 3) + 1):
            count = depths.get(depth, 0)
            bar = "#" * min(40, count)
            lines.append(f"    depth {depth}                  {count:>5} {bar}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "denominator_derivation": derivation_note(),
            "ngrams": {
                str(n): {
                    "observed": len(self.observed_ngrams[n]),
                    "feasible": len(feasible_ngrams(n)),
                    "missing": self.missing_ngrams(n),
                }
                for n in self.ngram_n
            },
            "boundary_predicates": {
                name: sorted(self.boundary_hits[name]) for name in BOUNDARY_PREDICATES
            },
            "boundary_fraction": list(self.boundary_fraction()),
            "preemption_depth": dict(sorted(self.preempt_depths.items())),
            "event_counts": dict(sorted(self.event_counts.items())),
            "transitions_witnessed": len(self.witnessed_transitions),
            "transitions_declared": None,
            "transitions_unwitnessed": self.unwitnessed_transitions(),
        }
