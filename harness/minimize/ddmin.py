"""Three-stage ddmin, with minimality verified rather than asserted."""

from __future__ import annotations

from dataclasses import dataclass

from harness.sim.driver import Case


@dataclass
class Minimization:
    case: Case
    before: dict
    after: dict
    reproduces: bool
    one_minimal: bool
    checks_run: int
    failed_removals: list[str]

    def summary(self) -> str:
        return (
            f"requests {self.before['requests']} -> {self.after['requests']}, "
            f"events {self.before['events']} -> {self.after['events']}, "
            f"prompt tokens {self.before['tokens']} -> {self.after['tokens']}"
        )


def _size(case: Case) -> dict:
    return {
        "requests": len(case.requests),
        "events": len(case.preempt_at) + len(case.refuse_cache_at) + len(case.chunk_plan),
        "tokens": sum(len(r.prompt) for r in case.requests),
    }


def _greedy_trim(items: list, still_fails, keep_at_least: int = 0) -> list:
    """Remove elements one at a time while the case keeps failing."""
    current = list(items)
    changed = True
    while changed:
        changed = False
        for index in range(len(current)):
            if len(current) <= keep_at_least:
                break
            candidate = current[:index] + current[index + 1:]
            if still_fails(candidate):
                current = candidate
                changed = True
                break
    return current


def _reduce(items: list, still_fails, keep_at_least: int = 0) -> list:
    """ddmin for the coarse cut, then a greedy pass for 1-minimality."""
    reduced = _ddmin(list(items), still_fails)
    return _greedy_trim(reduced, still_fails, keep_at_least)


def _ddmin(items: list, still_fails) -> list:
    """Classic ddmin over a list, returning a 1-minimal failing subset."""
    granularity = 2
    current = list(items)
    while len(current) >= 2:
        chunk_size = max(1, len(current) // granularity)
        chunks = [current[i:i + chunk_size] for i in range(0, len(current), chunk_size)]

        reduced = False
        for chunk in chunks:
            if len(chunk) < len(current) and still_fails(chunk):
                current, granularity, reduced = chunk, 2, True
                break
        if reduced:
            continue
        for chunk in chunks:
            complement = [x for x in current if x not in chunk]
            if complement and still_fails(complement):
                current = complement
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return current


def minimize(case: Case, fails, max_rounds: int = 6) -> Minimization:
    """Reduce `case` while `fails(case)` stays true, in three staged passes."""
    before = _size(case)
    checks = 0

    def probe(candidate: Case) -> bool:
        nonlocal checks
        checks += 1
        return fails(candidate)

    for _round in range(max_rounds):
        size_before_round = _size(case)
        case = _one_pass(case, probe)
        if _size(case) == size_before_round:
            break

    return _verify(case, fails, before, checks)


def _one_pass(case: Case, probe) -> Case:
    """One sweep of the three stages: requests, schedule events, prompt tokens."""
    def with_requests(subset):
        return case.shrunk(requests=tuple(subset))

    requests = _reduce(list(case.requests), lambda s: probe(with_requests(s)), 1)
    case = case.shrunk(requests=tuple(requests))

    def with_preempts(subset):
        return case.shrunk(preempt_at=tuple(subset))

    preempts = _reduce(list(case.preempt_at), lambda s: probe(with_preempts(s)))
    case = case.shrunk(preempt_at=tuple(preempts))

    def with_refusals(subset):
        return case.shrunk(refuse_cache_at=tuple(subset))

    refusals = _reduce(list(case.refuse_cache_at), lambda s: probe(with_refusals(s)))
    case = case.shrunk(refuse_cache_at=tuple(refusals))

    def with_chunks(subset):
        return case.shrunk(chunk_plan=tuple(subset))

    chunks = _reduce(list(case.chunk_plan), lambda s: probe(with_chunks(s)))
    case = case.shrunk(chunk_plan=tuple(chunks))

    for index, spec in enumerate(case.requests):
        def with_tokens(subset, index=index):
            specs = list(case.requests)
            specs[index] = type(spec)(
                uid=spec.uid, prompt=tuple(subset), seed=spec.seed,
                max_new_tokens=spec.max_new_tokens,
                temperature=spec.temperature, top_p=spec.top_p,
            )
            return case.shrunk(requests=tuple(specs))

        if len(spec.prompt) > 1:
            tokens = _reduce(list(spec.prompt), lambda s: probe(with_tokens(s)), 1)
            if tokens and len(tokens) < len(spec.prompt):
                case = with_tokens(tokens)

    return case


def _verify(case: Case, fails, before: dict, checks: int) -> "Minimization":
    """Check the two things a minimization claim rests on."""
    reproduces = fails(case)

    # Every single-element removal is attempted and must fail. Without this the
    # three passes leave removable elements behind, which the check caught.
    one_minimal = True
    failed_removals: list[str] = []
    for label, field, values in (
        ("request", "requests", case.requests),
        ("preemption", "preempt_at", case.preempt_at),
        ("cache refusal", "refuse_cache_at", case.refuse_cache_at),
        ("chunk", "chunk_plan", case.chunk_plan),
    ):
        for position in range(len(values)):
            trimmed = tuple(v for i, v in enumerate(values) if i != position)
            if not trimmed and field == "requests":
                continue
            if fails(case.shrunk(**{field: trimmed})):
                one_minimal = False
                failed_removals.append(f"{label}[{position}] could still be removed")

    for index, spec in enumerate(case.requests):
        for position in range(len(spec.prompt)):
            if len(spec.prompt) <= 1:
                break
            trimmed = tuple(t for i, t in enumerate(spec.prompt) if i != position)
            specs = list(case.requests)
            specs[index] = type(spec)(
                uid=spec.uid, prompt=trimmed, seed=spec.seed,
                max_new_tokens=spec.max_new_tokens,
                temperature=spec.temperature, top_p=spec.top_p,
            )
            if fails(case.shrunk(requests=tuple(specs))):
                one_minimal = False
                failed_removals.append(f"{spec.uid} token[{position}] could still be removed")
                break

    return Minimization(
        case=case,
        before=before,
        after=_size(case),
        reproduces=reproduces,
        one_minimal=one_minimal,
        checks_run=checks,
        failed_removals=failed_removals,
    )
