"""Read the scheduler trace and build per-token block widths.

The RMSNorm trace answers "how many tokens were in this launch". It cannot
answer "which tokens", because it records counts and nothing else. This module
reads the second instrument, which records
`scheduler_output.num_scheduled_tokens` per step plus a hash of each request's
prompt the first time that request is scheduled.

With both, a token's block width is measured rather than inferred: the step that
computed it is known, the step's total token count is known, and the kernel's
own predicate turns that count into a width.

Line formats, tab separated:

    epoch  new   <req_id>  <sha1(prompt_token_ids)[:16]>  <n_prompt_tokens>
    epoch  step  <total>   <n_requests>  <req_id>=<tokens>,...

Request ids are assigned fresh on every submission, so they cannot be matched
across repeats. The prompt hash can, and is why it is recorded.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.rmsnorm_trace import NARROW_BLOCK_THRESHOLD, block_width  # noqa: E402

HIDDEN = 1024  # Qwen3-0.6B; the width only moves when hidden > 256.


def load(paths: list[Path]) -> tuple[dict[str, str], list[tuple[float, int, dict]]]:
    """Returns (req_id -> prompt hash, ordered steps)."""
    prompts: dict[str, str] = {}
    steps: list[tuple[float, int, dict]] = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[1] == "new":
                prompts[parts[2]] = parts[3]
            elif len(parts) >= 5 and parts[1] == "step":
                total = int(parts[2])
                per = {}
                if parts[4]:
                    for item in parts[4].split(","):
                        req, _, tokens = item.partition("=")
                        per[req] = int(tokens)
                steps.append((float(parts[0]), total, per))
    steps.sort(key=lambda s: s[0])
    return prompts, steps


def rle(widths: list[int]) -> tuple[tuple[int, int], ...]:
    """Run-length encode a per-token width vector.

    The vectors are thousands of entries and almost entirely one value, so the
    encoded form is what goes in the artifact. It is lossless: equality of the
    encoding is equality of the vector.
    """
    out: list[list[int]] = []
    for width in widths:
        if out and out[-1][0] == width:
            out[-1][1] += 1
        else:
            out.append([width, 1])
    return tuple((w, n) for w, n in out)


def per_token_widths(steps, prompts, start: float, end: float) -> dict:
    """For each request in this window, the width each of its tokens reduced at.

    Keyed by prompt hash so the same request can be found in another repeat. A
    prompt submitted more than once in one repeat maps to a sorted list of
    vectors, and `ambiguous_prompts` reports it: identical prompts are
    interchangeable only while their vectors agree, and if they ever disagree
    the multiset stops being a faithful signature.
    """
    vectors: dict[str, list[int]] = defaultdict(list)
    seen_steps = 0
    for epoch, total, per in steps:
        if not (start <= epoch <= end) or total <= 0:
            continue
        seen_steps += 1
        width = block_width(total, HIDDEN)
        for req, tokens in per.items():
            vectors[req].extend([width] * tokens)

    by_prompt: dict[str, list] = defaultdict(list)
    unknown = 0
    for req, widths in vectors.items():
        prompt = prompts.get(req)
        if prompt is None:
            unknown += 1
            continue
        by_prompt[prompt].append(rle(widths))

    ambiguous = sorted(
        prompt for prompt, group in by_prompt.items() if len(set(group)) > 1
    )
    return {
        "steps": seen_steps,
        "signature": {prompt: sorted(group) for prompt, group in by_prompt.items()},
        "requests": sum(len(g) for g in by_prompt.values()),
        "requests_without_a_recorded_prompt": unknown,
        "ambiguous_prompts": ambiguous,
        "tokens_at_narrow_width": sum(
            count for group in by_prompt.values() for vector in group
            for width, count in vector if width == 256
        ),
    }


def step_totals(steps, start: float, end: float) -> list[int]:
    return [total for epoch, total, _ in steps
            if start <= epoch <= end and total > 0]


def crossed(steps, start: float, end: float) -> bool:
    return any(t >= NARROW_BLOCK_THRESHOLD for t in step_totals(steps, start, end))
