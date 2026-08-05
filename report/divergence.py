"""The divergence report.

Frontend spec 1.3: "This is the most important thing the CLI ever prints. It is
what gets pasted into a GitHub issue." Constraints, verbatim: fits in 80 columns,
fits on one screen, contains no information the reader must scroll to find, and
is directly pasteable into an issue without editing.

Built in week 1, before anything diverges, because deciding what a finding *is*
before having one is the point. The fields below are the definition of a finding:
if a future divergence cannot fill them in, it is not yet minimized.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

WIDTH = 80
INDENT = "  "
SIDE_LABEL_WIDTH = 23  # aligns "expected (canonical)" and "observed (fuzzed)"
META_LABEL_WIDTH = 11  # aligns "trigger", "schedule", "env"
ELLIPSIS = "…"

# Frontend spec 1.2: color carries exactly three states and nothing else.
ANSI = {
    "divergence": "\033[38;5;160m",  # --divergence #C11B2F
    "signal": "\033[38;5;130m",  # --signal     #B54A00, boundary condition hit
    "lock": "\033[38;5;29m",  # --lock       #1E5A52, invariance held
    "reset": "\033[0m",
}


def color_enabled(stream=None) -> bool:
    """NO_COLOR is respected; so is a non-tty, since this output gets piped."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


def _paint(text: str, key: str, enabled: bool) -> str:
    return f"{ANSI[key]}{text}{ANSI['reset']}" if enabled else text


def abbreviate_hash(digest: str, head: int = 4, tail: int = 2) -> str:
    """`8f3a…c1`. Short enough to align, long enough to eyeball a mismatch.

    The full digests are in the artifact; this line is for the human deciding
    whether two runs differ, which one glance at four hex characters settles.
    """
    digest = digest.strip().lower()
    if len(digest) <= head + tail + 1:
        return digest
    return f"{digest[:head]}{ELLIPSIS}{digest[-tail:]}"


def _fit(text: str, width: int) -> str:
    """Truncate to `width`, marking that truncation happened."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + ELLIPSIS


def _row(label: str, value: str, label_width: int, paint: str | None = None) -> str:
    """One `label   value` row, fitted to 80 columns.

    Fitting happens before painting so that ANSI escapes never count against the
    column budget; a colored line and a NO_COLOR line wrap identically.
    """
    body_width = WIDTH - len(INDENT) - label_width
    body = _fit(value, body_width)
    if paint:
        body = _paint(body, paint, True)
    return f"{INDENT}{label:<{label_width}}{body}"


@dataclass
class Divergence:
    """A minimized bitwise divergence between two executions of one request.

    `expected` is canonical execution C(r) per the architecture doc section 2:
    batch size 1, single uninterrupted prefill, cold cache, no speculation.
    """

    request_uid: str
    position: int
    trigger: str
    schedule_events: int
    env_fingerprint: str
    replay_artifact: str
    # A bitwise divergence carries two digests. A crash carries neither, because
    # the run that would have produced logits raised instead. Placeholder digests
    # in a document whose whole purpose is to be pasted into an issue are worse
    # than no digests, so the crash form omits the lines rather than filling them.
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    failure_class: str = "divergence"   # divergence | crash
    exception_type: str | None = None
    first_differing_byte: int | None = None
    schedule_events_before_minimization: int | None = None
    expected_label: str = "canonical"
    observed_label: str = "fuzzed"
    boundary_hit: bool = False

    def __post_init__(self) -> None:
        if self.failure_class == "divergence" and not (
            self.expected_sha256 and self.observed_sha256
        ):
            raise ValueError(
                "a divergence must carry both digests; if there are none, this is "
                "a crash and failure_class should say so"
            )
        if self.failure_class == "crash" and not self.exception_type:
            raise ValueError("a crash must name its exception type")

    def _header(self) -> str:
        parts = [
            "DIVERGENCE" if self.failure_class == "divergence" else "CRASH",
            f"req={self.request_uid}",
            f"position={self.position}",
        ]
        if self.first_differing_byte is not None:
            parts.append(f"first differing byte=0x{self.first_differing_byte:04x}")
        return _fit("  ".join(parts), WIDTH)

    def _schedule_line(self) -> str:
        plural = "" if self.schedule_events == 1 else "s"
        text = f"{self.schedule_events} event{plural}"
        before = self.schedule_events_before_minimization
        if before is not None:
            text += f", minimized from {before}"
        return text

    def render(self, color: bool | None = None, fenced: bool = False) -> str:
        """The 80-column block.

        `fenced=True` wraps the block in a Markdown code fence, which is what
        the issue-paste path uses; `for_issue()` is that path. Unfenced, GitHub
        renders the bare block as a paragraph followed by an accidental code
        block, so only the fenced form is pasteable unedited. Frontend spec 1.3
        now shows both.
        """
        painted = color_enabled() if color is None else color

        lines = [_paint(self._header(), "divergence", painted), ""]

        if self.failure_class == "divergence":
            lines += [
                _row(
                    f"expected ({self.expected_label})",
                    f"logits[{self.position}] sha256:{abbreviate_hash(self.expected_sha256)}",
                    SIDE_LABEL_WIDTH,
                ),
                _row(
                    f"observed ({self.observed_label})",
                    f"logits[{self.position}] sha256:{abbreviate_hash(self.observed_sha256)}",
                    SIDE_LABEL_WIDTH,
                ),
            ]
        else:
            # No digests exist: the run raised before producing logits. The
            # exception type is what a reader needs instead.
            lines.append(_row("class", f"crash, {self.exception_type}", SIDE_LABEL_WIDTH))

        lines += [
            "",
            _row(
                "trigger",
                self.trigger,
                META_LABEL_WIDTH,
                paint="signal" if (painted and self.boundary_hit) else None,
            ),
            _row("schedule", self._schedule_line(), META_LABEL_WIDTH),
            _row("env", self.env_fingerprint, META_LABEL_WIDTH),
            "",
            f"{INDENT}reproduce",
            _fit(f"{INDENT}{INDENT}lockstep replay {self.replay_artifact}", WIDTH),
        ]

        block = "\n".join(lines)
        return f"```\n{block}\n```" if fenced else block

    def for_issue(self) -> str:
        """The form that goes into a GitHub issue: fenced, uncolored.

        Separate from `render` so that the issue path cannot accidentally inherit
        terminal colouring or the unfenced layout.
        """
        return self.render(color=False, fenced=True)

    def __str__(self) -> str:
        return self.render(color=False)


# A synthetic case, so the layout can be inspected before anything diverges.
# The trigger is the SGLang block-boundary condition named in the PRD, prefix_len
# exactly equal to block_size, which is the poster child the architecture doc's
# coverage section asks the boundary predicates to be built around.
EXAMPLE = Divergence(
    request_uid="r02",
    position=131,
    first_differing_byte=0x1A2E,
    expected_sha256="8f3a1c04d9e2b77a6f5039c8ba14ed2277b0c9f31e6a48d5029cb7361af8e2c1",
    observed_sha256="2b7de1904cc35a8f2610bd47e9a0f38c15d2760be4193ac8f5d02e6b7a4c1309",
    trigger="cache_hit(len=64) with block_size=64",
    schedule_events=12,
    schedule_events_before_minimization=847,
    env_fingerprint="sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0",
    replay_artifact="results/2026-08-14/case-0031.json",
    boundary_hit=True,
)


# The crash variant, for the same reason the divergence example exists: the
# layout is decided before there is a real one to print.
EXAMPLE_CRASH = Divergence(
    request_uid="r00",
    position=61,
    failure_class="crash",
    exception_type="paged.OutOfBlocks",
    trigger="request needs 3 blocks, pool holds 2",
    schedule_events=0,
    schedule_events_before_minimization=9,
    env_fingerprint="sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0",
    replay_artifact="results/2026-08-04/case-0003.json",
    boundary_hit=True,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the example reports.")
    parser.add_argument("--crash", action="store_true", help="Render the crash variant.")
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Print the bare 80-column block instead of the issue-ready fenced form.",
    )
    args = parser.parse_args()
    example = EXAMPLE_CRASH if args.crash else EXAMPLE
    print(example.render() if args.terminal else example.for_issue())
