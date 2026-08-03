"""The 80-column constraint from frontend spec 1.3 is a test, not a convention."""

from __future__ import annotations

import re
import sys
from pathlib import Path


from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report.divergence import WIDTH, Divergence, abbreviate_hash  # noqa: E402

ANSI_RE = re.compile(r"\033\[[0-9;]*m")

SYNTHETIC = Divergence(
    request_uid="r02",
    position=131,
    first_differing_byte=0x1A2E,
    expected_sha256="8f3a" + "0" * 58 + "c1",
    observed_sha256="2b7d" + "0" * 58 + "09",
    trigger="cache_hit(len=64) with block_size=64",
    schedule_events=12,
    schedule_events_before_minimization=847,
    env_fingerprint="sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0",
    replay_artifact="results/2026-08-14/case-0031.json",
    boundary_hit=True,
)


def test_matches_the_spec_layout():
    assert SYNTHETIC.render(color=False).splitlines() == [
        "DIVERGENCE  req=r02  position=131  first differing byte=0x1a2e",
        "",
        "  expected (canonical)   logits[131] sha256:8f3a…c1",
        "  observed (fuzzed)      logits[131] sha256:2b7d…09",
        "",
        "  trigger    cache_hit(len=64) with block_size=64",
        "  schedule   12 events, minimized from 847",
        "  env        sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0",
        "",
        "  reproduce",
        "    lockstep replay results/2026-08-14/case-0031.json",
    ]


def test_fits_on_one_screen():
    """'Fits on one screen' with no scrolling: keep it well under a 24-row term."""
    assert len(SYNTHETIC.render(color=False).splitlines()) <= 24


def test_leads_with_the_reproduce_command():
    """Frontend spec 1.2: failures lead with the repro, not a stack trace.

    The reproduce block is the last thing printed and therefore the first thing
    visible above a shell prompt, and no line after it competes for attention.
    """
    lines = SYNTHETIC.render(color=False).splitlines()
    assert lines[-2].strip() == "reproduce"
    assert lines[-1].strip().startswith("lockstep replay ")


def test_color_never_changes_the_layout():
    plain = SYNTHETIC.render(color=False)
    stripped = ANSI_RE.sub("", SYNTHETIC.render(color=True))
    assert plain == stripped


def test_fenced_form_is_the_block_plus_fences():
    fenced = SYNTHETIC.render(color=False, fenced=True).splitlines()
    assert fenced[0] == "```" and fenced[-1] == "```"
    assert fenced[1:-1] == SYNTHETIC.render(color=False).splitlines()


# Divergence fields are engine-generated single-line values. Control characters
# are excluded from the strategy rather than skipped inside the test, because a
# skip inside a Hypothesis test abandons the whole property rather than the one
# example, which would silently stop checking the column budget at all.
SINGLE_LINE = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp"))
)


@settings(max_examples=200)
@given(
    uid=SINGLE_LINE.filter(lambda s: len(s) >= 1),
    position=st.integers(min_value=0, max_value=10**9),
    trigger=SINGLE_LINE,
    fingerprint=SINGLE_LINE,
    artifact=SINGLE_LINE.filter(lambda s: len(s) >= 1),
    events=st.integers(min_value=0, max_value=10**7),
    before=st.one_of(st.none(), st.integers(min_value=0, max_value=10**7)),
    byte=st.one_of(st.none(), st.integers(min_value=0, max_value=2**32)),
    color=st.booleans(),
)
def test_never_exceeds_80_columns(
    uid, position, trigger, fingerprint, artifact, events, before, byte, color
):
    """Long triggers and long artifact paths are the realistic overflow risks."""
    rendered = Divergence(
        request_uid=uid,
        position=position,
        first_differing_byte=byte,
        expected_sha256="a" * 64,
        observed_sha256="b" * 64,
        trigger=trigger,
        schedule_events=events,
        schedule_events_before_minimization=before,
        env_fingerprint=fingerprint,
        replay_artifact=artifact,
    ).render(color=color)

    for line in ANSI_RE.sub("", rendered).splitlines():
        assert len(line) <= WIDTH, f"{len(line)} columns: {line!r}"


def test_abbreviate_hash_keeps_both_ends():
    assert abbreviate_hash("8f3a" + "0" * 58 + "c1") == "8f3a…c1"
    assert abbreviate_hash("abc") == "abc"
