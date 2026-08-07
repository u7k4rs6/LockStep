"""The report's artifact choice must be declared, not derived from a filename.

Two defects motivate this file. The report picked whichever name sorted last in
evidence/, so publishing an artifact could move a figure with no visible cause.
And the README's evidence table named two files that had been superseded, which
nothing checked because nothing read the table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVIDENCE = REPO_ROOT / "evidence"
INDEX = EVIDENCE / "index.json"
README = REPO_ROOT / "README.md"

KINDS_THE_REPORT_READS = ("verify", "fuzz", "fidelity", "throughput", "certify")


def index() -> dict:
    return json.loads(INDEX.read_text())


def test_every_kind_the_report_reads_is_declared():
    declared = index()["selected"]
    for kind in KINDS_THE_REPORT_READS:
        assert kind in declared, (
            f"the report renders {kind!r} but evidence/index.json does not name "
            "an artifact for it, so the choice falls back to a directory listing"
        )


def test_every_declared_artifact_exists_and_matches_its_kind():
    for kind, name in index()["selected"].items():
        path = EVIDENCE / name
        assert path.is_file(), f"index.json names {name} for {kind}, which is absent"
        assert json.loads(path.read_text()).get("kind") == kind, (
            f"{name} is filed under {kind!r} but does not declare that kind"
        )


def test_every_committed_artifact_is_accounted_for():
    """An artifact in evidence/ that no entry mentions is unexplained weight."""
    data = index()
    known = set(data["selected"].values()) | set(data["supporting"]) | {INDEX.name}
    present = {p.name for p in EVIDENCE.glob("*.json")}
    assert present <= known, (
        f"committed but unexplained: {sorted(present - known)}. Every artifact in "
        "evidence/ is either what backs a claim or support for one."
    )
    assert known - {INDEX.name} <= present, (
        f"named but missing: {sorted(known - present - {INDEX.name})}"
    )


def test_the_selection_is_not_reproducible_by_sorting_or_recency():
    """The sentinel: if someone restores either rule, this fails.

    Both rejected rules must actually disagree with the declared choice, or this
    file would pass against the code it exists to prevent.
    """
    declared = index()["selected"]
    by_name, by_time = 0, 0
    for kind, name in declared.items():
        candidates = sorted(EVIDENCE.glob(f"{kind}-*.json"))
        if candidates and candidates[-1].name != name:
            by_name += 1
        newest = max(
            candidates, key=lambda p: json.loads(p.read_text())["created_utc"],
            default=None,
        )
        if newest is not None and newest.name != name:
            by_time += 1
    assert by_name, "no declared choice differs from the filename sort"
    assert by_time, "no declared choice differs from the newest artifact"


@pytest.mark.parametrize("named", sorted(set(
    re.findall(r"evidence/[A-Za-z0-9_.-]+\.json", README.read_text())
)))
def test_every_evidence_file_the_readme_names_exists(named):
    assert (REPO_ROOT / named).is_file(), (
        f"README.md points a reader at {named}, which is not committed"
    )


def test_readme_evidence_table_covers_what_backs_the_claims():
    text = README.read_text()
    for name in index()["selected"].values():
        assert f"evidence/{name}" in text, (
            f"{name} backs a published figure but the README's evidence table "
            "does not name it, so a reader cannot get from a number to its artifact"
        )
