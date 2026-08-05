"""Promote an artifact from `results/` to `evidence/`, which is committed.

PRD 1.1 requires every published number to ship with the artifact that produced
it. `results/` is gitignored, so for most of this project's life that rule was
satisfied locally and not at all from a fresh clone: the README cited numbers
whose artifacts existed only on the machine that ran them, and the divergence
report's reproduce line named a path a reader could not have.

`evidence/` is the committed set. It is small on purpose. Promoting is a
deliberate act, one artifact at a time, because an evidence directory that
accumulates everything is `results/` with a different name and stops being a
statement about which numbers are load-bearing.

    python3 -m report.publish results/2026-08-05/fuzz-0003.json

The destination name drops the date directory and keeps the kind and sequence, so
a citation in the README is stable across reruns: `evidence/fuzz-0003.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVIDENCE_DIR = REPO_ROOT / "evidence"


def publish(source: Path, name: str | None = None) -> Path:
    """Copy one artifact into `evidence/`, refusing anything without an env."""
    artifact = json.loads(source.read_text())
    if not artifact.get("env"):
        # Security doc section 7. A claim without an environment tuple is invalid
        # by construction, so an artifact missing one must not become evidence.
        raise SystemExit(
            f"refusing to publish {source}: no env tuple. A claim without one is "
            "invalid by construction and committing it would make the evidence "
            "directory a place where that rule does not apply."
        )

    EVIDENCE_DIR.mkdir(exist_ok=True)
    destination = EVIDENCE_DIR / (name or source.name)
    shutil.copy2(source, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--name", default=None,
                        help="destination filename; only valid with one source")
    args = parser.parse_args()

    if args.name and len(args.artifacts) > 1:
        raise SystemExit("--name takes a single source artifact")

    for source in args.artifacts:
        destination = publish(source, args.name)
        size = destination.stat().st_size
        print(f"published  {destination.relative_to(REPO_ROOT)}  ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
