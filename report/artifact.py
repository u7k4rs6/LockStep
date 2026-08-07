"""Result artifacts."""

from __future__ import annotations

import datetime
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.envlock import EnvLock  # noqa: E402

SCHEMA_VERSION = 1
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_SEQ = re.compile(r"-(\d{4})\.json$")


@dataclass
class Artifact:
    """One claim-producing run, serialized."""

    kind: str
    payload: dict
    env: EnvLock
    created_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "created_utc": self.created_utc,
            "env": self.env.to_dict(),
            "payload": self.payload,
        }

    def write(self, results_dir: Path | None = None) -> Path:
        base = results_dir or RESULTS_DIR
        day = base / self.created_utc[:10]
        day.mkdir(parents=True, exist_ok=True)

        used = {
            int(m.group(1))
            for p in day.glob(f"{self.kind}-*.json")
            if (m := _SEQ.search(p.name))
        }
        seq = max(used, default=0) + 1
        path = day / f"{self.kind}-{seq:04d}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return path


class DirtyTree(SystemExit):
    """A claim was about to be produced from an uncommitted working tree."""


def require_clean_tree(allow_dirty: bool = False) -> dict:
    """Refuse to produce a claim artifact from a tree that is not committed."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=15)
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return {"clean": None, "override": allow_dirty,
                "note": "git unavailable; provenance not established"}

    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    revision = head.stdout.strip()[:12] if head.returncode == 0 else "unknown"

    if dirty and not allow_dirty:
        listing = "\n    ".join(dirty[:10])
        raise DirtyTree(
            f"refusing to produce a claim artifact from an uncommitted tree at "
            f"{revision}. {len(dirty)} file(s) differ:\n    {listing}\n"
            "Commit them, or pass --allow-dirty deliberately. The artifact will "
            "record that the override was used."
        )
    return {"clean": not dirty, "override": bool(dirty and allow_dirty),
            "revision": revision, "dirty_files": len(dirty)}


def load(path: Path) -> dict:
    """Read an artifact back, rejecting anything that lost its env tuple."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get("env"), dict) or not data["env"].get("fingerprint"):
        raise ValueError(f"{path}: artifact carries no environment tuple; claim invalid")
    return data


def relpath(path: Path) -> str:
    """Repo-relative path, for printing. Never leaks an absolute home path."""
    root = Path(__file__).resolve().parent.parent
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return Path(path).name
