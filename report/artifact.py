"""Result artifacts.

Frontend spec 1.1: "Every command that produces a claim writes a JSON artifact
to `results/` with `env.lock` embedded. `report` consumes only those artifacts,
never live state, so the report is always reproducible from committed data."

Security doc section 7: a claim without an environment tuple is invalid by
construction, so `env` is not an optional field and there is no writer path that
omits it.
"""

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
    """One claim-producing run, serialized.

    `kind` names the command that produced it ("fidelity", "invariance"). The
    file path is `results/<UTC date>/<kind>-NNNN.json`, matching the shape the
    frontend spec's divergence report prints as a reproduce target.
    """

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
