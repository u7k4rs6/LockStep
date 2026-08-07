"""The concurrency ladder: does divergence scale with batch width?"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUNGS = [
    (0, True, "rung 0: sequential, one in flight"),
    (0, False, "rung 1: concurrent, no fillers"),
    (1, False, "rung 2: 1 filler"),
    (4, False, "rung 3: 4 fillers"),
    (13, False, "rung 4: 13 fillers"),
    (31, False, "rung 5: 31 fillers"),
]


def run_rung(width: int, sequential: bool, label: str, repeats: int) -> dict | None:
    command = [
        sys.executable, "-m", "certify.session",
        "--repeats", str(repeats),
        "--cache-mode", "warm",
        "--filler-mode", "fixed",
        "--fixed-width", str(width),
        "--label", label,
    ]
    if sequential:
        command.append("--sequential")

    print(f"\n=== {label}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=3600)
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        sys.stdout.write(completed.stderr[-1500:])
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("artifact  "):
            path = REPO_ROOT / line.split("artifact  ", 1)[1].strip()
            return json.loads(path.read_text())["payload"] | {"_path": str(path)}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-invariant", action="store_true")
    args = parser.parse_args()

    rows = []
    for width, sequential, label in RUNGS:
        payload = run_rung(width, sequential, label, args.repeats)
        if payload is None:
            rows.append({"label": label, "failed": True})
            continue
        rows.append({
            "label": label,
            "width": width,
            "sequential": sequential,
            "clean": payload["clean"],
            "total": payload["total"],
            "peak_concurrent": payload.get("max_concurrent_running"),
            "diverged": [r["case"] for r in payload["results"] if not r["clean"]],
            "artifact": payload["_path"],
        })

    print("\n\nconcurrency ladder")
    print(f"  {'rung':<34} {'width':>6} {'clean':>7} {'peak':>5}")
    print("  " + "-" * 58)
    for row in rows:
        if row.get("failed"):
            print(f"  {row['label']:<34}  RUNG FAILED")
            continue
        print(f"  {row['label']:<34} {row['width']:>6} "
              f"{row['clean']:>3}/{row['total']:<3} {row['peak_concurrent']:>5}")

    out = REPO_ROOT / "results" / "ladder.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rungs": rows}, indent=2) + "\n")
    print(f"\nladder  {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
