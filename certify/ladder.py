"""The concurrency ladder: does divergence scale with batch width?

The factorial failed to separate its factors, and the reason is worth stating
plainly because it is a limit of black-box testing rather than a bug in the
script. Holding the *filler count* fixed does not hold the *batch geometry*
fixed. vLLM batches whatever is in flight at each decode step, so arrival jitter
reshuffles the step-level composition between repeats no matter how many requests
the client sends. Every cell, the baseline included, had batch composition
varying underneath it, which is why the baseline diverged too.

What a black box can still do is vary the *amount* of cohabitation and look for a
dose-response. That is this file. Each rung holds the cohabitant count fixed for
all repeats and only changes between rungs:

    rung 0   sequential, one request in flight    the negative control
    rung 1   0 fillers, concurrent                the case's own requests only
    rung 2   1 filler
    rung 3   4 fillers
    rung 4   13 fillers
    rung 5   31 fillers

The reading, decided before the run:

  * clean at rung 0 and degrading as width grows: cohabitation is implicated, and
    vLLM's claim covers exactly that ("independent of the batch size or the order
    of requests in a batch")
  * already diverging at rung 0: not batching at all, and the finding is about
    something else entirely, most likely this harness
  * flat across every rung including rung 0: run-to-run nondeterminism unrelated
    to width, which the claim's first clause covers and its second does not

Rung 0 matters most. The withdrawn sequential certification was 7 of 7 clean, but
it ran on older code, so it cannot serve as a control for the current one. This
reproduces that condition on the current code path, which is the only way a
difference between sequential and concurrent is attributable to concurrency
rather than to everything else that changed in between.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# (fixed filler width, sequential, label). Cache is held warm throughout so the
# cache is not a variable; the factorial already showed warm and cold behave the
# same, so warm is chosen for being the cheaper of the two.
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
