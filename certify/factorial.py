"""The 2x2 that separates batch composition from cache cold-versus-warm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.run import RELATIONS  # noqa: E402

CELLS = [
    ("disabled", "varying", "cleanest: caching off, geometry varies"),
    ("warm", "varying", "batch composition alone"),
    ("cold", "fixed", "cache cold vs warm alone"),
    ("warm", "fixed", "baseline: neither varies"),
    ("cold", "varying", "both, confounded (the original run)"),
    ("disabled", "fixed", "strictest baseline: caching off, geometry fixed"),
]


def run_cell(cache_mode: str, filler_mode: str, label: str, repeats: int,
             invariant: bool) -> dict | None:
    """One cell, in its own server lifetime. Returns the parsed artifact."""
    command = [
        sys.executable, "-m", "certify.session",
        "--repeats", str(repeats),
        "--cache-mode", cache_mode,
        "--filler-mode", filler_mode,
        "--label", label,
    ]
    if not invariant:
        command.append("--no-invariant")

    print(f"\n=== cell: cache={cache_mode} filler={filler_mode}  {label}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                               timeout=3600)
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        sys.stdout.write(completed.stderr[-2000:])
        return None

    for line in completed.stdout.splitlines():
        if line.startswith("artifact  "):
            path = REPO_ROOT / line.split("artifact  ", 1)[1].strip()
            return json.loads(path.read_text())["payload"] | {"_path": str(path)}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1,
                        help="how many times to repeat the whole grid, each in "
                             "fresh server lifetimes")
    parser.add_argument("--no-invariant", action="store_true",
                        help="run the grid against default mode instead")
    args = parser.parse_args()

    invariant = not args.no_invariant
    grid: list[dict] = []

    for run_index in range(args.runs):
        for cache_mode, filler_mode, label in CELLS:
            payload = run_cell(cache_mode, filler_mode, label, args.repeats, invariant)
            if payload is None:
                print(f"  cell failed: cache={cache_mode} filler={filler_mode}")
                grid.append({"run": run_index, "cache_mode": cache_mode,
                             "filler_mode": filler_mode, "failed": True})
                continue
            diverged = [r["case"] for r in payload["results"] if not r["clean"]]
            vacuous = [r["case"] for r in payload["results"] if r.get("vacuous")]
            grid.append({
                "run": run_index,
                "cache_mode": cache_mode,
                "filler_mode": filler_mode,
                "label": label,
                "relation": payload.get("relation", ""),
                "clean": payload["clean"],
                "total": payload["total"],
                "batched": payload.get("batched_cases"),
                "peak_concurrent": payload.get("max_concurrent_running"),
                "diverged": diverged,
                "vacuous": vacuous,
                "artifact": payload["_path"],
            })

    print("\n\nfactorial")
    print(f"  {'run':>3} {'cache':<9} {'filler':<8} {'clean':>7} {'batched':>8} "
          f"{'peak':>5}  diverged")
    print("  " + "-" * 92)
    for cell in grid:
        if cell.get("failed"):
            print(f"  {cell['run']:>3} {cell['cache_mode']:<9} "
                  f"{cell['filler_mode']:<8}  CELL FAILED")
            continue
        print(f"  {cell['run']:>3} {cell['cache_mode']:<9} {cell['filler_mode']:<8} "
              f"{cell['clean']:>3}/{cell['total']:<3} {cell['batched']:>8} "
              f"{cell['peak_concurrent']:>5}  "
              + (", ".join(c[:34] for c in cell["diverged"]) or "none"))

    out = REPO_ROOT / "results" / "factorial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mode": "VLLM_BATCH_INVARIANT=1" if invariant else "default",
        "repeats": args.repeats,
        "runs": args.runs,
        "relations": {f"{k[0]}/{k[1]}": v for k, v in RELATIONS.items()},
        "grid": grid,
    }, indent=2) + "\n")
    print(f"\ngrid  {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
