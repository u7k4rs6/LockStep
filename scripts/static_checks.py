"""Static gates on kernels under claim.

docs/02-technical-architecture.md section 12, gate 4: "no `tl.atomic_` in kernels
under claim, no autotune decorators, kernel config registry unchanged without an
accompanying claims-table review."

The first two are checked here. The third is a CI policy over commits, not a
property of a working tree, so it is not implemented in week 1; the input it
needs, the registry digest in env.lock, exists.

These are greps by intent, not by implementation: the architecture doc says CI
"greps for `tl.atomic_`", and a grep is what a reader can rerun by hand and
trust. The AST would be more precise and less checkable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# "Kernels under claim" is every kernel the invariance claims run through.
KERNEL_ROOTS = [REPO_ROOT / "engine"]

CHECKS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "atomics on accumulators",
        re.compile(r"tl\.atomic_"),
        "Atomic accumulation orders by arrival, which is a function of the "
        "scheduler rather than of the request. No split-K, no atomic combine.",
    ),
    (
        "autotune decorators",
        re.compile(r"@\s*triton\.autotune|@\s*autotune|triton\.autotune\s*\("),
        "Autotuning selects by measured time, so the same binary can pick "
        "different configs across runs. Configs are pinned in "
        "engine/kernels/registry.py.",
    ),
    (
        "torch reductions in the engine",
        re.compile(r"torch\.(matmul|softmax|sum|mean|einsum|bmm)\s*\("),
        "Kernel selection for these is shape-dependent and therefore "
        "batch-dependent. engine/model/qwen3.py's docstring has asserted their "
        "absence since week 1 and nothing checked it, which is a rule enforced "
        "by prose. engine/sampler/philox.py is the one exception, documented "
        "there: its softmax and cumsum run on CPU in fp64 over a single row, "
        "outside any batched path.",
    ),
    (
        "triton heuristics",
        re.compile(r"@\s*triton\.heuristics|@\s*heuristics\b"),
        "A heuristic keyed on a runtime shape is how a batch-derived quantity "
        "reaches a kernel config without anyone deciding to let it.",
    ),
]

SELF = Path(__file__).resolve()

# The sampler's CPU fp64 top-p path. Named rather than pattern-matched, so adding
# a second exception is a visible edit to this list.
EXCEPTIONS = {"torch reductions in the engine": {"engine/sampler/philox.py"}}


def python_sources() -> list[Path]:
    seen: list[Path] = []
    for root in KERNEL_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.resolve() != SELF:
                seen.append(path)
    return seen


def scan() -> tuple[list[str], int]:
    findings: list[str] = []
    files = python_sources()
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for label, pattern, _ in CHECKS:
                if str(rel) in EXCEPTIONS.get(label, ()):
                    continue
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: {label}: {line.strip()}")
    return findings, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show", action="store_true", help="Also print the pinned config registry."
    )
    args = parser.parse_args()

    findings, file_count = scan()

    scope = ", ".join(str(r.relative_to(REPO_ROOT)) + "/" for r in KERNEL_ROOTS)
    print(f"static checks over {file_count} files under {scope}")
    for label, pattern, _ in CHECKS:
        hits = [f for f in findings if f": {label}: " in f]
        status = f"{len(hits)} found" if hits else "none found"
        print(f"  {label:<26} /{pattern.pattern}/  ->  {status}")

    if findings:
        print("\nFAIL", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        for label, _, why in CHECKS:
            if any(f": {label}: " in f for f in findings):
                print(f"\n{label}: {why}", file=sys.stderr)
        return 1

    if args.show:
        sys.path.insert(0, str(REPO_ROOT))
        from engine.kernels import registry

        print()
        print(registry.describe())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
