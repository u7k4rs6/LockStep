"""`engine/` must never import from `harness/`.

The ablation's non-invariant probes live under `harness/` deliberately: one of
them uses `tl.atomic_add`, which is exactly what the static gate in
scripts/static_checks.py greps `engine/` to forbid. That separation is only worth
anything if it cannot be crossed by accident, so the direction of the dependency
is asserted here rather than left as a convention.

Checked by parsing imports rather than by importing, so a cycle or a missing
optional dependency cannot make the check silently pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_IN_ENGINE = ("harness", "bench", "certify", "report")


def module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def imports_of(path: Path) -> set[str]:
    """Top-level package names this module imports, absolute and relative."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import cannot leave the package it starts in by
                # name, so it can never reach harness/ from engine/.
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def engine_modules() -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "engine").rglob("*.py"))


def test_engine_imports_nothing_from_the_harness_or_benchmarks():
    offenders = []
    for path in engine_modules():
        for imported in imports_of(path):
            if imported in FORBIDDEN_IN_ENGINE:
                offenders.append(f"{module_name(path)} imports {imported}")
    assert not offenders, (
        "engine/ is under the invariance claim and the static gate; it must not "
        "reach into layers that deliberately contain non-invariant code:\n  "
        + "\n  ".join(offenders)
    )


def test_the_check_covers_a_real_set_of_modules():
    """A layering test that silently matched nothing would always pass."""
    modules = engine_modules()
    assert len(modules) >= 10, f"only found {len(modules)} modules under engine/"
    assert any("kernels" in str(p) for p in modules)
    assert any("sched" in str(p) for p in modules)


def test_the_atomics_probe_is_outside_the_gated_tree():
    """The probe exists, uses atomics, and is not under engine/."""
    probe = REPO_ROOT / "harness" / "mr" / "ablation.py"
    assert probe.is_file()
    assert "tl.atomic_add" in probe.read_text()
    assert not str(probe).startswith(str(REPO_ROOT / "engine"))
