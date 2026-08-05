"""The certifier's declared knobs must reach the code that uses them.

This file exists because the same failure happened four times in one session, in
four different places, and every instance had the same shape: an edit was applied
by string replacement, the pattern did not match, nothing reported an error, and
the resulting run printed a header describing behaviour it was not performing.

The worst instance is the one these tests target. `certify/session.py` printed
"cell: cache=disabled filler=varying" while calling `certify()` with neither
argument, so every cell of a factorial ran the same default configuration under
five different labels. A grid of five cells that are secretly one cell is worse
than no grid, because it looks like evidence.

The general lesson, and the reason these are tests rather than a careful reading:
a header is printed from the arguments, and the behaviour comes from the call.
Nothing forces those to agree unless something checks. These check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify.run import RELATIONS, certify  # noqa: E402

SESSION = REPO_ROOT / "certify" / "session.py"
RUN = REPO_ROOT / "certify" / "run.py"


def _certify_calls(source: Path) -> list[ast.Call]:
    tree = ast.parse(source.read_text())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "certify"
    ]


def test_session_passes_every_factorial_factor_to_certify():
    """The flags the CLI declares must reach `certify`, not just the banner.

    Without this, `--cache-mode disabled` prints "disabled" and runs whatever
    the default is, which is exactly what happened.
    """
    calls = _certify_calls(SESSION)
    assert calls, "certify/session.py never calls certify()"

    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        for required in ("cache_mode", "filler_mode"):
            assert required in keywords, (
                f"certify() is called without {required}, so the CLI flag that "
                "sets it cannot affect the run while the banner still prints it"
            )


def test_session_enforces_the_declared_concurrency_cap():
    """`max_concurrency` is read from config, not left at the function default.

    It sat unread in the security config for weeks. A default that happens to
    match is not enforcement.
    """
    calls = _certify_calls(SESSION)
    for call in calls:
        passes_cap = len(call.args) >= 6 or any(
            kw.arg == "cap" for kw in call.keywords
        )
        assert passes_cap, (
            "certify() is called without the concurrency cap, so "
            "certify/config.json's max_concurrency is decorative again"
        )


def test_every_declared_cell_has_a_relation():
    """A cell the factorial can run must say which property it tests."""
    import certify.factorial as factorial

    for cache_mode, filler_mode, _label in factorial.CELLS:
        assert (cache_mode, filler_mode) in RELATIONS, (
            f"cell ({cache_mode}, {filler_mode}) has no declared relation, so a "
            "result from it could not say what it measured"
        )


def test_cache_mode_and_filler_mode_actually_change_behaviour():
    """The parameters exist in the signature and are not ignored inside.

    A signature can accept an argument and drop it, which reads as wired from
    the outside. This asserts both names are referenced in the body.
    """
    import inspect

    source = inspect.getsource(certify)
    for name in ("cache_mode", "filler_mode"):
        # Once in the signature and at least once in the body.
        assert source.count(name) >= 2, (
            f"{name} appears only in certify()'s signature, so it is accepted "
            "and ignored"
        )


def test_cold_mode_refuses_to_score_when_the_cache_cannot_be_dropped():
    """A cold cell that could not go cold must not be recorded as a cold result.

    This is the guard that caught the mislabelled factorial: the disabled cell
    was really running cold, its cache reset failed because the endpoint was not
    enabled, and it reported vacuous rather than producing a clean-looking cell
    under the wrong name.
    """
    source = RUN.read_text()
    assert "cold_enforced" in source
    assert 'cache_mode != "cold" or cold_enforced' in source, (
        "the guard that ties a cold verdict to an enforced cache reset is gone"
    )


def test_the_witness_gauge_name_is_the_one_vllm_exposes():
    """A typo here would silently report every case as not batched."""
    from certify.run import BatchWitness

    assert BatchWitness.GAUGE == "vllm:num_requests_running"
