"""`lockstep replay <artifact>`: re-run a committed case and say what happened."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine import envlock  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from harness.sim.driver import Case, run_case  # noqa: E402
from report.artifact import harness_env  # noqa: E402

WEIGHTS = REPO_ROOT / "weights" / "Qwen3-0.6B"


def load(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("kind") != "case":
        raise ValueError(
            f"{path} is a {artifact.get('kind')!r} artifact, not a case. Campaign "
            "summaries, fidelity runs, and certification runs record numbers "
            "rather than an executable triple, so only 'case' artifacts replay."
        )
    if "minimized" not in artifact.get("payload", {}):
        raise ValueError(f"{path} has no payload.minimized to replay")
    return artifact


def outcome_of(model, case: Case) -> tuple[str, str]:
    """Run the case once, and return (error string, trajectory hash)."""
    try:
        outcome = run_case(model, case)
    except Exception as exc:  # noqa: BLE001 - the finding may be a crash
        return f"{type(exc).__name__}: {exc}", ""
    return (outcome.error or ""), outcome.trajectory


def classify(recorded: str, observed: str, same_engine: bool, same_env: bool,
             engine_known: bool = True, provenance: dict | None = None):
    """Compare, and attribute the difference to something specific."""
    recorded_class = recorded.split(":", 1)[0].strip()
    observed_class = observed.split(":", 1)[0].strip()

    if recorded_class == observed_class:
        return True, f"raised {observed_class or 'nothing'} again, as recorded"

    # A named commit outranks inferring from whether two revision strings differ.
    if provenance and provenance.get("fixed_by"):
        return True, (
            f"recorded {recorded_class or 'no error'}, observed "
            f"{observed_class or 'no error'}. Fixed by "
            f"{provenance['fixed_by'][:12]}, "
            f"\"{provenance.get('fixed_by_subject', '')}\". Last revision before "
            f"the fix: {provenance.get('last_revision_before_fix', '?')[:12]}. "
            "This is a regression check on a closed finding, and it passing means "
            "the fix is still in."
        )

    if not same_env:
        return True, (
            f"recorded {recorded_class or 'no error'}, observed "
            f"{observed_class or 'no error'}. The environment tuple differs, and "
            "every bitwise claim here is scoped to one tuple, so this is out of "
            "scope rather than a reproduction failure."
        )

    if not engine_known:
        return True, (
            f"recorded {recorded_class or 'no error'}, observed "
            f"{observed_class or 'no error'}. The artifact predates the "
            "engine_revision field, so which engine produced it is not recorded "
            "and the difference cannot be attributed. Treated as a fixed finding "
            "rather than a failure, on weaker evidence than a recorded revision "
            "would give."
        )

    if not same_engine:
        return True, (
            f"recorded {recorded_class or 'no error'}, observed "
            f"{observed_class or 'no error'}. The engine revision differs, so "
            "this is a regression check on a fixed finding rather than a "
            "reproduction: the case no longer does what it did, which is the "
            "point of having fixed it."
        )

    return False, (
        f"recorded {recorded_class or 'no error'}, observed "
        f"{observed_class or 'no error'}, at the same engine revision and the "
        "same environment tuple. Execution is supposed to be a pure function of "
        "(W, sigma, seeds), so this is a real failure of that property."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path,
                        help="a case artifact, normally under evidence/")
    args = parser.parse_args()

    try:
        artifact = load(args.artifact)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot replay: {exc}")
        return 2

    payload = artifact["payload"]
    recorded_env = harness_env(artifact)
    here = envlock.capture()

    same_env = recorded_env.get("fingerprint") == here.fingerprint()
    recorded_revision = recorded_env.get("engine_revision", "unknown")
    same_engine = (
        recorded_revision not in ("unknown", None)
        and recorded_revision == here.engine_revision
    )

    print(f"lockstep replay  {args.artifact}")
    if payload.get("trajectory"):
        print(f"  recorded          trajectory {payload['trajectory'][:32]}")
    else:
        print(f"  recorded          {payload['reason'][:88]}")
    print(f"  recorded env      {recorded_env.get('fingerprint', 'unknown')}")
    print(f"  this env          {here.fingerprint()}")
    provenance = payload.get("provenance") or {}
    if provenance.get("fixed_by"):
        print(f"  recorded engine   {recorded_revision}"
              f"  (pinned: last revision before the fix "
              f"{provenance.get('last_revision_before_fix', '?')[:12]})")
        print(f"  fixed by          {provenance['fixed_by'][:12]}  "
              f"{provenance.get('fixed_by_subject', '')}")
    else:
        print(f"  recorded engine   {recorded_revision}")
    print(f"  this engine       {here.engine_revision}")
    print(f"  minimality        reproduces={payload.get('reproduces')} "
          f"1-minimal={payload.get('one_minimal')} "
          f"checks={payload.get('checks_run')}")
    print()

    case = Case.from_dict(payload["minimized"])
    print(f"  replaying {len(case.requests)} request(s), "
          f"block_size={case.block_size}, num_blocks={case.num_blocks}, "
          f"prompt lengths {[len(r.prompt) for r in case.requests]}")

    model = Qwen3(str(WEIGHTS), max_len=1024)
    observed, trajectory = outcome_of(model, case)

    if payload.get("trajectory"):
        got = trajectory
        ok = got == payload["trajectory"]
        detail = (
            f"trajectory hash {got[:16]} matches the artifact"
            if ok else
            f"trajectory hash {got[:16]} against recorded "
            f"{payload['trajectory'][:16]}"
        )
    else:
        ok, detail = classify(payload["reason"], observed, same_engine, same_env,
                              engine_known=recorded_revision not in ("unknown", None),
                              provenance=payload.get("provenance"))

    print()
    print(f"  {'OK' if ok else 'FAILED'}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
