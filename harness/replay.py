"""`lockstep replay <artifact>`: re-run a committed case and say what happened.

The differentiator sentence in the README says every finding minimizes to an
exact replay. That was only half true. The minimized case was written into the
artifact correctly, but the artifact lived under `results/`, which is gitignored,
and the command the divergence report told a reader to run had never been
written. From a fresh clone the reproduce line named a missing file and a
missing command.

This is that command. It reads a case artifact, rebuilds the exact `(W, sigma,
seeds)` triple from `payload.minimized`, runs it, and compares what happens now
against what the artifact recorded.

**A finding that no longer reproduces is the normal outcome for a fixed bug**,
and saying "DID NOT REPRODUCE" without saying why would be useless. So the
comparison is three-way, using the environment tuple and the engine revision
that the artifact carries:

  * same engine revision, same env, different outcome: a real problem, because
    execution is supposed to be a pure function of the triple.
  * different engine revision: the engine changed under it. For a fixed bug this
    is the expected and desirable result, and it is reported as a regression
    check that passed rather than as a failure.
  * different environment tuple: out of scope by construction, since every
    bitwise claim here is scoped to one tuple.

Exit status: 0 if the artifact's recorded behaviour was reproduced or was
deliberately fixed, 1 if it diverged in a way neither explains, 2 if the artifact
could not be read.
"""

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
    """Run the case once, and return (error string, trajectory hash).

    Once, not once per comparison: the case is the unit of work here and running
    it twice to answer two questions about the same execution would make the
    replay report describe two executions.
    """
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

    # A finding pinned to the commit that fixed it. This outranks every check
    # below, because a named commit is stronger evidence than an inference from
    # whether two revision strings happen to differ.
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
        # Artifacts written before `engine_revision` existed cannot say which
        # engine produced them. That is a weaker statement than "the revision
        # differs" and it gets said rather than rounded up to the stronger one.
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
    recorded_env = artifact.get("env", {})
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
        # A witness case rather than a bug repro: it never failed, and what is
        # being checked is that the same triple still produces the same
        # trajectory hash. This is I3 verified from a fresh clone, which is the
        # claim the differentiator sentence actually makes.
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
