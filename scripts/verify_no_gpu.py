"""Check what can be checked without a GPU or a model download."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.sched.lifecycle import (  # noqa: E402
    TRANSITIONS, UNREACHABLE_BY_DESIGN, feasible_ngrams, reachable_states,
)
from harness.sim.driver import Case  # noqa: E402
from report.artifact import harness_env  # noqa: E402


def main() -> int:
    ok = True
    print("lockstep verify, CPU only, no weights required")
    print()

    print("1. coverage denominators, recomputed from the declared transition relation")
    two, three = len(feasible_ngrams(2)), len(feasible_ngrams(3))
    print(f"   {len(TRANSITIONS)} transitions over {len(reachable_states())} reachable states")
    print(f"   2-grams {two}   3-grams {three}   (README claims 25 and 79)")
    ok &= (two, three) == (25, 79)
    overlap = set(TRANSITIONS) & set(UNREACHABLE_BY_DESIGN)
    print(f"   declared legal and unreachable at once: {len(overlap)}")
    ok &= not overlap

    print()
    print("2. evidence artifacts parse, carry an env tuple, and hash to these digests")
    for path in sorted((REPO_ROOT / "evidence").glob("*.json")):
        raw = path.read_bytes()
        doc = json.loads(raw)
        env = harness_env(doc) if ("env" in doc or "environment" in doc) else {}
        rev = env.get("engine_revision", "n/a")
        print(f"   {path.name:<32} sha256:{hashlib.sha256(raw).hexdigest()[:16]}  {rev}")
        if "env" in doc or "environment" in doc:
            ok &= bool(env.get("fingerprint"))

    print()
    print("3. the committed repro rebuilds without running")
    case = Case.from_dict(json.loads((REPO_ROOT / "evidence" / "case-0003.json").read_text())["payload"]["minimized"])
    total = len(case.requests[0].prompt) + case.requests[0].max_new_tokens
    needed = -(-total // case.block_size)
    print(f"   {len(case.requests)} request, {len(case.requests[0].prompt)} prompt tokens, "
          f"{case.requests[0].max_new_tokens} new")
    print(f"   needs {needed} blocks at block_size {case.block_size}, pool holds {case.num_blocks}")
    print(f"   so it can never be admitted, which is the finding, visible without a GPU")
    ok &= needed > case.num_blocks

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
