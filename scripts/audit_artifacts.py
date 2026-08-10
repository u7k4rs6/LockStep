"""Audit every committed artifact: one environment tuple, and fields read strictly.

This ran once as a throwaway and reported "0 of 10 killed" from an artifact that
says 10 of 10, because it read a field name that does not exist and took the
default. It is committed now, and it reads through `report.artifact.Record`, so
the same mistake raises instead of printing a number a reader would believe.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from report.artifact import (  # noqa: E402
    Record, SubjectNotRecorded, harness_env, read, subject_env,
)

EVIDENCE = REPO_ROOT / "evidence"
TUPLE_FIELDS = ("torch_version", "triton_version", "cuda_version", "driver_version",
                "gpu_arch", "python_version", "numpy_version")

# Files in evidence/ that are not result artifacts and carry no env by design.
NOT_ARTIFACTS = {
    "index.json": "the pointer file naming which artifact backs each claim",
    "upstream-finding.json": "a provenance index for vllm#51187, pointing at "
                             "artifacts that each carry their own tuple",
    "prediction-rmsnorm-blocksize.json": "a pre-registration, written before the "
                                         "run it predicts; it has no result and "
                                         "so no environment to report",
}


def env_tuples() -> tuple[dict, list[str]]:
    groups = collections.defaultdict(list)
    skipped = []
    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name in NOT_ARTIFACTS:
            skipped.append(path.name)
            continue
        env = harness_env(read(path))
        groups[tuple(env[f] for f in TUPLE_FIELDS)].append(path.name)
    return groups, skipped


def main() -> int:
    print("1. every result artifact reports the same environment tuple")
    groups, skipped = env_tuples()
    for tup, members in groups.items():
        print(f"   {len(members)} artifact(s):")
        for field, value in zip(TUPLE_FIELDS, tup):
            print(f"     {field:<16} {value}")
        for name in members:
            print(f"       {name}")
    for name in skipped:
        print(f"   not a result artifact, no tuple expected: {name}"
              f"  ({NOT_ARTIFACTS[name]})")

    print()
    print("2. the subject environment, where an artifact records one")
    # A certify artifact's env.lock describes the certifier, not the engine it
    # certified. Artifacts written before this field existed say so rather than
    # being silently reported as agreeing.
    for path in sorted(EVIDENCE.glob("certify-*.json")):
        doc = read(path)
        try:
            subject = subject_env(doc)
        except SubjectNotRecorded:
            # Schema 1 kept the field in the payload for one artifact before the
            # environment block existed. Read it there rather than reporting a
            # tuple that was recorded as absent.
            subject = doc["payload"].optional(
                "subject_env", None,
                because="added after the certify artifacts backing vllm#51187 "
                        "were written; those are deliberately not being re-run",
            )
        if subject is None:
            print(f"   {path.name:<32} not recorded (predates the field)")
        else:
            print(f"   {path.name:<32} {subject['engine']}, "
                  f"torch {subject['torch_version']}, {subject['python_version']}")

    print()
    print("3. mutation verdicts, read by their real field name")
    fuzz = read(EVIDENCE / json.loads((EVIDENCE / "index.json").read_text())
                ["selected"]["fuzz"])
    faults = fuzz["payload"]["seeded_faults"]
    verdicts = collections.Counter(f["verdict"] for f in faults)
    exercised = sum(1 for f in faults if f["mutation_took_effect"])
    print(f"   {dict(verdicts)}, mutation took effect on {exercised}/{len(faults)}")

    print()
    if len(groups) > 1:
        print("FAIL: result artifacts span more than one environment tuple, so the "
              "scope claim in README.md is wrong as written")
        return 1
    print(f"PASS  {sum(len(m) for m in groups.values())} result artifacts, "
          f"{len(groups)} environment tuple, {len(skipped)} non-artifact file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
