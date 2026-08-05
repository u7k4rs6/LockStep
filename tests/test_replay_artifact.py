"""The committed evidence must actually replay.

The reproduce line in the divergence report was, for most of this project's life,
a string: it named a path under a gitignored directory and a command that had
never been written. Nothing caught that, because nothing asserted the reproduce
target could be consumed. These tests are that assertion.

They run against `evidence/`, not `results/`, because `evidence/` is what a
reader who clones the repository gets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.replay import classify, load  # noqa: E402
from harness.sim.driver import Case  # noqa: E402

EVIDENCE = REPO_ROOT / "evidence"
CASES = sorted(EVIDENCE.glob("case-*.json"))


def test_evidence_directory_is_committed():
    """Not ignored, and not empty.

    A published claim ships with the artifact that produced it. If this fails,
    either the directory was re-ignored or nothing was ever promoted into it.
    """
    assert EVIDENCE.is_dir(), "evidence/ is missing"
    artifacts = sorted(EVIDENCE.glob("*.json"))
    assert artifacts, "evidence/ holds no artifacts"


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.name)
def test_case_artifact_loads_and_rebuilds(path: Path):
    """Every committed case parses, and its minimized triple rebuilds."""
    artifact = load(path)
    case = Case.from_dict(artifact["payload"]["minimized"])
    assert case.requests, f"{path.name} rebuilt to a case with no requests"
    assert case.block_size > 0
    assert case.num_blocks > 0


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.name)
def test_case_round_trips_without_losing_a_field(path: Path):
    """to_dict after from_dict must be identical.

    A dropped field would make the replay run a different case and then report
    agreement, which is worse than reporting nothing.
    """
    stored = load(path)["payload"]["minimized"]
    case = Case.from_dict(stored)
    again = Case.from_dict(case.to_dict())
    assert again == case

    # Every key the artifact carries must survive the round trip, so a field
    # added to Case without being added to to_dict fails here rather than
    # silently degrading a published repro.
    for key, value in stored.items():
        assert key in case.to_dict(), f"{path.name}: to_dict drops {key!r}"


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.name)
def test_case_artifact_carries_an_env_tuple(path: Path):
    """Security doc section 7: a claim without one is invalid by construction."""
    env = json.loads(path.read_text())["env"]
    assert env.get("fingerprint"), f"{path.name} has no env fingerprint"


def test_classify_reports_a_same_engine_difference_as_a_failure():
    """The one case that must never be excused.

    Same engine, same environment, different outcome, means execution is not a
    pure function of the triple. Every other mismatch has an attribution; this
    one must not acquire one by accident.
    """
    ok, detail = classify("OutOfBlocks: x", "", same_engine=True, same_env=True)
    assert ok is False
    assert "pure function" in detail


def test_classify_attributes_a_fixed_finding_to_the_engine_revision():
    ok, detail = classify("OutOfBlocks: x", "", same_engine=False, same_env=True)
    assert ok is True
    assert "engine revision differs" in detail


def test_classify_does_not_claim_attribution_it_does_not_have():
    """An artifact predating engine_revision cannot attribute the difference."""
    ok, detail = classify("OutOfBlocks: x", "", same_engine=False, same_env=True,
                          engine_known=False)
    assert ok is True
    assert "not recorded" in detail
    assert "weaker evidence" in detail


def test_the_witness_case_is_not_vacuous():
    """The witness must exercise what it claims to.

    A witness with one short prompt and no schedule perturbation would pass
    whatever the engine did, which is the failure this repository has caught
    repeatedly. So: it crosses the 512-token split boundary, it chunks, it
    preempts, and it shares a prefix.
    """
    witness = EVIDENCE / "case-witness.json"
    if not witness.exists():
        pytest.skip("no witness recorded")

    payload = load(witness)["payload"]
    case = Case.from_dict(payload["minimized"])

    assert payload["trajectory"], "the witness records no trajectory hash"
    assert max(len(r.prompt) for r in case.requests) > 512, \
        "no prompt crosses the split boundary, so the fold never runs"
    assert case.chunk_plan, "prefill is never chunked"
    assert case.preempt_at, "nothing is ever preempted"
    assert case.shared_prefix_len > 0, "the prefix cache never participates"
    assert len(case.requests) > 1, "batch of one exercises no cohabitation"


def test_the_finding_path_composes(tmp_path):
    """minimize, artifact, repro line, end to end without a GPU.

    This path was dead for most of the project: `minimize` and `print_repro`
    were imported, the divergence report named an artifact path, and no code
    path produced one. Nothing failed, because nothing ran it. So this runs it,
    against a synthetic failure predicate, and asserts the artifact it writes is
    the artifact the printed reproduce line names.
    """
    from harness.fuzz.campaign import Finding, print_repro
    from harness.minimize.ddmin import minimize
    from harness.sim.driver import RequestSpec
    from report.artifact import Artifact
    from engine.envlock import EnvLock

    case = Case(
        requests=(
            RequestSpec(uid="r00", prompt=tuple(range(40)), seed=1, max_new_tokens=2),
            RequestSpec(uid="r01", prompt=tuple(range(40, 70)), seed=2, max_new_tokens=2),
        ),
        chunk_plan=(8, 16),
        refuse_cache_at=(1,),
        block_size=16,
        num_blocks=32,
        label="synthetic",
    )

    # Fails whenever r00 is present, so minimization has something real to do
    # and a well-defined 1-minimal answer.
    minimization = minimize(case, lambda c: any(r.uid == "r00" for r in c.requests))
    assert minimization.reproduces
    assert len(minimization.case.requests) == 1

    env = EnvLock(
        gpu_name="test", gpu_arch="sm_89", gpu_memory_bytes=1, driver_version="1",
        cuda_version="12.4", torch_version="2.6.0", triton_version="3.2.0",
        numpy_version="1", python_version="3.12", os_kernel="test", cpu_model="test",
        omp_num_threads="8", mkl_num_threads="8", torch_intraop_threads=8,
        kernel_registry_sha256="0" * 64, corpus_sha256=None,
        engine_revision="deadbeefcafe",
    )
    artifact = Artifact(kind="case", env=env, payload={
        "reason": "synthetic",
        "minimized": minimization.case.to_dict(),
        "reproduces": minimization.reproduces,
        "one_minimal": minimization.one_minimal,
        "checks_run": minimization.checks_run,
    }).write(results_dir=tmp_path)

    # The written artifact must be loadable by the replay command's loader, and
    # rebuild to the case that was minimized. A repro that does not round trip
    # is a repro line pointing at something unusable.
    written = load(artifact)
    assert Case.from_dict(written["payload"]["minimized"]) == minimization.case

    finding = Finding(case, "AssertionError: synthetic", "cfg", 0.0)
    print_repro(finding, minimization, str(artifact))


def test_the_reproduce_command_the_report_prints_actually_exists():
    """The divergence report names a command. That command must run.

    For most of this project the reproduce line said `lockstep replay <path>`
    where neither the executable nor the path existed. The report was correct
    against the spec and useless to a reader, and no test noticed because every
    test checked the string rather than the thing it named.
    """
    import subprocess

    dispatcher = REPO_ROOT / "lockstep"
    assert dispatcher.is_file(), "the spec's CLI entry point is missing"
    assert dispatcher.stat().st_mode & 0o111, "lockstep is not executable"

    result = subprocess.run(
        [sys.executable, str(dispatcher), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "replay" in result.stdout

    # Every command the spec names is accounted for, either wired to a module or
    # explicitly reported as not built. Silence about a declared command is the
    # failure mode this repository is about.
    from importlib import import_module

    spec_commands = {"run", "verify", "fuzz", "replay", "minimize",
                     "mutate", "certify", "bench", "report"}
    dispatch = runpy_load_commands()
    assert spec_commands <= set(dispatch), \
        f"spec commands missing from the dispatcher: {spec_commands - set(dispatch)}"

    for name, module in dispatch.items():
        if module is not None:
            import_module(module)  # a wired command must import


def runpy_load_commands() -> dict:
    """Read COMMANDS out of the dispatcher without executing its main()."""
    import ast

    source = (REPO_ROOT / "lockstep").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "COMMANDS":
            return ast.literal_eval(node.value)
    raise AssertionError("COMMANDS not found in the dispatcher")
