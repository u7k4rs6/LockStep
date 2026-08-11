"""env.lock carries no identity, and no artifact escapes without one."""

from __future__ import annotations

import getpass
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import envlock  # noqa: E402
from report.artifact import (  # noqa: E402
    SAME_PROCESS, Artifact, harness_env, load, subject_env,
)


def test_captured_lock_leaks_no_identity():
    payload = json.dumps(envlock.capture().to_dict())
    for term in (socket.gethostname(), getpass.getuser(), str(Path.home())):
        if len(term) >= 3:
            assert term not in payload, f"env.lock leaked {term!r}"
    assert "/home/" not in payload
    assert "/Users/" not in payload


def test_emitter_refuses_to_release_a_leaked_field():
    """The guard has to fire, not merely exist."""
    lock = envlock.capture()

    poisoned = replace(lock, gpu_name=f"gpu-{socket.gethostname()}")
    with pytest.raises(envlock.EnvLockLeak, match="identifying information"):
        envlock._assert_scrubbed(poisoned)

    poisoned = replace(lock, cpu_model="built under /home/someone-else/src")
    with pytest.raises(envlock.EnvLockLeak, match="absolute home path"):
        envlock._assert_scrubbed(poisoned)


@pytest.mark.gpu
def test_fingerprint_has_the_shape_the_divergence_report_prints():
    parts = envlock.capture().fingerprint().split(" / ")
    assert len(parts) == 4
    assert parts[0].startswith("sm_")
    assert parts[1].startswith("cu")
    assert parts[2].startswith("triton ")
    assert parts[3].startswith("torch ") and "+" not in parts[3]


def test_registry_digest_tracks_the_registry_file():
    from engine.kernels import registry

    assert envlock.capture().kernel_registry_sha256 == registry.digest()


def test_artifact_round_trips_with_both_environments(tmp_path):
    artifact = Artifact(kind="fidelity", payload={"positions": 3},
                        harness=envlock.capture(), subject=SAME_PROCESS)
    path = artifact.write(results_dir=tmp_path)
    data = load(path)
    assert data["kind"] == "fidelity"
    assert data["payload"]["positions"] == 3
    assert harness_env(data)["fingerprint"] == artifact.harness.fingerprint()
    assert subject_env(data)["kind"] == "same-process"


def test_an_artifact_cannot_be_produced_without_declaring_its_subject():
    """The defect this schema exists to prevent, asserted rather than intended.

    Six committed artifacts carried one environment block that named the
    certifier while the claim beside it was about a server on a different torch.
    Nothing raised, because nothing had to be declared.
    """
    with pytest.raises(TypeError):
        Artifact(kind="fidelity", payload={}, harness=envlock.capture())


def test_a_schema_1_artifact_refuses_to_yield_a_subject_tuple(tmp_path):
    """Reading the harness block as the subject's is the substitution, not a fix."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "schema_version": 1, "kind": "certify", "payload": {},
        "env": {"fingerprint": "sm_89 / cu12.4 / triton 3.2.0 / torch 2.6.0"},
    }))
    data = load(path)
    assert harness_env(data)["fingerprint"].endswith("torch 2.6.0")
    with pytest.raises(KeyError, match="schema 1"):
        subject_env(data)


def test_artifact_sequence_does_not_collide(tmp_path):
    env = envlock.capture()
    names = {
        Artifact(kind="fidelity", payload={}, harness=env, subject=SAME_PROCESS).write(results_dir=tmp_path).name
        for _ in range(3)
    }
    assert len(names) == 3


def test_loading_an_artifact_without_an_env_tuple_is_an_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "fidelity", "payload": {}}))
    with pytest.raises(ValueError, match="no environment tuple"):
        load(path)


def test_registry_get_takes_only_a_name():
    """The structural guarantee that no batch-derived quantity reaches a config."""
    import inspect

    from engine.kernels import registry

    parameters = list(inspect.signature(registry.get).parameters.values())
    assert [p.name for p in parameters] == ["name"], (
        f"registry.get gained parameters {[p.name for p in parameters]}. A config "
        "is looked up by name and by nothing else; see the module docstring."
    )
    # A string, not the type: registry.py uses `from __future__ import annotations`.
    assert parameters[0].annotation in (str, "str")
    assert not any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters
    ), "registry.get must not accept *args or **kwargs"


def test_corpus_sha_defaults_to_null_not_a_placeholder():
    """A placeholder string can be published and read as a real value."""
    assert envlock.capture().corpus_sha256 is None
    assert envlock.capture().to_dict()["corpus_sha256"] is None


def test_schema_1_artifacts_in_evidence_degrade_rather_than_crash():
    """The artifacts backing vllm#51187 are schema 1 and must stay readable.

    subject_env() raising on them is correct: they never recorded a subject.
    What must not happen is a reader crashing on the whole directory, or
    quietly substituting the harness block, because those artifacts are the
    evidence for a filed upstream issue and are deliberately never re-run.
    """
    from report.artifact import SubjectNotRecorded, harness_env, read, subject_env

    evidence = Path(__file__).resolve().parent.parent / "evidence"
    schema_1 = []
    for path in sorted(evidence.glob("certify-*.json")):
        doc = read(path)
        # Every artifact yields a harness tuple under either schema.
        assert harness_env(doc)["fingerprint"]
        if doc.optional("schema_version", 1, because="schema 1 predates it") == 1:
            schema_1.append(path.name)
            with pytest.raises(SubjectNotRecorded):
                subject_env(doc)

    assert schema_1, (
        "no schema 1 certify artifacts found in evidence/. If they were "
        "migrated, this test is obsolete; if they were re-run, the artifacts "
        "backing vllm#51187 were replaced, which is the thing not to do."
    )


def test_the_audit_runs_over_the_committed_evidence_directory():
    """End to end: the report path handles both schemas without crashing."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, str(root / "scripts" / "audit_artifacts.py")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "not recorded (schema 1)" in done.stdout
