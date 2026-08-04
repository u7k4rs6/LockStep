"""env.lock carries no identity, and no artifact escapes without one.

docs/03-security-and-access.md section 4: "Confirm it does not include hostname,
username, or absolute home paths. Strip them in the emitter." Section 7: "Every
artifact embeds env.lock. A claim without an environment tuple is invalid by
construction."

Both are properties of the emitter, so both are tested rather than reviewed.
"""

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
from report.artifact import Artifact, load  # noqa: E402


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

    # Branch one: a term drawn from this machine's identity.
    poisoned = replace(lock, gpu_name=f"gpu-{socket.gethostname()}")
    with pytest.raises(envlock.EnvLockLeak, match="identifying information"):
        envlock._assert_scrubbed(poisoned)

    # Branch two: a home path belonging to nobody here, which the term list
    # cannot know about and the path pattern must catch on its own.
    poisoned = replace(lock, cpu_model="built under /home/someone-else/src")
    with pytest.raises(envlock.EnvLockLeak, match="absolute home path"):
        envlock._assert_scrubbed(poisoned)


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


def test_artifact_round_trips_with_its_env(tmp_path):
    artifact = Artifact(kind="fidelity", payload={"positions": 3}, env=envlock.capture())
    path = artifact.write(results_dir=tmp_path)
    data = load(path)
    assert data["kind"] == "fidelity"
    assert data["payload"]["positions"] == 3
    assert data["env"]["fingerprint"] == artifact.env.fingerprint()


def test_artifact_sequence_does_not_collide(tmp_path):
    env = envlock.capture()
    names = {
        Artifact(kind="fidelity", payload={}, env=env).write(results_dir=tmp_path).name
        for _ in range(3)
    }
    assert len(names) == 3


def test_loading_an_artifact_without_an_env_tuple_is_an_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "fidelity", "payload": {}}))
    with pytest.raises(ValueError, match="no environment tuple"):
        load(path)


def test_registry_get_takes_only_a_name():
    """The structural guarantee that no batch-derived quantity reaches a config.

    `registry.get` accepting a second parameter would be the exact door through
    which a shape could arrive, and nothing else in the codebase would notice.
    This asserts the door stays shut rather than relying on nobody opening it.
    """
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
