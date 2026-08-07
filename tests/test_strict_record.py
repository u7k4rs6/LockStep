"""A wrong field name must raise, not return a believable number."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from certify import subject  # noqa: E402
from report.artifact import Record  # noqa: E402

FAULT = {"fault": "refcount_decrement_missing_on_free", "verdict": "killed",
         "mutation_took_effect": 1}


def test_the_exact_bug_this_exists_for():
    """`killed` is not a field; the field is `verdict`. It read 0 of 10."""
    record = Record({"seeded_faults": [FAULT] * 10}, "fuzz.payload")
    assert sum(1 for f in record["seeded_faults"] if f["verdict"] == "killed") == 10
    with pytest.raises(KeyError, match="no field 'killed'"):
        record["seeded_faults"][0]["killed"]


def test_get_is_unavailable_because_get_is_what_produced_the_zero():
    record = Record(dict(FAULT), "fault")
    with pytest.raises(TypeError, match="not available"):
        record.get("killed", 0)


def test_optional_needs_a_stated_reason():
    record = Record(dict(FAULT), "fault")
    assert record.optional("subject_env", None, because="added later") is None
    assert record.optional("verdict", None, because="added later") == "killed"
    with pytest.raises(ValueError):
        record.optional("subject_env", None, because="")


def test_the_error_names_the_fields_that_are_there():
    record = Record(dict(FAULT), "fault")
    with pytest.raises(KeyError) as caught:
        record["verdcit"]
    assert "verdict" in str(caught.value)


def test_nested_records_stay_strict():
    record = Record({"payload": {"results": [{"clean": True}]}}, "artifact")
    assert record["payload"]["results"][0]["clean"] is True
    with pytest.raises(KeyError):
        record["payload"]["results"][0]["cleen"]


def test_subject_version_comes_from_the_servers_own_output(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO [api_utils.py:345]  version 0.26.0\n"
        "INFO [api_utils.py:273] non-default args: {'port': 30011, 'dtype': 'float16', "
        "'block_size': 16, 'enforce_eager': True}\n"
        "INFO [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: ...\n"
    )
    parsed = subject.from_startup_log(log)
    assert parsed["parsed"] and parsed["engine_version"] == "0.26.0"
    assert parsed["non_default_args"]["block_size"] == 16
    assert parsed["non_default_args"]["enforce_eager"] is True


def test_a_missing_log_is_reported_rather_than_guessed(tmp_path):
    parsed = subject.from_startup_log(tmp_path / "absent.log")
    assert parsed["parsed"] is False


def test_absolute_paths_are_scrubbed_before_the_artifact_is_committed():
    scrubbed = subject.scrub({"model": f"{REPO_ROOT}/weights/Qwen3-0.6B",
                              "home": f"{Path.home()}/lockstep-extenv"})
    assert scrubbed["model"] == "<repo>/weights/Qwen3-0.6B"
    assert str(Path.home()) not in scrubbed["home"]
