"""The environment of the engine under test, as opposed to the certifier's.

A certify artifact's `env.lock` records the process making the requests. The
engine being certified runs in its own virtual environment on its own torch, so
for six committed artifacts the tuple beside the claim scoped the observer and
not the observed. The only place the subject's tuple was written down was one
hand-maintained provenance file, for one finding.

Nothing here is inferred. The version and configuration are read back from what
the server itself printed at startup, and the interpreter tuple is read from the
interpreter the certifier launched.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BANNER_VERSION = re.compile(r"\bversion (\d+\.\d+\.\d+\S*)")
ENGINE_VERSION = re.compile(r"LLM engine \(v([^)]+)\)")
NON_DEFAULT_ARGS = re.compile(r"non-default args: (\{.*?\})\s*$")

PROBE = (
    "import json, sys, torch;"
    "print(json.dumps({"
    "'torch_version': torch.__version__,"
    "'cuda_version': torch.version.cuda,"
    "'python_version': '.'.join(str(v) for v in sys.version_info[:3]),"
    "'triton_version': __import__('triton').__version__"
    "  if __import__('importlib.util', fromlist=['x']).find_spec('triton') else None,"
    "}))"
)


def from_startup_log(log: Path) -> dict:
    """vLLM's own banner and argument line, parsed rather than assumed."""
    if not log.is_file():
        return {"parsed": False, "reason": "no startup log"}

    version, args = None, None
    for line in log.read_text(errors="replace").splitlines():
        if version is None:
            match = ENGINE_VERSION.search(line) or BANNER_VERSION.search(line)
            if match:
                version = match.group(1)
        if args is None:
            match = NON_DEFAULT_ARGS.search(line)
            if match:
                try:
                    args = ast.literal_eval(match.group(1))
                except (ValueError, SyntaxError):
                    args = None
        if version and args:
            break

    return {
        "parsed": version is not None,
        "engine_version": version,
        "non_default_args": args,
        "source": "the server's own startup output",
    }


def interpreter(python: Path) -> dict:
    """torch, triton and CUDA as the subject's environment reports them."""
    try:
        done = subprocess.run([str(python), "-c", PROBE],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return {"probed": False, "reason": str(error)[:200]}
    if done.returncode != 0:
        return {"probed": False, "reason": done.stderr.strip()[-200:]}
    return {"probed": True} | json.loads(done.stdout)


def scrub(value):
    """Absolute paths become placeholders. The artifact is committed publicly."""
    if isinstance(value, str):
        return (value.replace(str(REPO_ROOT), "<repo>")
                     .replace(str(Path.home()), "<home>"))
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def unprobed(engine_name: str, reason: str) -> dict:
    """A subject this runner cannot interrogate.

    `certify/run.py` attaches to a server someone else started, so it has no
    interpreter to probe and no startup log to parse. That is a real limitation
    and it is written down as one. The alternative on offer was to fall back to
    the harness tuple, which is exactly the substitution schema 2 exists to stop.
    """
    if not reason:
        raise ValueError("an unprobed subject must say why it could not be probed")
    return {
        "engine": engine_name,
        "engine_version": None,
        "non_default_args": None,
        "torch_version": None,
        "triton_version": None,
        "cuda_version": None,
        "python_version": None,
        "startup_parsed": False,
        "interpreter_probed": False,
        "reason": reason,
        "note": "the subject's environment was NOT recorded. Read no tuple into "
                "this artifact: the harness block beside it describes a "
                "different process.",
    }


def capture(python: Path, log: Path, engine_name: str) -> dict:
    """Everything recorded about the subject, for the artifact."""
    startup = scrub(from_startup_log(log))
    probe = interpreter(python)
    return {
        "engine": engine_name,
        "engine_version": startup["engine_version"],
        "non_default_args": startup["non_default_args"],
        "torch_version": probe.get("torch_version"),
        "triton_version": probe.get("triton_version"),
        "cuda_version": probe.get("cuda_version"),
        "python_version": probe.get("python_version"),
        "interpreter": scrub(str(python)),
        "note": "the environment of the engine under test. The artifact's own "
                "env.lock describes the certifier, which is a different process "
                "in a different virtual environment on a different torch.",
        "startup_parsed": startup["parsed"],
        "interpreter_probed": probe["probed"],
    }
