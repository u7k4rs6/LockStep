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

# Everything a reader would need to reconstruct the subject: the engine and its
# version tuple, the three libraries whose kernels are in play, and the driver.
# The driver is read in the subject's process rather than copied from the
# harness: they happen to be the same machine here, and writing down a value
# because it "must" match is how the harness tuple ended up beside a claim about
# a different environment in the first place.
PROBE = (
    "import json, sys, torch;"
    "_spec=__import__('importlib.util', fromlist=['x']).find_spec;"
    "_ver=__import__('importlib.metadata', fromlist=['x']).version;"
    "_opt=lambda name: (_ver(name) if _spec(name.split('[')[0]) else None);"
    "_read=lambda p: (open(p).read() if __import__('os').path.exists(p) else '');"
    "_drv=__import__('re').search(r'NVRM version:.*?\\b(\\d+\\.\\d+(?:\\.\\d+)?)\\b',"
    "  _read('/proc/driver/nvidia/version'));"
    "import vllm;"
    "import vllm.model_executor.layers.batch_invariant as _bi;"
    "_sha=lambda f: (__import__('hashlib').sha256(open(f,'rb').read()).hexdigest()"
    "  if __import__('os').path.exists(f) else None);"
    "print(json.dumps({"
    "'torch_version': torch.__version__,"
    "'cuda_version': torch.version.cuda,"
    "'python_version': '.'.join(str(v) for v in sys.version_info[:3]),"
    "'triton_version': __import__('triton').__version__ if _spec('triton') else None,"
    "'flashinfer_version': __import__('flashinfer').__version__"
    "  if _spec('flashinfer') else None,"
    "'engine_package_version': vllm.__version__,"
    "'engine_version_tuple': list(vllm.version.__version_tuple__),"
    "'driver_version': _drv.group(1) if _drv else None,"
    # These runs deliberately patch the subject to record num_tokens at every
    # RMSNorm launch. An artifact that did not say so would describe a stock
    # 0.26.0, which is not what ran. The digest is read in the subject's
    # process, from the module the subject actually imported.
    "'batch_invariant_file': _bi.__file__,"
    "'batch_invariant_sha256': _sha(_bi.__file__),"
    "'batch_invariant_stock_sha256': _sha(_bi.__file__ + '.lockstep-orig'),"
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
        "flashinfer_version": probe.get("flashinfer_version"),
        "cuda_version": probe.get("cuda_version"),
        "driver_version": probe.get("driver_version"),
        "python_version": probe.get("python_version"),
        "engine_package_version": probe.get("engine_package_version"),
        "engine_version_tuple": probe.get("engine_version_tuple"),
        "batch_invariant_file": scrub(probe.get("batch_invariant_file")),
        "batch_invariant_sha256": probe.get("batch_invariant_sha256"),
        "batch_invariant_stock_sha256": probe.get("batch_invariant_stock_sha256"),
        "batch_invariant_is_stock": (
            probe.get("batch_invariant_stock_sha256") is None
            or probe.get("batch_invariant_sha256")
            == probe.get("batch_invariant_stock_sha256")
        ),
        "engine_commit": None,
        "engine_commit_note": "vLLM 0.26.0 is an installed wheel, not a git "
                              "checkout: it ships no commit and none can be "
                              "read back. The released version is the finest "
                              "identifier available for this subject, and "
                              "saying so is more useful than a plausible SHA.",
        "interpreter": scrub(str(python)),
        "note": "the environment of the engine under test. The artifact's own "
                "env.lock describes the certifier, which is a different process "
                "in a different virtual environment on a different torch.",
        "startup_parsed": startup["parsed"],
        "interpreter_probed": probe["probed"],
    }
