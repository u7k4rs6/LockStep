"""env.lock: the environment tuple every published claim is scoped to.

docs/02-technical-architecture.md section 1: "No claim in this project is
portable across this tuple." docs/03-security-and-access.md section 7: "A claim
without an environment tuple is invalid by construction."

Section 4 of the security doc requires that the emitted tuple carry no hostname,
username, or absolute home path. That is enforced here by `_assert_scrubbed`
rather than left to review, because the emitter runs on the developer's machine
and review is the thing that fails first.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_REGISTRY = REPO_ROOT / "engine" / "kernels" / "registry.py"

UNKNOWN = "unknown"


class EnvLockLeak(RuntimeError):
    """The environment tuple contained identifying information."""


@dataclass(frozen=True)
class EnvLock:
    """A pinned environment. Serialized into every result artifact."""

    gpu_name: str
    gpu_arch: str  # "sm_89"
    gpu_memory_bytes: int
    driver_version: str
    cuda_version: str
    torch_version: str
    triton_version: str
    numpy_version: str
    python_version: str
    os_kernel: str
    cpu_model: str
    omp_num_threads: str | None
    mkl_num_threads: str | None
    torch_intraop_threads: int
    kernel_registry_sha256: str
    corpus_sha256: str | None

    def fingerprint(self) -> str:
        """The short form that terminates every bitwise claim line.

        Frontend spec 1.2: never print a bitwise claim without the environment
        tag. Frontend spec 1.3 shows the exact shape:
        `sm_89 / cu12.4 / triton 3.x / torch 2.x`.

        The torch local version suffix ("+cu124") is dropped here because the
        CUDA version is already the second field; the full string stays in the
        artifact, which is what a mismatch is checked against.
        """
        torch_version = self.torch_version.split("+")[0]
        return (
            f"{self.gpu_arch} / cu{self.cuda_version} / "
            f"triton {self.triton_version} / torch {torch_version}"
        )

    def digest(self) -> str:
        """Stable hash of the whole tuple, for artifact-to-artifact comparison."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        d = dict(asdict(self))
        d["fingerprint"] = self.fingerprint()
        d["digest"] = self.digest()
        return d


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return UNKNOWN
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str:
    """Read from /proc/cpuinfo. Carries no user-identifying fields."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or UNKNOWN


def _driver_version() -> str:
    """NVIDIA driver version from /proc, avoiding a subprocess and nvidia-smi."""
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        return UNKNOWN
    # "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.84  Release
    # Build ...". The wording varies between the proprietary and open modules, so
    # match the first dotted number on the NVRM line rather than a fixed phrase.
    for line in text.splitlines():
        if line.startswith("NVRM version:"):
            match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", line)
            if match:
                return match.group(1)
    return UNKNOWN


def _gpu_fields() -> tuple[str, str, int]:
    import torch

    if not torch.cuda.is_available():
        return UNKNOWN, UNKNOWN, 0
    props = torch.cuda.get_device_properties(0)
    return props.name, f"sm_{props.major}{props.minor}", props.total_memory


def _scrub_terms() -> list[str]:
    """Strings that must not appear anywhere in the serialized tuple."""
    terms = []
    for value in (socket.gethostname(), socket.getfqdn(), str(Path.home())):
        if value and value not in ("/", "."):
            terms.append(value)
    try:
        terms.append(getpass.getuser())
    except (OSError, KeyError):
        pass
    for var in ("USER", "LOGNAME", "HOSTNAME"):
        value = os.environ.get(var)
        if value:
            terms.append(value)
    # Deduplicate, drop anything too short to be a meaningful match.
    return sorted({t for t in terms if len(t) >= 3})


def _assert_scrubbed(lock: EnvLock) -> None:
    payload = json.dumps(asdict(lock))
    leaked = [term for term in _scrub_terms() if term in payload]
    if leaked:
        raise EnvLockLeak(
            "env.lock contains identifying information and was not emitted: "
            + ", ".join(repr(t) for t in leaked)
        )
    if re.search(r"/home/[^/\"]+", payload) or re.search(r"/Users/[^/\"]+", payload):
        raise EnvLockLeak("env.lock contains an absolute home path and was not emitted")


def capture(corpus_sha256: str | None = None) -> EnvLock:
    """Snapshot the current environment.

    `corpus_sha256` ties a fidelity number to the prompt corpus that produced it.
    Callers that do not touch a corpus leave it None rather than passing a
    human-readable placeholder: a placeholder string can be published inside an
    artifact and read as a real value, whereas null cannot be mistaken for one.
    """
    import numpy
    import torch

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = UNKNOWN

    gpu_name, gpu_arch, gpu_memory = _gpu_fields()

    lock = EnvLock(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_memory_bytes=gpu_memory,
        driver_version=_driver_version(),
        cuda_version=torch.version.cuda or UNKNOWN,
        torch_version=torch.__version__,
        triton_version=triton_version,
        numpy_version=numpy.__version__,
        python_version=platform.python_version(),
        os_kernel=f"{platform.system()} {platform.release()} {platform.machine()}",
        cpu_model=_cpu_model(),
        # All three, because OMP_NUM_THREADS and MKL_NUM_THREADS are
        # backend-specific environment variables while torch's intraop pool is
        # what actually governs reduction order whichever backend is underneath.
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        torch_intraop_threads=torch.get_num_threads(),
        kernel_registry_sha256=_sha256_file(KERNEL_REGISTRY),
        corpus_sha256=corpus_sha256,
    )
    _assert_scrubbed(lock)
    return lock


def main() -> int:
    print(json.dumps(capture().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
