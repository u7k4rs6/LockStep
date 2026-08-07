"""env.lock: the environment tuple every published claim is scoped to."""

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
    gpu_arch: str
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
    engine_revision: str = UNKNOWN

    def fingerprint(self) -> str:
        """The short form that terminates every bitwise claim line."""
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


def _engine_revision() -> str:
    """The commit this engine is at, marked dirty if the tree has changes."""
    import subprocess

    try:
        root = Path(__file__).resolve().parent.parent
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if head.returncode != 0:
            return UNKNOWN
        sha = head.stdout.strip()[:12]
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN


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
    """Snapshot the current environment."""
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
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        torch_intraop_threads=torch.get_num_threads(),
        kernel_registry_sha256=_sha256_file(KERNEL_REGISTRY),
        corpus_sha256=corpus_sha256,
        engine_revision=_engine_revision(),
    )
    _assert_scrubbed(lock)
    return lock


def main() -> int:
    print(json.dumps(capture().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
