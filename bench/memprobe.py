"""Host memory instrumentation for the fp64 reference pass."""

from __future__ import annotations

import gc
import resource
from dataclasses import dataclass

import torch

MIB = 2**20


def peak_rss_bytes() -> int:
    """High-water RSS for this process. ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def current_rss_bytes() -> int:
    """Resident set right now, read from /proc rather than the rusage peak."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * resource.getpagesize()
    except (OSError, IndexError, ValueError):
        return 0


def rss_split() -> tuple[int, int]:
    """(anonymous, file-backed) resident bytes."""
    anon = file = 0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                anon = int(line.split()[1]) * 1024
            elif line.startswith("RssFile:"):
                file = int(line.split()[1]) * 1024
    except OSError:
        pass
    return anon, file


def available_bytes() -> int:
    """MemAvailable from /proc/meminfo: what can be had without swapping."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


@dataclass(frozen=True)
class TensorGroup:
    device: str
    dtype: str
    count: int
    bytes: int
    largest_shape: tuple


def live_tensor_breakdown(top: int = 8) -> list[TensorGroup]:
    """Largest live torch tensors, grouped by (device, dtype)."""
    gc.collect()
    seen: set[int] = set()
    groups: dict[tuple[str, str], list] = {}

    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            storage = obj.untyped_storage()
            key = storage.data_ptr()
            if key in seen or key == 0:
                continue
            seen.add(key)
            size = storage.nbytes()
        except (RuntimeError, AttributeError, NotImplementedError):
            continue

        bucket = groups.setdefault((str(obj.device), str(obj.dtype)), [0, 0, 0, ()])
        bucket[0] += 1
        bucket[1] += size
        if size > bucket[2]:
            bucket[2] = size
            bucket[3] = tuple(obj.shape)

    rows = [
        TensorGroup(device=device, dtype=dtype, count=count, bytes=total, largest_shape=shape)
        for (device, dtype), (count, total, _, shape) in groups.items()
    ]
    rows.sort(key=lambda row: row.bytes, reverse=True)
    return rows[:top]


def format_breakdown(label: str, rows: list[TensorGroup]) -> str:
    lines = [f"  {label}"]
    anon, file_backed = rss_split()
    lines.append(f"    rss {current_rss_bytes() / MIB:8.1f} MiB "
                 f"(anon {anon / MIB:7.1f}, file {file_backed / MIB:7.1f})   "
                 f"peak {peak_rss_bytes() / MIB:8.1f} MiB   "
                 f"avail {available_bytes() / MIB:8.1f} MiB")
    for row in rows:
        if row.bytes < 8 * MIB:
            continue
        lines.append(
            f"    {row.device:<6} {row.dtype:<16} {row.count:>5} tensors "
            f"{row.bytes / MIB:>9.1f} MiB   largest {row.largest_shape}"
        )
    return "\n".join(lines)


class MemoryCeilingExceeded(RuntimeError):
    """The pass would need more host memory than it is allowed."""


def require_headroom(projected_bytes: int, ceiling_bytes: int, what: str) -> None:
    """Fail fast with the projection rather than thrashing into swap."""
    if projected_bytes > ceiling_bytes:
        raise MemoryCeilingExceeded(
            f"{what} projects {projected_bytes / MIB:.0f} MiB, over the "
            f"{ceiling_bytes / MIB:.0f} MiB ceiling. Lower the chunk size, or raise "
            f"--memory-ceiling-mib deliberately."
        )
    available = available_bytes()
    if available and projected_bytes > available:
        raise MemoryCeilingExceeded(
            f"{what} projects {projected_bytes / MIB:.0f} MiB but only "
            f"{available / MIB:.0f} MiB is available; this would swap."
        )
