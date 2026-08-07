"""Fetch Qwen3-0.6B at the revision pinned in weights.lock, and verify it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = REPO_ROOT / "weights.lock"
DEFAULT_DEST = REPO_ROOT / "weights" / "Qwen3-0.6B"

CHUNK = 1 << 20
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_sha: str, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return sha256_of(path) == expected_sha


def download(url: str, dest: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        hint = " (set HF_TOKEN if this revision is gated)" if exc.code in (401, 403) else ""
        raise SystemExit(f"download failed: {url} -> HTTP {exc.code}{hint}") from exc
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files against weights.lock and download nothing.",
    )
    args = parser.parse_args()

    lock = json.loads(LOCKFILE.read_text())
    model_id, revision = lock["model_id"], lock["revision"]

    bad = [name for name in lock["files"] if name.endswith(FORBIDDEN_SUFFIXES)]
    if bad:
        raise SystemExit(
            f"weights.lock names pickle-format files, which are never loaded: {bad}"
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    failures: list[str] = []

    for name, meta in lock["files"].items():
        path = args.dest / name
        if verify(path, meta["sha256"], meta["size"]):
            print(f"ok       {name}")
            continue
        if args.verify_only:
            state = "missing" if not path.is_file() else "CHECKSUM MISMATCH"
            print(f"{state:8} {name}")
            failures.append(name)
            continue

        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{name}"
        print(f"fetch    {name} ({meta['size'] / 1e6:.1f} MB)")
        download(url, path, token)
        if not verify(path, meta["sha256"], meta["size"]):
            actual = sha256_of(path) if path.is_file() else "<absent>"
            print(
                f"MISMATCH {name}\n"
                f"         expected sha256:{meta['sha256']}\n"
                f"         observed sha256:{actual}",
                file=sys.stderr,
            )
            failures.append(name)
        else:
            print(f"ok       {name}")

    if failures:
        print(f"\n{len(failures)}/{len(lock['files'])} files failed verification", file=sys.stderr)
        return 1

    print(f"\n{len(lock['files'])}/{len(lock['files'])} files verified at {revision[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
