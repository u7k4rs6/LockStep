"""Pre-commit secret scanner."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys

PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY")),
]

GENERIC_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b (?: api[_-]?key | secret | passwd | password | token | credential
         | auth[_-]?token | access[_-]?key | private[_-]?key )
    \b \s* [:=] \s* ['"]  (?P<value> [^'"\n]{16,}) ['"]
    """
)

PLACEHOLDER = re.compile(
    r"(?i)^(?:x{4,}|\.{3,}|<[^>]+>|\$\{[^}]+\}|your[_-]|example|placeholder|"
    r"changeme|redacted|dummy|fake|test[_-]?only|none|null)"
)

SKIP_PATH = re.compile(r"(?i)\.(?:png|jpg|jpeg|gif|pdf|safetensors|bin|pt|npz|npy|lock)$")


def shannon_entropy(value: str) -> float:
    """Bits per character. Real keys sit above ~3.5; English prose below ~3.2."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def staged_content(path: str) -> str | None:
    out = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, check=False
    )
    if out.returncode != 0:
        return None
    try:
        return out.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_text(path: str, text: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 4096:
            continue
        for label, pattern in PROVIDER_PATTERNS:
            if pattern.search(line):
                findings.append(f"{path}:{lineno}: {label}")
        match = GENERIC_ASSIGNMENT.search(line)
        if match:
            value = match.group("value")
            if not PLACEHOLDER.match(value) and shannon_entropy(value) >= 3.5:
                findings.append(
                    f"{path}:{lineno}: high-entropy assignment to a secret-named field"
                )
    return findings


def run_gitleaks() -> int:
    print("secret-scan: using gitleaks", file=sys.stderr)
    return subprocess.run(
        ["gitleaks", "protect", "--staged", "--redact", "--no-banner"], check=False
    ).returncode


def main() -> int:
    if shutil.which("gitleaks"):
        return run_gitleaks()

    findings: list[str] = []
    for path in staged_files():
        if path == ".env" or path.startswith(".env."):
            if path != ".env.example":
                findings.append(f"{path}: .env files must never be committed")
                continue
        if SKIP_PATH.search(path):
            continue
        text = staged_content(path)
        if text is None:
            continue
        findings.extend(scan_text(path, text))

    if findings:
        print("secret-scan: blocking commit\n", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(
            "\nRemove the value, or if this is a false positive, commit with"
            " --no-verify and say why in the message.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
