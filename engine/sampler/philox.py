"""Counter-based sampling, keyed on (seed, uid, position) and nothing else."""

from __future__ import annotations

import hashlib

import torch

PHILOX_ROUNDS = 10
MULT_A = 0xD2511F53
MULT_B = 0xCD9E8D57
WEYL_0 = 0x9E3779B9
WEYL_1 = 0xBB67AE85
UINT32 = 0xFFFFFFFF

TWO_POW_NEG32 = 2.0**-32


def uid_key(uid: str) -> int:
    """A stable 64-bit key for a request uid."""
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big")


def _mulhilo(a: int, b: int) -> tuple[int, int]:
    product = (a * b) & 0xFFFFFFFFFFFFFFFF
    return (product >> 32) & UINT32, product & UINT32


def philox_4x32(counter: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, ...]:
    """Ten rounds of Philox-4x32. Pure function of (counter, key)."""
    c0, c1, c2, c3 = (x & UINT32 for x in counter)
    k0, k1 = (x & UINT32 for x in key)

    for round_index in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo(MULT_A, c0)
        hi1, lo1 = _mulhilo(MULT_B, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & UINT32,
            lo1,
            (hi0 ^ c3 ^ k1) & UINT32,
            lo0,
        )
        if round_index < PHILOX_ROUNDS - 1:
            k0 = (k0 + WEYL_0) & UINT32
            k1 = (k1 + WEYL_1) & UINT32

    return c0, c1, c2, c3


# Keyed on (seed, uid, position) and nothing else. A global step counter here
# would make a request's draw depend on how many others were resident, which is
# cross-request coupling that no output comparison catches until two runs happen
# to differ in batch composition.
def uniform(seed: int, uid: str, position: int, index: int = 0) -> float:
    """A draw in [0, 1) for (seed, uid, position)."""
    key64 = uid_key(uid)
    words = philox_4x32(
        counter=(position & UINT32, (position >> 32) & UINT32, index & UINT32, 0),
        key=((seed ^ key64) & UINT32, ((seed ^ key64) >> 32) & UINT32),
    )
    return words[0] * TWO_POW_NEG32


def greedy(logits: torch.Tensor) -> int:
    """Argmax with ties broken by the lowest token ID."""
    values = logits.to(torch.float32)
    best = values.max()
    return int(torch.nonzero(values == best, as_tuple=False)[0].item())


def top_p(
    logits: torch.Tensor,
    seed: int,
    uid: str,
    position: int,
    p: float = 1.0,
    temperature: float = 1.0,
) -> int:
    """Nucleus sampling with a stable order and a counter-based draw."""
    if temperature <= 0:
        return greedy(logits)

    values = (logits.to(torch.float32) / temperature).to(torch.float64).cpu()
    # The named exception to the torch-reduction gate in scripts/static_checks.py:
    # CPU, fp64, one row, outside any batched path.
    probs = torch.softmax(values, dim=-1)

    order = sorted(range(probs.numel()), key=lambda i: (-float(probs[i]), i))
    ordered = torch.tensor([float(probs[i]) for i in order], dtype=torch.float64)
    cumulative = torch.cumsum(ordered, dim=0)

    keep = int(torch.searchsorted(cumulative, torch.tensor(p, dtype=torch.float64)).item()) + 1
    keep = min(keep, len(order))

    head = ordered[:keep]
    draw = uniform(seed, uid, position) * float(head.sum())
    chosen = int(torch.searchsorted(torch.cumsum(head, dim=0), torch.tensor(draw)).item())
    return order[min(chosen, keep - 1)]
