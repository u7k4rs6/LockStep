"""Counter-based sampling, keyed on (seed, uid, position) and nothing else.

docs/02-technical-architecture.md invariant I4: "`r`'s sampled tokens are a
function only of `(k, r.uid, position)` and `r`'s logits. Adding or removing any
other request never changes `r`'s draws."

Section 5 names the failure this avoids: "RNG keyed on global step. A classic
source of cross-request coupling." A global step counter makes a request's draw
depend on how many other requests were resident, which is cross-request coupling
wearing a plausible disguise, and it is invisible until someone runs the same
prompt in a different batch.

Philox is counter-based, so there is no stream to advance and no state to carry.
The draw for (seed, uid, position) is computed from those three values directly,
which is why MR7 can delete a request and expect its neighbours' tokens to be
bitwise unchanged: there was never a shared stream for the deletion to shift.

The implementation is the Philox-4x32-10 bijection, written out here rather than
taken from `torch.Generator`, because a torch generator is stateful and advancing
it is exactly the coupling being avoided. Ten rounds is the standard count from
Salmon et al. (SC 2011).

Tie-breaking is by lowest token ID, in both argmax and top-p, per the PRD's
must-have row. `torch.argmax` already returns the first maximal index; the tests
assert that rather than trusting it, since it is a documented-but-incidental
property that a future release could reasonably change.
"""

from __future__ import annotations

import hashlib

import torch

PHILOX_ROUNDS = 10
MULT_A = 0xD2511F53
MULT_B = 0xCD9E8D57
WEYL_0 = 0x9E3779B9
WEYL_1 = 0xBB67AE85
UINT32 = 0xFFFFFFFF

# 2^-32, for mapping a uint32 to [0, 1). Exact in fp64.
TWO_POW_NEG32 = 2.0**-32


def uid_key(uid: str) -> int:
    """A stable 64-bit key for a request uid.

    sha256 rather than Python's `hash`, which is salted per process by
    PYTHONHASHSEED and would make the same workload draw different tokens on the
    next run. That would be a determinism bug whose cause is invisible in the
    engine's own code.
    """
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


def uniform(seed: int, uid: str, position: int, index: int = 0) -> float:
    """A draw in [0, 1) for (seed, uid, position).

    `index` lets one position draw more than one number without introducing a
    counter that would have to be carried between calls.
    """
    key64 = uid_key(uid)
    words = philox_4x32(
        counter=(position & UINT32, (position >> 32) & UINT32, index & UINT32, 0),
        key=((seed ^ key64) & UINT32, ((seed ^ key64) >> 32) & UINT32),
    )
    return words[0] * TWO_POW_NEG32


def greedy(logits: torch.Tensor) -> int:
    """Argmax with ties broken by the lowest token ID.

    Computed in fp32 from the fp16 logits so the comparison is over the same
    values the fidelity report measures.
    """
    values = logits.to(torch.float32)
    best = values.max()
    # nonzero() is ascending, so the first tied index is the lowest token ID.
    # This is the tie-break rule stated explicitly rather than inherited from
    # whatever argmax happens to do.
    return int(torch.nonzero(values == best, as_tuple=False)[0].item())


def top_p(
    logits: torch.Tensor,
    seed: int,
    uid: str,
    position: int,
    p: float = 1.0,
    temperature: float = 1.0,
) -> int:
    """Nucleus sampling with a stable order and a counter-based draw.

    Sorting is by (descending probability, ascending token ID). `torch.sort` is
    not guaranteed stable on CUDA, so the order is made total by construction
    rather than assumed: ties are separated by token ID before the cut, so the
    nucleus membership of a tied pair never depends on which the sort happened to
    place first.
    """
    if temperature <= 0:
        return greedy(logits)

    values = (logits.to(torch.float32) / temperature).to(torch.float64).cpu()
    probs = torch.softmax(values, dim=-1)

    order = sorted(range(probs.numel()), key=lambda i: (-float(probs[i]), i))
    ordered = torch.tensor([float(probs[i]) for i in order], dtype=torch.float64)
    cumulative = torch.cumsum(ordered, dim=0)

    # Smallest prefix whose mass reaches p; always at least one token.
    keep = int(torch.searchsorted(cumulative, torch.tensor(p, dtype=torch.float64)).item()) + 1
    keep = min(keep, len(order))

    head = ordered[:keep]
    draw = uniform(seed, uid, position) * float(head.sum())
    chosen = int(torch.searchsorted(torch.cumsum(head, dim=0), torch.tensor(draw)).item())
    return order[min(chosen, keep - 1)]
