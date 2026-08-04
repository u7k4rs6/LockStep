"""The trajectory hash: one digest over everything the engine did.

docs/02-technical-architecture.md invariant I3: "Replay determinism. Identical
`(W, sigma, seeds)` twice yields an identical trajectory hash over all engine
state." MR6 is the relation that tests it.

"All engine state" is the load-bearing phrase. A hash over emitted tokens alone
would be satisfied by an engine that reached the same tokens through a different
allocator state, which is precisely the class of bug the mutation operators in
section 10.1 inject: a refcount incremented twice on fork, a block freed while
still referenced, a copy-on-write that copies without bumping. Those change the
allocator ledger long before they change a token, and often never change a token
at all. Section 10.2 makes that explicit: "Output bits alone leave too many
mutants equivalent."

So the hash covers, per step:

  * the packed work list: which uid contributed which tokens at which position,
    in order, which is where a scheduler reordering shows up
  * every emitted token
  * the raw fp16 logit bytes for every row, which is the bitwise part of the
    "bit-identical" definition in section 2
  * the allocator state digest: refcounts, free pool, per-sequence block tables,
    and the allocate/free/COW/fork counters

Absorbed in step order into a single running sha256, so the digest is a function
of the whole trajectory rather than of an unordered set of observations. Feeding
each field with a length prefix keeps two different structures from colliding
into one byte stream.
"""

from __future__ import annotations

import hashlib

import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes of a tensor, in its own dtype.

    Not `.tolist()` or a float cast: architecture doc section 2 defines
    bit-identical as "identical raw fp16 logit bytes", so the hash has to see the
    bytes rather than a rounded decimal view of them.
    """
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class TrajectoryHash:
    """A running sha256 over the engine's whole observable trajectory."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.steps = 0
        self.per_step: list[str] = []

    def _absorb(self, label: str, payload: bytes) -> None:
        self._digest.update(label.encode("utf-8"))
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def observe_step(
        self,
        step: int,
        work: list[tuple[str, list[int], int]],
        emitted: list[tuple[str, int]],
        logits: dict[str, torch.Tensor],
        pool_state: tuple,
    ) -> None:
        self._absorb("step", str(step).encode())
        self._absorb("work", repr(work).encode())
        self._absorb("emitted", repr(emitted).encode())
        for uid in sorted(logits):
            self._absorb(f"logits:{uid}", _tensor_bytes(logits[uid]))
        self._absorb("pool", repr(pool_state).encode())

        self.steps += 1
        self.per_step.append(self._digest.hexdigest())

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def first_divergence(self, other: "TrajectoryHash") -> int | None:
        """The first step at which two trajectories stopped agreeing.

        What the divergence report's `position` field is filled from when MR6
        fails, so that a replay mismatch names a step instead of just failing.
        """
        for index, (mine, theirs) in enumerate(zip(self.per_step, other.per_step)):
            if mine != theirs:
                return index
        if len(self.per_step) != len(other.per_step):
            return min(len(self.per_step), len(other.per_step))
        return None
