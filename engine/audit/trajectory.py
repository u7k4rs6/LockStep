"""The trajectory hash: one digest over everything the engine did."""

from __future__ import annotations

import hashlib

import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes of a tensor, in its own dtype."""
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
        """The first step at which two trajectories stopped agreeing."""
        for index, (mine, theirs) in enumerate(zip(self.per_step, other.per_step)):
            if mine != theirs:
                return index
        if len(self.per_step) != len(other.per_step):
            return min(len(self.per_step), len(other.per_step))
        return None
