"""Paged KV: block table, refcounted blocks, copy-on-write fork, allocator."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch

from engine.audit.counters import Counters
from engine.kernels import registry

DEFAULT_BLOCK_SIZE = 16

SUPPORTED_BLOCK_SIZES = (8, 16, 32, 64, 128)

for _size in SUPPORTED_BLOCK_SIZES:
    assert registry.KV_TILE % _size == 0, (
        f"KV tile {registry.KV_TILE} is not a whole number of {_size}-token blocks"
    )

POISON = float("nan")


class OutOfBlocks(RuntimeError):
    """The pool is exhausted and the policy did not free anything."""


class AuditFailure(AssertionError):
    """An internal KV invariant broke. Always checked in debug builds."""


@dataclass
class Sequence:
    """One request's logical-to-physical block mapping."""

    uid: str
    block_ids: list[int] = field(default_factory=list)
    length: int = 0

    def logical_blocks(self) -> int:
        return len(self.block_ids)


class PagedKVCache:
    """A pool of fixed-size KV blocks shared by every live sequence."""

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        poison_on_free: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
        counters: Counters | None = None,
    ):
        self.counters = counters if counters is not None else Counters()
        if registry.KV_TILE % block_size != 0:
            raise ValueError(
                f"block_size {block_size} does not divide the KV tile "
                f"{registry.KV_TILE}; a tile would straddle a block boundary"
            )
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.poison_on_free = poison_on_free

        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)]

        self._free: list[int] = list(range(num_blocks))
        heapq.heapify(self._free)
        self.refcount: list[int] = [0] * num_blocks
        self.sequences: dict[str, Sequence] = {}

        self.pinned: dict[int, int] = {}

        self.stats = {"allocated": 0, "freed": 0}


    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def _take_block(self) -> int:
        if not self._free:
            self.counters.hit("out_of_blocks")
            raise OutOfBlocks(
                f"all {self.num_blocks} blocks are held; the policy must evict or "
                "preempt before asking for more"
            )
        block = heapq.heappop(self._free)
        assert self.refcount[block] == 0, f"block {block} was free with refcount != 0"
        self.refcount[block] = 1
        self.stats["allocated"] += 1
        return block

    def _release_block(self, block: int) -> None:
        assert self.refcount[block] > 0, f"releasing block {block} at refcount 0"
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            if self.poison_on_free:
                for layer in range(self.num_layers):
                    self.k[layer][block].fill_(POISON)
                    self.v[layer][block].fill_(POISON)
            heapq.heappush(self._free, block)
            self.stats["freed"] += 1
            self.counters.hit("block_reclaimed_at_zero")


    def pin(self, block: int) -> None:
        """Take a reference on behalf of the prefix cache index."""
        assert self.refcount[block] > 0, f"pinning free block {block}"
        self.refcount[block] += 1
        self.pinned[block] = self.pinned.get(block, 0) + 1

    def unpin(self, block: int) -> None:
        """Release the prefix cache index's reference."""
        held = self.pinned.get(block, 0)
        assert held > 0, f"unpinning block {block}, which the cache does not hold"
        if held == 1:
            del self.pinned[block]
        else:
            self.pinned[block] = held - 1
        self._release_block(block)

    def adopt(self, uid: str, blocks: list[int]) -> None:
        """Share `blocks` into `uid` as its leading logical blocks."""
        sequence = self.sequences[uid]
        assert not sequence.block_ids, "adopt expects a sequence with no blocks yet"
        for block in blocks:
            assert self.refcount[block] > 0, f"adopting free block {block}"
            self.refcount[block] += 1
        sequence.block_ids = list(blocks)

    def create(self, uid: str) -> Sequence:
        if uid in self.sequences:
            raise KeyError(f"sequence {uid!r} already exists")
        sequence = Sequence(uid=uid)
        self.sequences[uid] = sequence
        return sequence

    def reserve(self, uid: str, total_tokens: int) -> None:
        """Grow `uid` so it can hold `total_tokens`. Does not shrink."""
        sequence = self.sequences[uid]
        needed = -(-total_tokens // self.block_size)
        while sequence.logical_blocks() < needed:
            sequence.block_ids.append(self._take_block())

    def release(self, uid: str) -> None:
        sequence = self.sequences.pop(uid)
        for block in sequence.block_ids:
            self._release_block(block)
        sequence.block_ids.clear()
        sequence.length = 0

    def assert_exclusive(self, uid: str, block: int) -> None:
        """The invariant that makes copy-on-write unnecessary here."""
        holders = sum(
            1 for sequence in self.sequences.values()
            if sequence.uid != uid and block in sequence.block_ids
        )
        if holders or self.pinned.get(block, 0):
            raise AuditFailure(
                f"{uid} is about to write physical block {block}, which is also "
                f"held by {holders} other sequence(s) and {self.pinned.get(block, 0)} "
                "cache entries. Prefix sharing is supposed to be whole-block, so a "
                "write should only ever land in a block this sequence reserved."
            )


    def block_table(self, uid: str) -> torch.Tensor:
        """Logical-to-physical mapping for `uid`, int32, on the pool's device."""
        return torch.tensor(
            self.sequences[uid].block_ids, dtype=torch.int32, device=self.device
        )

    def evictable(self) -> list[int]:
        """Blocks held by exactly one sequence and by no other holder."""
        return sorted(b for b in range(self.num_blocks) if self.refcount[b] == 1)


    def audit(self) -> None:
        """Internal invariants, checked every scheduler step in debug builds."""
        expected = [0] * self.num_blocks
        for block, count in self.pinned.items():
            expected[block] += count
        for sequence in self.sequences.values():
            seen = set()
            for block in sequence.block_ids:
                if block in seen:
                    raise AuditFailure(
                        f"sequence {sequence.uid!r} maps two logical blocks to "
                        f"physical block {block} without a copy-on-write record"
                    )
                seen.add(block)
                expected[block] += 1

        for block in range(self.num_blocks):
            if self.refcount[block] != expected[block]:
                raise AuditFailure(
                    f"refcount ledger does not balance at block {block}: refcount "
                    f"{self.refcount[block]}, but holders total {expected[block]} "
                    f"({self.pinned.get(block, 0)} cache, "
                    f"{expected[block] - self.pinned.get(block, 0)} sequences)"
                )

        free = set(self._free)
        if len(free) != len(self._free):
            raise AuditFailure("a block appears in the free pool more than once")
        for block in range(self.num_blocks):
            if (self.refcount[block] == 0) != (block in free):
                raise AuditFailure(
                    f"block {block} has refcount {self.refcount[block]} but is "
                    f"{'in' if block in free else 'not in'} the free pool"
                )

    def state_digest(self) -> tuple:
        """Allocator state, for the trajectory hash. Order-stable by construction."""
        return (
            tuple(self.refcount),
            tuple(sorted(self._free)),
            tuple(
                (uid, tuple(self.sequences[uid].block_ids), self.sequences[uid].length)
                for uid in sorted(self.sequences)
            ),
            tuple(sorted(self.stats.items())),
            tuple(sorted(self.pinned.items())),
            self.block_size,
        )
