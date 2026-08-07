"""Exact-prefix cache: block-aligned, hash-keyed, refcounted."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

ROOT_KEY = b"lockstep-prefix-root"


def block_key(parent: bytes, tokens: tuple[int, ...]) -> bytes:
    """H(parent key, tokens). The chain is what makes a match a real prefix."""
    digest = hashlib.sha256()
    digest.update(parent)
    digest.update(b"|")
    digest.update(",".join(str(t) for t in tokens).encode("ascii"))
    return digest.digest()


@dataclass
class CacheEntry:
    physical_block: int
    tokens: tuple[int, ...]
    hits: int = 0


@dataclass
class PrefixCache:
    """Maps a block-aligned token prefix to the physical block holding its KV."""

    block_size: int
    counters: object = None
    entries: dict[bytes, CacheEntry] = field(default_factory=dict)
    stats: dict = field(
        default_factory=lambda: {"lookups": 0, "hit_blocks": 0, "inserts": 0, "evictions": 0}
    )

    def keys_for(self, tokens: list[int]) -> list[tuple[bytes, tuple[int, ...]]]:
        """The hash chain over whole blocks of `tokens`."""
        chain: list[tuple[bytes, tuple[int, ...]]] = []
        parent = ROOT_KEY
        whole = len(tokens) // self.block_size
        for index in range(whole):
            chunk = tuple(tokens[index * self.block_size : (index + 1) * self.block_size])
            parent = block_key(parent, chunk)
            chain.append((parent, chunk))
        return chain

    def lookup(self, tokens: list[int]) -> tuple[int, list[int]]:
        """Longest block-aligned prefix present, as (token count, blocks)."""
        self.stats["lookups"] += 1
        blocks: list[int] = []
        for key, chunk in self.keys_for(tokens):
            entry = self.entries.get(key)
            if entry is None:
                break
            if entry.tokens != chunk:
                break
            entry.hits += 1
            blocks.append(entry.physical_block)
        self.stats["hit_blocks"] += len(blocks)
        if self.counters is not None:
            self.counters.hit("cache_hit" if blocks else "cache_miss")
        return len(blocks) * self.block_size, blocks

    def insert(self, tokens: list[int], block_ids: list[int], pool) -> int:
        """Index every whole block of `tokens`, taking a reference on each."""
        created = 0
        for index, (key, chunk) in enumerate(self.keys_for(tokens)):
            if index >= len(block_ids):
                break
            if key in self.entries:
                continue
            physical = block_ids[index]
            self.entries[key] = CacheEntry(physical_block=physical, tokens=chunk)
            pool.pin(physical)
            created += 1
        self.stats["inserts"] += created
        if created and self.counters is not None:
            self.counters.hit("cache_insert", created)
        return created

    def evictable_blocks(self, pool) -> list[int]:
        """Indexed blocks that no live sequence is using."""
        return sorted(
            entry.physical_block
            for entry in self.entries.values()
            if pool.refcount[entry.physical_block] == 1
        )

    def evict(self, physical_block: int, pool) -> bool:
        """Drop the entry naming `physical_block` and release its reference."""
        for key, entry in list(self.entries.items()):
            if entry.physical_block == physical_block:
                del self.entries[key]
                pool.unpin(physical_block)
                self.stats["evictions"] += 1
                if self.counters is not None:
                    self.counters.hit("eviction_taken")
                return True
        return False

    def state_digest(self) -> tuple:
        """For the trajectory hash. Sorted, so dict order never leaks in."""
        return tuple(
            sorted((key.hex()[:16], entry.physical_block, entry.tokens)
                   for key, entry in self.entries.items())
        )
