"""Exact-prefix cache: block-aligned, hash-keyed, refcounted.

PRD must-have: "Exact-prefix, block-aligned, dict keyed on token hash, refcounted
blocks with copy-on-write."

docs/02-technical-architecture.md section 4.3 is the reason this is sound and
also the reason it is dangerous: "A cache hit returns stored bytes.
Hit-versus-miss is therefore bit-identical if and only if chunk invariance holds,
because the cached bits equal what recomputation would produce. Without chunk
invariance, caching changes numerics."

Week 3 proved chunk invariance in the compute path. What this file must not do is
launder a *different* chunk boundary into a stored result: if a block's KV were
written by a prefill chunked one way and later reused by a request that would
have chunked it another way, the cache would be the thing that made the two
differ. MR4 crosses that dimension deliberately rather than testing only cold
versus warm.

Keying is a hash chain, not a hash of the whole prefix. Block i's key is
H(key of block i-1, tokens of block i), so a match at block i implies a match on
every block before it. A flat per-block hash would collide across different
histories that happen to share one block of tokens, and that is precisely the
"one request reads another request's KV" class the security doc calls a security
issue rather than a correctness bug.

Only whole blocks are cached. A partial trailing block is never inserted: its
remaining slots would be written by whoever reused it, so sharing it would let
one sequence observe another's writes.
"""

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
    """Maps a block-aligned token prefix to the physical block holding its KV.

    The cache holds one reference on every block it indexes, so an entry keeps
    its block alive after the producing sequence has finished. Eviction is what
    releases that reference, and choosing the victim is a policy decision that
    lives in engine/sched/policy.py; this class only reports candidates.
    """

    block_size: int
    entries: dict[bytes, CacheEntry] = field(default_factory=dict)
    stats: dict = field(
        default_factory=lambda: {"lookups": 0, "hit_blocks": 0, "inserts": 0, "evictions": 0}
    )

    def keys_for(self, tokens: list[int]) -> list[tuple[bytes, tuple[int, ...]]]:
        """The hash chain over whole blocks of `tokens`.

        A trailing partial block is excluded, so the returned list covers exactly
        len(tokens) // block_size blocks.
        """
        chain: list[tuple[bytes, tuple[int, ...]]] = []
        parent = ROOT_KEY
        whole = len(tokens) // self.block_size
        for index in range(whole):
            chunk = tuple(tokens[index * self.block_size : (index + 1) * self.block_size])
            parent = block_key(parent, chunk)
            chain.append((parent, chunk))
        return chain

    def lookup(self, tokens: list[int]) -> tuple[int, list[int]]:
        """Longest block-aligned prefix present, as (token count, blocks).

        Stops at the first miss rather than probing past it: the chain means a
        later block cannot match unless every earlier one did, so probing on
        would be looking for a key that cannot exist.
        """
        self.stats["lookups"] += 1
        blocks: list[int] = []
        for key, chunk in self.keys_for(tokens):
            entry = self.entries.get(key)
            if entry is None:
                break
            if entry.tokens != chunk:
                # sha256 collision, or a bug. Either way, do not serve it.
                break
            entry.hits += 1
            blocks.append(entry.physical_block)
        self.stats["hit_blocks"] += len(blocks)
        return len(blocks) * self.block_size, blocks

    def insert(self, tokens: list[int], block_ids: list[int], pool) -> int:
        """Index every whole block of `tokens`, taking a reference on each.

        Returns how many new entries were created. Blocks already indexed under
        the same key are left alone: re-inserting would take a second reference
        the cache would never release.
        """
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
        return created

    def evictable_blocks(self, pool) -> list[int]:
        """Indexed blocks that no live sequence is using.

        A block whose refcount is exactly the cache's own single reference is
        reclaimable. Anything higher is in use by a running sequence and evicting
        it would be the "eviction eligible-set includes a running sequence"
        mutation from architecture doc 10.1.
        """
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
                return True
        return False

    def state_digest(self) -> tuple:
        """For the trajectory hash. Sorted, so dict order never leaks in."""
        return tuple(
            sorted((key.hex()[:16], entry.physical_block, entry.tokens)
                   for key, entry in self.entries.items())
        )
