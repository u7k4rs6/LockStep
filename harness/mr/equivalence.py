"""Decode-versus-prefill KV equivalence, MR1, and MR2."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine.kernels import registry  # noqa: E402
from engine.kv import paged  # noqa: E402
from engine.model.qwen3 import KVCache, Qwen3  # noqa: E402

SPLIT = registry.SPLIT_SIZE


@dataclass
class Result:
    relation: str
    statement: str
    passed: bool
    detail: str = ""
    cases: list = field(default_factory=list)


def _blocks_for(lengths: list[int], block_size: int) -> int:
    """Blocks needed for a set of sequences."""
    return sum(-(-length // block_size) for length in lengths) + 2


def _pool(model: Qwen3, lengths: list[int], block_size: int) -> paged.PagedKVCache:
    return paged.PagedKVCache(
        num_blocks=_blocks_for(lengths, block_size),
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        device=model.device,
        dtype=torch.float16,
        poison_on_free=False,
        block_size=block_size,
    )


def read_kv(pool: paged.PagedKVCache, uid: str, layer: int, count: int) -> tuple:
    """Gather logical positions [0, count) out of their physical blocks."""
    table = pool.sequences[uid].block_ids
    keys, values = [], []
    taken = 0
    while taken < count:
        logical, slot = divmod(taken, pool.block_size)
        take = min(pool.block_size - slot, count - taken)
        physical = table[logical]
        keys.append(pool.k[layer][physical, slot : slot + take])
        values.append(pool.v[layer][physical, slot : slot + take])
        taken += take
    return torch.cat(keys), torch.cat(values)


def kv_by_partition(
    model: Qwen3, prompt: list[int], partition: list[int], block_size: int
) -> tuple[list[tuple], torch.Tensor]:
    """Run `prompt` through the engine in the given chunk sizes."""
    assert sum(partition) == len(prompt), "partition must cover the prompt exactly"
    pool = _pool(model, [len(prompt)], block_size)
    uid = "seq"
    pool.create(uid)
    pool.reserve(uid, len(prompt))

    start = 0
    logits = None
    for size in partition:
        out = model.forward_batch(pool, [(uid, prompt[start : start + size], start)])
        logits = out[uid][-1].clone()
        start += size

    kv = [read_kv(pool, uid, layer, len(prompt)) for layer in range(model.cfg.num_hidden_layers)]
    kv = [(k.clone(), v.clone()) for k, v in kv]
    del pool
    torch.cuda.empty_cache()
    return kv, logits


def first_kv_divergence(a: list[tuple], b: list[tuple]) -> tuple | None:
    """(layer, tensor, position) of the first differing KV entry, or None."""
    for layer, ((ka, va), (kb, vb)) in enumerate(zip(a, b)):
        for name, x, y in (("K", ka, kb), ("V", va, vb)):
            if not torch.equal(x, y):
                rows = torch.nonzero((x != y).flatten(1).any(dim=1)).flatten()
                return (layer, name, int(rows[0]), float((x - y).abs().max()))
    return None


def boundary_partitions(length: int, block_size: int) -> list[tuple[str, list[int]]]:
    """Partitions landing exactly on, and one either side of, each boundary."""
    cases: list[tuple[str, list[int]]] = []

    def add(label: str, first: int) -> None:
        if 1 <= first < length:
            cases.append((label, [first, length - first]))

    for name, value in (("block_size", block_size), ("split_size", SPLIT)):
        for delta, suffix in ((-1, " - 1"), (0, ""), (1, " + 1")):
            add(f"first chunk = {name}{suffix}", value + delta)

    add("first chunk = 1", 1)
    add("first chunk = length - 1", length - 1)
    cases.append(("single uninterrupted prefill", [length]))
    cases.append(("token at a time", [1] * length))

    ragged, remaining, size = [], length, 7
    while remaining > 0:
        take = min(size, remaining)
        ragged.append(take)
        remaining -= take
        size = size * 2 + 3
    cases.append(("ragged, misaligned to block and split", ragged))
    return cases


def decode_vs_prefill_kv(
    model: Qwen3, prompt: list[int], block_size: int = paged.DEFAULT_BLOCK_SIZE
) -> Result:
    """The property weeks 4 through 8 rest on, checked on the KV tensors."""
    reference, _ = kv_by_partition(model, prompt, [len(prompt)], block_size)
    decoded, _ = kv_by_partition(model, prompt, [1] * len(prompt), block_size)

    divergence = first_kv_divergence(reference, decoded)
    layers = model.cfg.num_hidden_layers
    entries = layers * 2 * len(prompt)
    if divergence is None:
        return Result(
            "KV-EQ",
            "decode-produced KV is bit-identical to prefill-produced KV",
            True,
            f"{entries} tensor rows over {layers} layers, {len(prompt)} positions, bitwise",
        )
    layer, tensor, position, magnitude = divergence
    return Result(
        "KV-EQ",
        "decode-produced KV is bit-identical to prefill-produced KV",
        False,
        f"first divergence layer {layer} {tensor}[{position}], max abs {magnitude:.3e}",
    )


def path_equivalence(
    model: Qwen3, prompts: list[list[int]],
    block_size: int = paged.DEFAULT_BLOCK_SIZE,
) -> Result:
    """`forward(x)` must equal `forward_batch([x])` bitwise, per position."""
    cases = []
    passed = True

    for index, prompt in enumerate(prompts):
        cache = KVCache(model.cfg, len(prompt), model.device, block_size=block_size)
        single = model.forward(
            torch.tensor(prompt, dtype=torch.long, device=model.device), 0, cache
        ).clone()
        del cache

        pool = _pool(model, [len(prompt)], block_size)
        uid = "solo"
        pool.create(uid)
        pool.reserve(uid, len(prompt))
        batched = model.forward_batch(pool, [(uid, prompt, 0)])[uid].clone()
        del pool
        torch.cuda.empty_cache()

        identical = torch.equal(single, batched)
        passed &= identical
        row = None
        if not identical:
            rows = torch.nonzero((single != batched).any(dim=-1)).flatten()
            row = int(rows[0])
        cases.append({
            "prompt_tokens": len(prompt),
            "positions": single.shape[0],
            "identical": identical,
            "first_differing_position": row,
            "max_abs": 0.0 if identical else float(
                (single.to(torch.float64) - batched.to(torch.float64)).abs().max()
            ),
        })

    return Result(
        "PATH-EQ",
        "forward and forward_batch produce identical logits for a lone request",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} prompts bitwise "
        f"identical across both implementations, "
        f"{sum(c['positions'] for c in cases)} positions, block_size={block_size}",
        cases,
    )


def mr1_batch_composition(
    model: Qwen3,
    prompts: list[list[int]],
    sizes=(1, 2, 3, 4, 8, 16, 31, 32),
    block_size: int = paged.DEFAULT_BLOCK_SIZE,
) -> Result:
    """A request's logit bytes do not depend on who else is in the batch."""
    canonical = None
    cases = []
    passed = True

    for size in sizes:
        pool = _pool(model, [len(p) for p in prompts[:size]], block_size)
        work = []
        for index, prompt in enumerate(prompts[:size]):
            uid = f"r{index:02d}"
            pool.create(uid)
            pool.reserve(uid, len(prompt))
            work.append((uid, prompt, 0))
        observed = model.forward_batch(pool, work)["r00"].clone()
        del pool
        torch.cuda.empty_cache()

        if canonical is None:
            canonical = observed
            cases.append({"batch_size": size, "identical": True, "note": "canonical C(r)"})
            continue
        identical = torch.equal(observed, canonical)
        passed &= identical
        cases.append({
            "batch_size": size,
            "identical": identical,
            "max_abs": 0.0 if identical
            else float((observed.to(torch.float64) - canonical.to(torch.float64)).abs().max()),
        })

    return Result(
        "MR1",
        "batch composition: cohabitants leave r00's logit bytes unchanged",
        passed,
        f"{sum(1 for c in cases if c['identical'])}/{len(cases)} batch sizes bitwise identical "
        f"to C(r) over sizes {list(sizes)}",
        cases,
    )


def mr2_chunk_partition(
    model: Qwen3, prompt: list[int], block_size: int = paged.DEFAULT_BLOCK_SIZE
) -> Result:
    """Any partition of prefill leaves the KV and the final logits unchanged."""
    reference_kv, reference_logits = kv_by_partition(model, prompt, [len(prompt)], block_size)

    cases = []
    passed = True
    for label, partition in boundary_partitions(len(prompt), block_size):
        kv, logits = kv_by_partition(model, prompt, partition, block_size)
        divergence = first_kv_divergence(reference_kv, kv)
        logits_ok = torch.equal(logits, reference_logits)
        ok = divergence is None and logits_ok
        passed &= ok
        cases.append({
            "partition": label,
            "chunks": len(partition),
            "kv_identical": divergence is None,
            "logits_identical": logits_ok,
            "first_divergence": None if divergence is None else {
                "layer": divergence[0], "tensor": divergence[1],
                "position": divergence[2], "max_abs": divergence[3],
            },
        })

    return Result(
        "MR2",
        "chunk partition: any prefill partition leaves KV and logits unchanged",
        passed,
        f"{sum(1 for c in cases if c['kv_identical'] and c['logits_identical'])}/{len(cases)} "
        f"partitions bitwise identical, block_size={block_size}",
        cases,
    )
