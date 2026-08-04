"""Chunk invariance at the KV tensor level, plus batch composition.

These are the suite versions of what harness/mr/ reports as a table. They run on
short prompts so the suite stays fast; harness/mr/run.py runs the same relations
on a 600-token prompt that reaches the 512 split boundary.

Architecture doc 4.1: recompute correctness reduces to decode-versus-prefill
equivalence per token, which is exactly chunk invariance. If this file goes red,
weeks 4 through 8 are built on sand, so it fails the suite rather than only a
report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernels under claim run on the pinned GPU"
)

from engine.kv.paged import SUPPORTED_BLOCK_SIZES  # noqa: E402
from engine.model.qwen3 import Qwen3  # noqa: E402
from harness.mr.equivalence import (  # noqa: E402
    first_kv_divergence,
    kv_by_partition,
    mr1_batch_composition,
)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "Qwen3-0.6B"


@pytest.fixture(scope="module")
def model():
    if not (WEIGHTS / "model.safetensors").is_file():
        pytest.skip("weights not downloaded")
    return Qwen3(WEIGHTS, max_len=512)


@pytest.fixture(scope="module")
def prompt(model):
    generator = torch.Generator().manual_seed(17)
    return torch.randint(0, model.cfg.vocab_size, (70,), generator=generator).tolist()


@pytest.mark.parametrize("block_size", SUPPORTED_BLOCK_SIZES)
def test_decode_produced_kv_equals_prefill_produced_kv(model, prompt, block_size):
    reference, _ = kv_by_partition(model, prompt, [len(prompt)], block_size)
    decoded, _ = kv_by_partition(model, prompt, [1] * len(prompt), block_size)
    assert first_kv_divergence(reference, decoded) is None


@pytest.mark.parametrize(
    "partition",
    [
        [15, 55], [16, 54], [17, 53],   # block_size 16 and either side
        [1, 69], [69, 1],               # extremes
        [7, 17, 23, 23],                # misaligned to every boundary
        [64, 6], [63, 7], [65, 5],      # block_size 64 and either side
    ],
)
def test_any_chunk_partition_leaves_kv_and_logits_unchanged(model, prompt, partition):
    reference_kv, reference_logits = kv_by_partition(model, prompt, [len(prompt)], 16)
    kv, logits = kv_by_partition(model, prompt, partition, 16)
    assert first_kv_divergence(reference_kv, kv) is None
    assert torch.equal(logits, reference_logits)


def test_a_partition_that_does_not_cover_the_prompt_is_refused(model, prompt):
    with pytest.raises(AssertionError, match="cover the prompt exactly"):
        kv_by_partition(model, prompt, [5, 5], 16)


def test_batch_composition_leaves_a_request_bitwise_unchanged(model):
    generator = torch.Generator().manual_seed(4242)
    prompts = [
        torch.randint(0, model.cfg.vocab_size, (12 + 5 * (i % 4),), generator=generator).tolist()
        for i in range(8)
    ]
    result = mr1_batch_composition(model, prompts, sizes=(1, 2, 3, 4, 8), block_size=16)
    assert result.passed, result.detail
