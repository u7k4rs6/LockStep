"""A pinned repro for a torch bug that would silently corrupt the F1 reference."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only hazard")

AFFECTED_WIDTHS = (513, 769)
SAFE_WIDTHS = (512, 514, 768, 770, 1024, 151936)


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    m = x.max(dim=-1, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


def _row(width: int, rows: int = 8) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(11)
    return torch.randn(rows, width, generator=generator, device="cuda", dtype=torch.float64) * 12


@pytest.mark.parametrize("width", AFFECTED_WIDTHS)
def test_torch_softmax_is_still_broken_at_these_widths(width):
    """Asserts the bug is present. If this fails, torch fixed it: delete the"""
    x = _row(width)
    cpu = torch.softmax(x.cpu(), dim=-1).cuda()
    assert (torch.softmax(x, dim=-1) - cpu).abs().max() > 1e-6, (
        f"torch.softmax fp64 CUDA is now correct at width {width}"
    )


@pytest.mark.parametrize("width", AFFECTED_WIDTHS + SAFE_WIDTHS)
def test_the_explicit_softmax_is_correct_everywhere(width):
    """The replacement must be right at both the broken and the safe widths."""
    x = _row(width)
    cpu = torch.softmax(x.cpu(), dim=-1).cuda()
    assert (stable_softmax(x) - cpu).abs().max() < 1e-12


@pytest.mark.parametrize("width", AFFECTED_WIDTHS)
def test_the_reductions_the_engine_relies_on_are_unaffected(width):
    """fidelity.py uses logsumexp, topk, and max on CUDA in fp64. They are clean"""
    x = _row(width)
    assert (torch.logsumexp(x, -1) - torch.logsumexp(x.cpu(), -1).cuda()).abs().max() < 1e-12
    assert (x.max(-1).values - x.cpu().max(-1).values.cuda()).abs().max() == 0
    assert (torch.topk(x, 8, -1).values - torch.topk(x.cpu(), 8, -1).values.cuda()).abs().max() == 0


def test_cpu_softmax_is_correct_at_the_affected_widths():
    """The fp64 reference runs on CPU. This is the property that lets it."""
    for width in AFFECTED_WIDTHS:
        x = _row(width).cpu()
        assert (torch.softmax(x, dim=-1) - stable_softmax(x)).abs().max() < 1e-12
