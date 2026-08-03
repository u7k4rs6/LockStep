"""A pinned repro for a torch bug that would silently corrupt the F1 reference.

`torch.softmax(dim=-1)` on CUDA in float64 returns wrong probabilities at row
widths 513 and 769 when the tensor has two or more rows. The error reaches 0.5 in
probability, so it is not a precision issue. float32 and float16 are unaffected,
CPU is unaffected, and torch.logsumexp, torch.topk, and torch.max are unaffected
at the same shapes.

This matters here for two specific reasons rather than as trivia:

  * 513 is one past the pinned attention split size of 512, which is exactly the
    boundary the attention tests must cover. A reference built on torch.softmax
    reports a divergence there that the kernel did not commit.
  * The fp64 reference in scripts/build_fp64_reference.py is ground truth for
    every fidelity number. It runs on CPU, where this bug does not occur, and
    build_fp64_reference asserts that. This test is what makes that assertion
    load-bearing rather than decorative, so that "run the reference on the GPU,
    it would be faster" cannot quietly become a wrong published number.

If a torch upgrade fixes this, this test fails and can be deleted along with the
workarounds it justifies. That failure is the signal, which is why it asserts the
bug is present rather than skipping when it is absent.
"""

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
    """Asserts the bug is present. If this fails, torch fixed it: delete the
    workaround in tests/test_kernels.py and the CPU assertion in the reference
    builder, and remove this file."""
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
    """fidelity.py uses logsumexp, topk, and max on CUDA in fp64. They are clean
    at the widths that break softmax, which is why they were left alone."""
    x = _row(width)
    assert (torch.logsumexp(x, -1) - torch.logsumexp(x.cpu(), -1).cuda()).abs().max() < 1e-12
    assert (x.max(-1).values - x.cpu().max(-1).values.cuda()).abs().max() == 0
    assert (torch.topk(x, 8, -1).values - torch.topk(x.cpu(), 8, -1).values.cuda()).abs().max() == 0


def test_cpu_softmax_is_correct_at_the_affected_widths():
    """The fp64 reference runs on CPU. This is the property that lets it."""
    for width in AFFECTED_WIDTHS:
        x = _row(width).cpu()
        assert (torch.softmax(x, dim=-1) - stable_softmax(x)).abs().max() < 1e-12
