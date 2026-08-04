"""Standalone repro: torch.softmax is wrong in fp64 on CUDA at some row widths.

Deliberately imports nothing from lockstep and touches no other CUDA work, so it
can be pasted into a pytorch issue and run as-is.

Run:  python3 scripts/repro_torch_softmax_fp64.py
"""

import torch


def stable_softmax(x):
    """Textbook max-subtract softmax, for comparison."""
    m = x.max(dim=-1, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


def main():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"device {torch.cuda.get_device_name(0)}")
    print(f"deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    print()

    generator = torch.Generator(device="cuda").manual_seed(11)
    base = torch.randn(64, 4096, generator=generator, device="cuda", dtype=torch.float64) * 12

    print("A. probabilities must sum to 1. rows=8, float64, CUDA")
    for width in (512, 513, 514, 768, 769, 770, 1024):
        x = base[:8, :width].contiguous()
        sums = torch.softmax(x, dim=-1).sum(dim=-1)
        worst = float((sums - 1.0).abs().max())
        flag = "   <-- WRONG" if worst > 1e-12 else ""
        print(f"   width {width:>5}: max |sum - 1| = {worst:.3e}{flag}")

    print()
    print("B. which (rows, width) pairs disagree with the CPU result")
    print(f"   {'width':>7}" + "".join(f"{r:>12}" for r in (1, 2, 4, 8, 16, 32)))
    for width in (512, 513, 768, 769, 1024):
        line = f"   {width:>7}"
        for rows in (1, 2, 4, 8, 16, 32):
            x = base[:rows, :width].contiguous()
            cpu = torch.softmax(x.cpu(), dim=-1).cuda()
            line += f"{float((torch.softmax(x, -1) - cpu).abs().max()):>12.1e}"
        print(line)

    x = base[:8, :513].contiguous()
    cpu = torch.softmax(x.cpu(), dim=-1).cuda()

    print()
    print("C. scope, all at width 513, rows 8")
    print(f"   softmax      float64 CUDA : {float((torch.softmax(x, -1) - cpu).abs().max()):.3e}")
    print(f"   softmax      float32 CUDA : "
          f"{float((torch.softmax(x.float(), -1) - cpu.float()).abs().max()):.3e}")
    print(f"   softmax      float64 CPU  : "
          f"{float((torch.softmax(x.cpu(), -1) - cpu.cpu()).abs().max()):.3e}")
    print(f"   stable_softmax float64 CUDA: {float((stable_softmax(x) - cpu).abs().max()):.3e}")

    log_cpu = torch.log_softmax(x.cpu(), dim=-1).cuda()
    print(f"   log_softmax  float64 CUDA : "
          f"{float((torch.log_softmax(x, -1) - log_cpu).abs().max()):.3e}")
    print(f"   logsumexp    float64 CUDA : "
          f"{float((torch.logsumexp(x, -1) - torch.logsumexp(x.cpu(), -1).cuda()).abs().max()):.3e}")

    # Non-contiguous input: a transposed view of the same values.
    wide = base[:8, :513].t().contiguous().t()
    print(f"   softmax on a non-contiguous view: "
          f"{float((torch.softmax(wide, -1) - torch.softmax(wide.cpu(), -1).cuda()).abs().max()):.3e}")

    print()
    print("D. does deterministic mode change it")
    torch.use_deterministic_algorithms(True)
    try:
        print(f"   deterministic algorithms on : "
              f"{float((torch.softmax(x, -1) - cpu).abs().max()):.3e}")
    finally:
        torch.use_deterministic_algorithms(False)

    print()
    print("Expected: A and B show width 513 and 769 failing for rows >= 2;")
    print("float32, CPU, log_softmax, logsumexp, and the explicit softmax are clean.")


if __name__ == "__main__":
    main()
