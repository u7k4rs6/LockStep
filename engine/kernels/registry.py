"""Pinned Triton launch configurations. No autotuning, anywhere, ever.

docs/02-technical-architecture.md section 5: "Triton autotuning. Timing-based, so
the same binary can pick different configs across runs. Every config is pinned in
a committed registry."

Section 12 gate 4: a change to this file is a claims-affecting change and
requires re-running the invariance suite before publishing any number. The file's
sha256 is recorded in env.lock, so an artifact produced under a different
registry is detectable rather than merely suspected.

The selection rule, which is the whole point of the file:

    A config is looked up by a *name*, and by nothing else.

Names are compile-time constants at their call sites. No shape, and in
particular no batch size, batch-max sequence length, or total token count, is an
argument to `get`. There is therefore no code path by which a batch-derived
quantity can reach a kernel config, because the lookup takes no such parameter.

M-tiling deserves a note, since the architecture doc (section 4.2) permits it to
vary: "Rows are independent, so M-tiling may vary freely; the K-loop and the KV
traversal may not." We pin BLOCK_M anyway. Varying it is provably safe but
requires the reader to reconstruct that proof; a constant requires nothing. The
cost is that a batch-1 decode pads M from 1 to 16, which this regime can afford
because it is launch-overhead bound rather than occupancy bound.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

REGISTRY_FILE = Path(__file__).resolve()


class UnknownConfig(KeyError):
    """A kernel asked for a config that is not pinned. Never fall back."""


@dataclass(frozen=True)
class Config:
    """One pinned launch configuration.

    `constants` are passed to the kernel as `tl.constexpr`. `num_warps` and
    `num_stages` are Triton launch parameters. `why` is not decoration: when this
    file changes, the claims-table review reads these.
    """

    constants: MappingProxyType
    num_warps: int
    num_stages: int
    why: str

    def __post_init__(self) -> None:
        for key, value in self.constants.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key}={value!r} is not a positive compile-time int")


def _cfg(why: str, num_warps: int, num_stages: int, **constants: int) -> Config:
    return Config(
        constants=MappingProxyType(dict(constants)),
        num_warps=num_warps,
        num_stages=num_stages,
        why=why,
    )


# Model-shape constants for Qwen3-0.6B. These are properties of the weights, not
# of any batch, so kernels may specialize on them freely.
HEAD_DIM = 128
HIDDEN_SIZE = 1024

# The two numbers the architecture doc fixes by name (section 4.2).
KV_TILE = 128  # "Constant KV tile (128)"
SPLIT_SIZE = 512  # "Fixed split size (512), never a fixed split count"

_REGISTRY: dict[str, Config] = {
    # ---- GEMM. Fixed BLOCK_K, no split-K, including lm_head. ----
    "gemm.default": _cfg(
        "BLOCK_K=64 divides every K in this model (1024, 2048, 3072), so the "
        "K-loop trip count is a function of the weight shape alone. BLOCK_M=16 "
        "is the native mma M on sm_89, the smallest tile that still reaches "
        "tensor cores, which keeps the batch-1 decode padding at 16 rows.",
        num_warps=4,
        num_stages=3,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=64,
    ),
    "gemm.lm_head": _cfg(
        "Vocab 151936 = 128 * 1187, so BLOCK_N=128 tiles N exactly. Split-K is "
        "the obvious move on a K=1024 N=151936 GEMM and is exactly the thing "
        "forbidden: cuBLAS split-K combines via atomics or a heuristically "
        "chosen reduce pass, either of which is batch-shape dependent.",
        num_warps=4,
        num_stages=3,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=64,
    ),
    # ---- RMSNorm. One CTA per row; the reduction never crosses a CTA. ----
    "rmsnorm.hidden": _cfg(
        "hidden_size 1024 fits one CTA tile, so the sum of squares is a single "
        "in-CTA tree reduction over a fixed 1024-wide tile with no cross-CTA "
        "combine and no dependence on how many rows are in flight.",
        num_warps=4,
        num_stages=1,
        BLOCK_SIZE=1024,
    ),
    "rmsnorm.head": _cfg(
        "Qwen3's per-head q_norm and k_norm run over head_dim 128. Same "
        "one-CTA-per-row structure at a smaller tile.",
        num_warps=2,
        num_stages=1,
        BLOCK_SIZE=128,
    ),
    # ---- Softmax. One CTA per row, sequential fixed-size tiles. ----
    "softmax.vocab": _cfg(
        "151936 logits do not fit one tile, so the CTA walks ceil(151936/4096) "
        "= 38 tiles in ascending order. The trip count is a function of the "
        "vocab size, which is a weight-shape constant, and the accumulation "
        "order is fixed by the loop rather than by a scheduler.",
        num_warps=8,
        num_stages=1,
        BLOCK_SIZE=4096,
    ),
    # ---- Elementwise. No reduction, so nothing to order. ----
    "swiglu": _cfg(
        "silu(gate) * up over a flat index space. Invariant by construction "
        "since each output element reads one gate element and one up element; "
        "the tile width only sets how many launches cover the tensor.",
        num_warps=4,
        num_stages=1,
        BLOCK_SIZE=1024,
    ),
    "rope": _cfg(
        "Purely elementwise over head_dim/2 rotation pairs. Listed here so the "
        "registry is the complete set of launches, not the interesting subset.",
        num_warps=4,
        num_stages=1,
        BLOCK_PAIRS=64,
    ),
    # ---- Attention. Fixed split size, fixed KV tile, fp32 partials. ----
    "attention.split": _cfg(
        "KV tile 128 and split size 512 are the architecture doc's numbers. Each "
        "program owns one 512-token span of one request's own KV and walks it in "
        "four ascending 128-token tiles. The number of splits is "
        "ceil(kv_len/512), a function of this request's own KV length and of "
        "nothing else in the batch. A fixed split *count* would divide by a "
        "batch-max sequence length and reintroduce the dependence. BLOCK_M "
        "matches gemm.default for the same reason it is pinned there.",
        num_warps=4,
        num_stages=2,
        BLOCK_M=16,
        BLOCK_N=KV_TILE,
        BLOCK_D=HEAD_DIM,
        SPLIT_SIZE=SPLIT_SIZE,
    ),
    "attention.combine": _cfg(
        "A single CTA per (query tile, head) folds that tile's per-split "
        "partials in ascending split index, accumulating in fp32. Every row's "
        "fold happens inside one CTA, so there are no atomics and no cross-CTA "
        "ordering; ascending order makes the fold a pure function of the split "
        "count. MAX_SPLITS is an asserted ceiling, not a tile: the fold loops to "
        "the request's own split count, and the assertion refuses a KV length "
        "past what has been validated. 128 splits of 512 covers 65536 KV "
        "tokens, well past what 8 GB of VRAM holds.",
        num_warps=4,
        num_stages=1,
        BLOCK_M=16,
        BLOCK_D=HEAD_DIM,
        MAX_SPLITS=128,
    ),
}

REGISTRY = MappingProxyType(_REGISTRY)


def get(name: str) -> Config:
    """Look up a pinned config by name.

    Takes no shape argument, by design. If you find yourself wanting to pass one,
    that is the constraint working: see the module docstring.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownConfig(
            f"{name!r} is not pinned. Add it to the registry deliberately; there "
            f"is no autotuned fallback. Known: {sorted(REGISTRY)}"
        ) from None


def digest() -> str:
    """sha256 of this file, recorded in env.lock."""
    return hashlib.sha256(REGISTRY_FILE.read_bytes()).hexdigest()


def describe() -> str:
    """The registry as a table. Printed by `scripts/static_checks.py --show`."""
    lines = [f"kernel config registry  sha256:{digest()[:16]}", ""]
    for name in sorted(REGISTRY):
        cfg = REGISTRY[name]
        constants = "  ".join(f"{k}={v}" for k, v in cfg.constants.items())
        lines.append(f"{name:<20} {constants}")
        lines.append(f"{'':<20} num_warps={cfg.num_warps}  num_stages={cfg.num_stages}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
