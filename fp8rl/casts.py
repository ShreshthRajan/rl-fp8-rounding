"""E4M3 casts with 128x128 block scales, as Transformer Engine's blockwise recipe applies them to FP8 primary weights.

A weight w is stored as q * s, where s is the block scale and q is an E4M3 value. With power-of-two scales (TE's
default for blockwise) s = 2^ceil(log2(amax / 448)), so s * q is exact in bf16. With amax scales s = amax / 448.
"""
import torch

E4M3_MAX = 448.0
BLOCK = 128
_SCALES = "pow2"


def set_block_scales(kind: str) -> None:
    """Choose power-of-two block scales ("pow2", TE's default) or FP32 amax scales ("amax")."""
    global _SCALES
    assert kind in ("pow2", "amax")
    _SCALES = kind


def block_scale(w: torch.Tensor) -> torch.Tensor:
    """Per-128x128-block scale of a 2-D weight, broadcast to w's shape."""
    r, c = w.shape
    pr, pc = (-r) % BLOCK, (-c) % BLOCK
    wp = torch.nn.functional.pad(w.abs(), (0, pc, 0, pr))
    amax = wp.view((r + pr) // BLOCK, BLOCK, (c + pc) // BLOCK, BLOCK).amax(dim=(1, 3)).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX))) if _SCALES == "pow2" else amax / E4M3_MAX
    return s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:r, :c]


def e4m3_ulp(y: torch.Tensor) -> torch.Tensor:
    """Spacing of the E4M3 grid at y (3 mantissa bits, minimum normal exponent -6)."""
    e = torch.floor(torch.log2(y.abs().clamp_min(2.0**-9))).clamp_min(-6)
    return torch.exp2(e - 3)


def fp8_rne(w: torch.Tensor) -> torch.Tensor:
    """Round to nearest (even), the cast TE applies today. Returns dequantized values."""
    s = block_scale(w)
    y = (w / s).clamp(-E4M3_MAX, E4M3_MAX)
    ulp = e4m3_ulp(y)
    return (torch.round(y / ulp) * ulp).clamp(-E4M3_MAX, E4M3_MAX) * s


def fp8_sr(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Stochastic rounding with fresh noise at every cast."""
    s = block_scale(w)
    y = (w / s).clamp(-E4M3_MAX, E4M3_MAX)
    ulp = e4m3_ulp(y)
    lo = torch.floor(y / ulp) * ulp
    u = torch.rand(y.shape, generator=gen, device="cpu").to(y.device)
    return torch.where(u < (y - lo) / ulp, lo + ulp, lo).clamp(-E4M3_MAX, E4M3_MAX) * s


def fp8_fixed_threshold(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Fixed-threshold rounding. u in [-1/2, 1/2) is drawn once per weight and kept for the run.

    |y| rounds up from its lower grid neighbour lo iff its position in the cell [lo, lo + ulp) is at least 1/2 + u.
    Averaged over u the cast is unbiased, grid points stay put, and no weight sits on a tie.
    """
    s = block_scale(w)
    y = (w / s).clamp(-E4M3_MAX, E4M3_MAX)
    a = y.abs()
    ulp = e4m3_ulp(a)
    lo = torch.floor(a / ulp) * ulp
    frac = (a - lo) / ulp
    q = torch.where(frac >= 0.5 + u, lo + ulp, lo).clamp(max=E4M3_MAX)
    return torch.sign(y) * q * s


def int4_g32(w: torch.Tensor) -> torch.Tensor:
    """Symmetric INT4 with group size 32 along the input dimension and bf16 scales, as Kimi K2.6 ships its experts."""
    r, c = w.shape
    g = w.float().view(r, c // 32, 32)
    sc = (g.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-12).to(torch.bfloat16).float()
    q = torch.round(g / sc).clamp(-8, 7)
    return (q * sc).view(r, c).to(torch.bfloat16).float()


def tie_fraction(w: torch.Tensor) -> float:
    """Fraction of weights exactly halfway between two E4M3 values under the current block scales."""
    y = w / block_scale(w)
    u = e4m3_ulp(y)
    return float(((y / u) - torch.floor(y / u) == 0.5).float().mean())
