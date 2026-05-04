# -*- coding: utf-8 -*-
"""
Soft F0 bucketing for conditional discriminator D_z.

Motivation
----------
If D_z is conditioned on exact F0 values, the mapping function M can game the
discriminator by matching F0 statistics exactly (F0 shortcut). If D_z is
conditioned on a hard pitch bucket (e.g., one-hot of 5 registers), boundary
artifacts appear and gradients vanish near bin edges.

A soft Gaussian bucketing in log-Hz space gives:
  - smooth gradients across bin boundaries
  - identity-preserving representation (log-Hz is the natural perceptual space)
  - fixed-dimensional conditioning vector independent of F0 resolution

The 5 registers roughly correspond to (in C-major terms):
  0: below C3  (~130 Hz)         — low chest
  1: C3 -- C4  (~130 -- 260 Hz)  — chest
  2: C4 -- C5  (~260 -- 520 Hz)  — mix
  3: C5 -- C6  (~520 -- 1040 Hz) — head
  4: above C6 (~1040 Hz+)        — whistle

Centers are set in log-Hz with σ=0.3 log-Hz (~ a perfect fifth spread), which
gives smooth overlap between adjacent registers.
"""
import numpy as np
import torch


# Default register centers in log-Hz (natural log).
# log(65.4) ≈ 4.18  (C2, below-chest reference)
# log(130.8) ≈ 4.87 (C3)
# log(261.6) ≈ 5.57 (C4)
# log(523.2) ≈ 6.26 (C5)
# log(1046.5) ≈ 6.95 (C6)
DEFAULT_REGISTER_CENTERS_LOGHZ = np.array(
    [4.18, 4.87, 5.57, 6.26, 6.95], dtype=np.float32
)
DEFAULT_NUM_BUCKETS = 5
DEFAULT_SIGMA_LOGHZ = 0.3


def f0_to_soft_register(
    f0: torch.Tensor,
    centers_loghz: torch.Tensor = None,
    sigma: float = DEFAULT_SIGMA_LOGHZ,
    f0_min: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert F0 (Hz) to soft register vector.

    Args:
        f0: (B, T) F0 in Hz. Unvoiced frames should be ≤ 0 (will be zeroed).
        centers_loghz: (K,) register centers in log-Hz. If None, uses defaults.
        sigma: Gaussian bandwidth in log-Hz.
        f0_min: minimum valid F0. Frames below this are treated as unvoiced.
        eps: numerical floor for log().

    Returns:
        (B, T, K) soft register vector, sums to 1 on voiced frames, all zero
        on unvoiced frames. Suitable for concatenation with z as condition.
    """
    if centers_loghz is None:
        centers_loghz = torch.as_tensor(
            DEFAULT_REGISTER_CENTERS_LOGHZ, dtype=f0.dtype, device=f0.device
        )
    else:
        centers_loghz = centers_loghz.to(device=f0.device, dtype=f0.dtype)

    voiced_mask = (f0 >= f0_min).float()          # (B, T)
    f0_safe = torch.clamp(f0, min=eps)
    log_f0 = torch.log(f0_safe)                   # (B, T)

    # (B, T, 1) - (K,) -> (B, T, K)
    diff = log_f0.unsqueeze(-1) - centers_loghz   # (B, T, K)
    logits = -0.5 * (diff / sigma) ** 2
    soft = torch.softmax(logits, dim=-1)          # (B, T, K), sums to 1

    # Zero out unvoiced frames so the conditioner gets "no register info".
    soft = soft * voiced_mask.unsqueeze(-1)
    return soft


def build_register_centers(num_buckets: int = DEFAULT_NUM_BUCKETS,
                           f0_lo: float = 65.4,
                           f0_hi: float = 1046.5) -> np.ndarray:
    """Build evenly-spaced register centers in log-Hz.

    Args:
        num_buckets: number of registers.
        f0_lo / f0_hi: range endpoints in Hz.

    Returns:
        (num_buckets,) log-Hz centers.
    """
    return np.linspace(np.log(f0_lo), np.log(f0_hi), num_buckets).astype(np.float32)


class SoftRegisterEncoder(torch.nn.Module):
    """Wrap soft bucketing as an nn.Module for use inside discriminators.

    Non-learnable by default (centers are buffers), but supports learnable
    centers via `learnable=True` for late-stage fine-tuning.
    """

    def __init__(self,
                 num_buckets: int = DEFAULT_NUM_BUCKETS,
                 sigma: float = DEFAULT_SIGMA_LOGHZ,
                 f0_lo: float = 65.4,
                 f0_hi: float = 1046.5,
                 f0_min: float = 50.0,
                 learnable: bool = False):
        super().__init__()
        self.num_buckets = num_buckets
        self.sigma = sigma
        self.f0_min = f0_min

        centers = torch.as_tensor(
            build_register_centers(num_buckets, f0_lo, f0_hi), dtype=torch.float32
        )
        if learnable:
            self.centers_loghz = torch.nn.Parameter(centers)
        else:
            self.register_buffer("centers_loghz", centers)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """(B, T) Hz -> (B, T, K) soft vector."""
        return f0_to_soft_register(
            f0, centers_loghz=self.centers_loghz,
            sigma=self.sigma, f0_min=self.f0_min,
        )
