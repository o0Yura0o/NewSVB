# -*- coding: utf-8 -*-
"""
NSVB-ZH core modules.

Components
----------
ResidualM         : residual mapping function f(z) = z + Δ(z), default kernel=1.
                    Warp-invariance comes from pointwise conv; expressivity
                    is bought back by kernel=3 via hparam if ||Δ|| stays too
                    small during Stage 2.
DiscriminatorZ    : conditional discriminator on latent z, conditioned on
                    soft F0 register + discrete phoneme id. Used in Stage 2
                    for the adversarial signal that drives M.
DomainDiscriminator : classifier "M4 vs OpenSinger" attached via GRL to φ.
                    Only active during Stage 1 when probe places us in Case B.
                    In Case A, set grl_lambda to 0.
GradientReversal  : standard DANN-style gradient reversal layer.

Design rationale
----------------
- All discriminators take a *time-wise* view of z, so spectral-normalised
  1-D convs with receptive field ~5 give a PatchGAN-style critic.
- D_z is strictly conditional: the condition is concatenated on the channel
  axis before the first conv, not added via projection, so the condition
  participates in every conv receptive field.
- The projection discriminator trick (Miyato 2018) is avoided because the
  soft register is continuous, not a class id.
- All Δ and D heads use spectral normalisation; M does not, since the residual
  path already regularises magnitude.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm

from modules.voice_conversion.soft_bucket import SoftRegisterEncoder


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (DANN)
# ---------------------------------------------------------------------------
class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Residual Mapping Function M: z -> z + Δ(z)
# ---------------------------------------------------------------------------
class ResidualM(nn.Module):
    """Residual mapping function.

    Args:
        latent_dim: channel dim of z, typically 192 in NSVB fVAE.
        hidden_dim: inner channel for Δ.
        kernel_size: 1 (default, pointwise) or 3 (expanded receptive field).
                     kernel=1 preserves alignment from CVAE exactly (warp-
                     invariant); kernel=3 gives more expressivity but risks
                     drifting away from the CVAE manifold.
        num_layers: depth of Δ network.
        norm: 'group' or 'layer'.
    """

    def __init__(self,
                 latent_dim: int = 192,
                 hidden_dim: int = 256,
                 kernel_size: int = 1,
                 num_layers: int = 4,
                 norm: str = 'group',
                 init_delta_scale: float = 1e-2):
        super().__init__()
        assert kernel_size in (1, 3), "kernel_size must be 1 or 3"
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        layers = []
        in_ch = latent_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding))
            if i < num_layers - 1:
                if norm == 'group':
                    groups = min(8, out_ch)
                    layers.append(nn.GroupNorm(groups, out_ch))
                elif norm == 'layer':
                    # Apply LayerNorm over channel dim
                    layers.append(_ChannelLayerNorm(out_ch))
                layers.append(nn.GELU())
            in_ch = out_ch

        self.delta = nn.Sequential(*layers)

        # Initialise the last conv to near-zero so M ≈ identity at start.
        with torch.no_grad():
            last_conv = self.delta[-1]
            nn.init.normal_(last_conv.weight, mean=0.0, std=init_delta_scale)
            nn.init.zeros_(last_conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, T). Adds residual Δ(z)."""
        return z + self.delta(z)

    def delta_only(self, z: torch.Tensor) -> torch.Tensor:
        """Return Δ(z) alone, for monitoring ||Δ|| / ||z||."""
        return self.delta(z)


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dim for 1-D conv tensors (B, C, T)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


# ---------------------------------------------------------------------------
# Conditional Discriminator D_z (on latent)
# ---------------------------------------------------------------------------
class DiscriminatorZ(nn.Module):
    """Conditional discriminator on latent z.

    Condition: soft register (K dim, default 5) + phoneme embedding (P dim).
    We concatenate the condition along the channel axis of z. Phoneme id is
    embedded via a learnable table; unvoiced/pad frames use id=0.

    Args:
        latent_dim: channel dim of z.
        soft_register_dim: K, output of SoftRegisterEncoder.
        phoneme_vocab_size: number of discrete phoneme classes (incl. pad=0).
        phoneme_embed_dim: embedding dim for phoneme.
        hidden_dim: critic hidden channels.
        num_layers: depth.
        kernel_size: receptive field; 5 gives local patch view.
    """

    def __init__(self,
                 latent_dim: int = 192,
                 soft_register_dim: int = 5,
                 phoneme_vocab_size: int = 80,
                 phoneme_embed_dim: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 kernel_size: int = 5):
        super().__init__()
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, phoneme_embed_dim,
                                          padding_idx=0)
        cond_dim = soft_register_dim + phoneme_embed_dim
        padding = kernel_size // 2

        layers = []
        in_ch = latent_dim + cond_dim
        for i in range(num_layers):
            out_ch = hidden_dim
            layers.append(spectral_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        # Output head: per-frame real/fake score
        layers.append(spectral_norm(
            nn.Conv1d(in_ch, 1, kernel_size=1)
        ))
        self.net = nn.Sequential(*layers)

    def forward(self,
                z: torch.Tensor,
                soft_register: torch.Tensor,
                phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (B, C_z, T)
            soft_register: (B, T, K)
            phoneme_ids: (B, T) long

        Returns:
            (B, 1, T) per-frame critic score. Loss aggregates over T.
        """
        ph_emb = self.phoneme_embed(phoneme_ids)              # (B, T, P)
        cond = torch.cat([soft_register, ph_emb], dim=-1)     # (B, T, K+P)
        cond = cond.transpose(1, 2)                            # (B, K+P, T)
        x = torch.cat([z, cond], dim=1)                        # (B, C_z+K+P, T)
        return self.net(x)


# ---------------------------------------------------------------------------
# Domain Discriminator (Stage 1, GRL-attached)
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    """Binary dataset classifier M4 vs OpenSinger, attached via GRL to φ.

    Pools over time (mean + std), two MLP layers, binary logit out.
    GRL is applied externally so this class stays a plain classifier.
    """

    def __init__(self, latent_dim: int = 192, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            z: (B, C, T)
            mask: (B, T) float mask, 1=valid, 0=pad. If None, no masking.

        Returns:
            (B,) raw logit.
        """
        if mask is not None:
            mask_c = mask.unsqueeze(1)                        # (B, 1, T)
            valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            mean = (z * mask_c).sum(dim=2) / valid            # (B, C)
            var = ((z - mean.unsqueeze(-1)) ** 2 * mask_c).sum(dim=2) / valid
            std = torch.sqrt(var + 1e-6)
        else:
            mean = z.mean(dim=2)
            std = z.std(dim=2)
        feat = torch.cat([mean, std], dim=1)                  # (B, 2C)
        return self.mlp(feat).squeeze(-1)                      # (B,)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def hinge_d_loss(real_scores: torch.Tensor,
                 fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss for D. Inputs are per-frame scores (B, 1, T)."""
    loss_real = F.relu(1.0 - real_scores).mean()
    loss_fake = F.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake


def hinge_g_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss for G (M). Non-saturating form."""
    return -fake_scores.mean()


def domain_bce_loss(logits: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
    """BCE for binary domain classifier. labels: (B,) in {0, 1}."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


# ---------------------------------------------------------------------------
# PatchNCE loss (for content preservation through M)
# ---------------------------------------------------------------------------
class PatchNCELoss(nn.Module):
    """PatchNCE between z and M(z): treat same time positions as positive pairs.

    Following Park et al. 2020 (CUT). Uses a learnable 2-layer MLP projection
    head, negatives are other positions within the same batch.
    """

    def __init__(self, latent_dim: int = 192, proj_dim: int = 64,
                 temperature: float = 0.07, num_patches: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, proj_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(proj_dim, proj_dim, kernel_size=1),
        )
        self.temperature = temperature
        self.num_patches = num_patches

    def _sample(self, z_proj: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample num_patches valid frames per batch item.

        Args:
            z_proj: (B, D, T)
            mask: (B, T) or None
        Returns:
            feats: (B*N, D), idx: (B, N) for re-using on paired tensor
        """
        B, D, T = z_proj.shape
        N = min(self.num_patches, T)
        if mask is None:
            idx = torch.randint(0, T, (B, N), device=z_proj.device)
        else:
            idx = []
            for b in range(B):
                valid = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
                if valid.numel() == 0:
                    idx.append(torch.randint(0, T, (N,), device=z_proj.device))
                else:
                    pick = valid[torch.randint(0, valid.numel(),
                                               (N,), device=z_proj.device)]
                    idx.append(pick)
            idx = torch.stack(idx, dim=0)                       # (B, N)
        # gather
        idx_exp = idx.unsqueeze(1).expand(-1, D, -1)           # (B, D, N)
        feats = torch.gather(z_proj, 2, idx_exp)               # (B, D, N)
        feats = feats.permute(0, 2, 1).reshape(B * N, D)        # (B*N, D)
        feats = F.normalize(feats, dim=-1)
        return feats, idx

    def forward(self, z: torch.Tensor, z_mapped: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            z: (B, C, T) original
            z_mapped: (B, C, T) = M(z)
            mask: (B, T) or None
        """
        B = z.size(0)
        proj_src = self.proj(z)                                # query
        proj_tgt = self.proj(z_mapped)                         # key

        q, idx = self._sample(proj_src, mask)                  # (B*N, D)
        D = proj_tgt.size(1)
        N = idx.size(1)
        idx_exp = idx.unsqueeze(1).expand(-1, D, -1)
        k = torch.gather(proj_tgt, 2, idx_exp).permute(0, 2, 1).reshape(B * N, D)
        k = F.normalize(k, dim=-1)

        # Similarity within the same batch item: positives on the diagonal,
        # negatives are other sampled positions from the same clip.
        q = q.view(B, N, D)
        k = k.view(B, N, D)
        logits = torch.einsum('bnd,bmd->bnm', q, k) / self.temperature  # (B, N, N)
        targets = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        loss = F.cross_entropy(logits.reshape(B * N, N), targets.reshape(-1))
        return loss
