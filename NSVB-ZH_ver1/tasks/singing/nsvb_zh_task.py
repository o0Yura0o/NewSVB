# -*- coding: utf-8 -*-
"""
NSVB-ZH training task.

Extends the original SVBVAEMleTask with:
    Stage 1 (unchanged CVAE) + optional GRL domain disentanglement
    Stage 2 replaced:
        - M : ResidualM (kernel=1 default)
        - D_z : conditional on soft register + discrete phoneme
        - D_mel : reused from Stage 1, low LR
        - L_PatchNCE + L_PPG + L_identity_pro
        - warmup: no D_z gradient to M for first `d_z_warmup_steps`

The original a2p alignment / L_map1 / gather warp pathways are bypassed.
Inference helpers live in `inference/nsvb_zh_inference.py`.

Hparams (new keys; see egs/datasets/audio/M4OpenSinger/*.yaml):
    task_stage              : 'stage1' | 'stage2'
    use_grl                 : bool
    grl_lambda_start        : float
    grl_lambda_end          : float
    grl_warmup_steps        : int
    m_kernel_size           : 1 | 3
    m_hidden_dim            : int
    m_num_layers            : int
    d_z_warmup_steps        : int
    lambda_adv_z            : float   (default 1.0)
    lambda_adv_mel          : float   (default 0.2)
    lambda_patchnce         : float   (default 1.0)
    lambda_ppg              : float   (default 0.5)
    lambda_identity_pro     : float   (default 0.1)
    identity_pro_prob       : float   (default 0.2)
    opt_m_lr                : float   (default 1e-4)
    opt_dz_lr               : float   (default 4e-4)
    opt_dmel_lr             : float   (default 1e-5)
    soft_bucket_num         : int     (default 5)
    soft_bucket_sigma       : float   (default 0.3)
    phoneme_vocab_size      : int     (match your zh txt_processor; pad=0)
    ppg_extractor_ckpt      : path to Chinese PPG extractor
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams
from tasks.singing.svb_vae_task import SVBVAEMleTask          # base task (existing)

from modules.voice_conversion import (
    ResidualM,
    DiscriminatorZ,
    DomainDiscriminator,
    PatchNCELoss,
    SoftRegisterEncoder,
    grad_reverse,
    hinge_d_loss,
    hinge_g_loss,
    domain_bce_loss,
)


class NsvbZhTask(SVBVAEMleTask):
    """Task for NSVB-ZH. Handles both Stage 1 and Stage 2 via `task_stage`."""

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    def build_model(self):
        super().build_model()                                   # builds fVAE / CVAE
        stage = hparams.get('task_stage', 'stage1')
        latent_dim = hparams.get('fvae_latent_size', 192)

        # ---- Stage 1 add-on: domain disentanglement via GRL -------------
        if hparams.get('use_grl', False):
            self.d_domain = DomainDiscriminator(
                latent_dim=latent_dim,
                hidden_dim=hparams.get('d_domain_hidden', 128),
            )
        else:
            self.d_domain = None

        # ---- Stage 2 add-ons -------------------------------------------
        if stage == 'stage2':
            self.m = ResidualM(
                latent_dim=latent_dim,
                hidden_dim=hparams.get('m_hidden_dim', 256),
                kernel_size=hparams.get('m_kernel_size', 1),
                num_layers=hparams.get('m_num_layers', 4),
            )
            self.d_z = DiscriminatorZ(
                latent_dim=latent_dim,
                soft_register_dim=hparams.get('soft_bucket_num', 5),
                phoneme_vocab_size=hparams.get('phoneme_vocab_size', 80),
                phoneme_embed_dim=hparams.get('phoneme_embed_dim', 32),
                hidden_dim=hparams.get('dz_hidden_dim', 256),
                num_layers=hparams.get('dz_num_layers', 4),
            )
            self.soft_register = SoftRegisterEncoder(
                num_buckets=hparams.get('soft_bucket_num', 5),
                sigma=hparams.get('soft_bucket_sigma', 0.3),
            )
            self.patchnce = PatchNCELoss(
                latent_dim=latent_dim,
                proj_dim=hparams.get('patchnce_proj_dim', 64),
                temperature=hparams.get('patchnce_temp', 0.07),
                num_patches=hparams.get('patchnce_num_patches', 128),
            )
            # D_mel is created by the base class; we reuse it.
            # Freeze CVAE encoder / decoder by default in Stage 2.
            if hparams.get('freeze_cvae_in_stage2', True):
                for p in self.model.fvae.parameters():
                    p.requires_grad = False
            # PPG extractor for L_PPG: reuse the existing VCASR if available.
            self._ppg_extractor = self._maybe_load_ppg()
        else:
            self.m = None
            self.d_z = None
            self.soft_register = None
            self.patchnce = None
            self._ppg_extractor = None

    def _maybe_load_ppg(self):
        """Load a Chinese PPG extractor. Falls back to None with a warning."""
        ckpt = hparams.get('ppg_extractor_ckpt', None)
        if not ckpt or not os.path.exists(ckpt):
            print("[nsvb_zh] WARN: no Chinese PPG extractor ckpt found; "
                  "L_PPG will be disabled.")
            return None
        from modules.voice_conversion.vc_modules import VCASR
        extractor = VCASR(hparams)
        state = torch.load(ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        extractor.load_state_dict(sd, strict=False)
        extractor.eval()
        for p in extractor.parameters():
            p.requires_grad = False
        return extractor

    # ---------------------------------------------------------------------
    # Optimizers (TTUR)
    # ---------------------------------------------------------------------
    def build_optimizer(self, model):
        stage = hparams.get('task_stage', 'stage1')
        if stage == 'stage1':
            return self._build_stage1_opt(model)
        return self._build_stage2_opt(model)

    def _build_stage1_opt(self, model):
        cvae_params = list(model.fvae.parameters()) if hasattr(model, 'fvae') \
            else list(model.parameters())
        opts = {
            'opt_cvae': torch.optim.Adam(
                [p for p in cvae_params if p.requires_grad],
                lr=hparams.get('lr', 2e-4), betas=(0.9, 0.98), eps=1e-9,
            ),
        }
        if self.d_domain is not None:
            opts['opt_d_domain'] = torch.optim.Adam(
                self.d_domain.parameters(),
                lr=hparams.get('opt_d_domain_lr', 2e-4),
                betas=(0.5, 0.999),
            )
        # D_mel optimiser created by parent (if used in Stage 1). Leave intact.
        return opts

    def _build_stage2_opt(self, model):
        opts = {
            'opt_m': torch.optim.Adam(
                self.m.parameters(),
                lr=hparams.get('opt_m_lr', 1e-4), betas=(0.5, 0.999),
            ),
            'opt_dz': torch.optim.Adam(
                self.d_z.parameters(),
                lr=hparams.get('opt_dz_lr', 4e-4), betas=(0.5, 0.999),
            ),
            'opt_patchnce': torch.optim.Adam(
                self.patchnce.parameters(),
                lr=hparams.get('opt_patchnce_lr', 1e-4), betas=(0.9, 0.98),
            ),
        }
        # D_mel: reuse the existing optimiser but override LR.
        if hasattr(self.model, 'discriminator') or hasattr(self, 'd_mel'):
            d_mel = getattr(self, 'd_mel', None) or self.model.discriminator
            opts['opt_dmel'] = torch.optim.Adam(
                d_mel.parameters(),
                lr=hparams.get('opt_dmel_lr', 1e-5), betas=(0.5, 0.999),
            )
        return opts

    # ---------------------------------------------------------------------
    # Schedules
    # ---------------------------------------------------------------------
    def _grl_lambda(self, step: int) -> float:
        """Linear warmup from lambda_start to lambda_end over grl_warmup_steps."""
        lo = hparams.get('grl_lambda_start', 0.1)
        hi = hparams.get('grl_lambda_end', 0.3)
        warm = hparams.get('grl_warmup_steps', 30000)
        if step >= warm:
            return hi
        return lo + (hi - lo) * (step / warm)

    def _d_z_active(self, step: int) -> bool:
        return step >= hparams.get('d_z_warmup_steps', 5000)

    # ---------------------------------------------------------------------
    # Stage 1 training step
    # ---------------------------------------------------------------------
    def _training_step_stage1(self, sample, batch_idx, opt_idx):
        """Stage 1: standard CVAE reconstruction + optional GRL."""
        if opt_idx == 0:                          # opt_cvae
            out = self.run_model(self.model, sample, return_z=True)
            loss_dict = out['loss_dict']

            if self.d_domain is not None and 'z' in out:
                z = out['z']                      # (B, C, T)
                mask = out.get('mel_mask', None)
                dataset_label = sample['dataset_label'].to(z.device)  # (B,) in {0,1}
                lam = self._grl_lambda(self.global_step)
                z_rev = grad_reverse(z, lam)
                logit = self.d_domain(z_rev, mask)
                loss_dict['grl_domain'] = domain_bce_loss(logit, dataset_label) * lam

            total = sum(v for v in loss_dict.values())
            return total, loss_dict

        elif opt_idx == 1 and self.d_domain is not None:  # opt_d_domain
            with torch.no_grad():
                out = self.run_model(self.model, sample, return_z=True)
            z = out['z'].detach()
            mask = out.get('mel_mask', None)
            dataset_label = sample['dataset_label'].to(z.device)
            logit = self.d_domain(z, mask)
            loss = domain_bce_loss(logit, dataset_label)
            return loss, {'d_domain_loss': loss}

        return None

    # ---------------------------------------------------------------------
    # Stage 2 training step
    # ---------------------------------------------------------------------
    def _training_step_stage2(self, sample, batch_idx, opt_idx):
        """Stage 2: ResidualM vs D_z with TTUR + content losses."""
        # Split batch into amateur (M4) and pro (OpenSinger) sides.
        # sample['dataset_label']: 0 = amateur, 1 = pro. Independent sampling
        # means each batch has roughly half-half; we just mask.
        lbl = sample['dataset_label']
        amateur_mask = (lbl == 0)
        pro_mask = (lbl == 1)

        if not amateur_mask.any() or not pro_mask.any():
            # Degenerate batch; skip.
            return None

        # --- Compute latents via frozen (or not) CVAE --------------------
        with torch.set_grad_enabled(not hparams.get('freeze_cvae_in_stage2', True)):
            enc_out = self.run_encoder(self.model, sample)
            z_all = enc_out['z']                       # (B, C, T)
            mel_mask = enc_out.get('mel_mask', None)   # (B, T)
            f0 = sample['f0']                          # (B, T)
            phoneme_ids = sample.get('phoneme_ids', None)  # (B, T) long
            if phoneme_ids is None:
                phoneme_ids = torch.zeros_like(f0, dtype=torch.long)

        z_a = z_all[amateur_mask]
        z_p = z_all[pro_mask]
        f0_a = f0[amateur_mask]; f0_p = f0[pro_mask]
        ph_a = phoneme_ids[amateur_mask]; ph_p = phoneme_ids[pro_mask]
        mask_a = mel_mask[amateur_mask] if mel_mask is not None else None
        mask_p = mel_mask[pro_mask] if mel_mask is not None else None

        sr_a = self.soft_register(f0_a)
        sr_p = self.soft_register(f0_p)

        # --- Optimiser 0: update M (generator) ---------------------------
        if opt_idx == 0:
            z_a_mapped = self.m(z_a)                   # (B_a, C, T)
            loss_dict = {}

            # (1) adversarial via D_z
            if self._d_z_active(self.global_step):
                fake_scores = self.d_z(z_a_mapped, sr_a, ph_a)
                loss_dict['G_adv_z'] = (hparams.get('lambda_adv_z', 1.0)
                                        * hinge_g_loss(fake_scores))
            else:
                loss_dict['G_adv_z'] = z_a_mapped.sum() * 0.0  # keep graph

            # (2) PatchNCE: preserve content between z_a and M(z_a)
            loss_dict['G_patchnce'] = (hparams.get('lambda_patchnce', 1.0)
                                       * self.patchnce(z_a, z_a_mapped, mask_a))

            # (3) L_PPG: phonetic posterior-gram preservation
            if self._ppg_extractor is not None and hparams.get('lambda_ppg', 0.5) > 0:
                mel_a = self.decode_mel(z_a, f0_a)
                mel_a_mapped = self.decode_mel(z_a_mapped, f0_a)
                with torch.no_grad():
                    ppg_src = self._ppg_extractor(mel_a)
                ppg_tgt = self._ppg_extractor(mel_a_mapped)
                T = min(ppg_src.size(1), ppg_tgt.size(1))
                loss_dict['G_ppg'] = (hparams.get('lambda_ppg', 0.5)
                                      * F.l1_loss(ppg_tgt[:, :T], ppg_src[:, :T]))

            # (4) adversarial via D_mel (reused from Stage 1)
            if hparams.get('lambda_adv_mel', 0.2) > 0 and hasattr(self, 'd_mel'):
                mel_a_mapped = self.decode_mel(z_a_mapped, f0_a)
                d_mel_score = self.d_mel(mel_a_mapped)
                loss_dict['G_adv_mel'] = (hparams.get('lambda_adv_mel', 0.2)
                                          * hinge_g_loss(d_mel_score))

            # (5) L_identity_pro: 20 % of batches, require M(z_p) ≈ z_p
            if (hparams.get('lambda_identity_pro', 0.1) > 0 and
                    torch.rand(1).item() < hparams.get('identity_pro_prob', 0.2)):
                z_p_mapped = self.m(z_p)
                loss_dict['G_identity_pro'] = (hparams.get('lambda_identity_pro', 0.1)
                                               * F.l1_loss(z_p_mapped, z_p))

            # (6) ||Δ|| monitoring — not a loss, just logged
            with torch.no_grad():
                delta = self.m.delta_only(z_a)
                loss_dict['_delta_over_z'] = (delta.norm() / (z_a.norm() + 1e-6)).detach()

            total = sum(v for k, v in loss_dict.items() if not k.startswith('_'))
            return total, loss_dict

        # --- Optimiser 1: update D_z (critic on latent) ------------------
        if opt_idx == 1:
            if not self._d_z_active(self.global_step):
                return None
            with torch.no_grad():
                z_a_mapped = self.m(z_a)
            real_scores = self.d_z(z_p, sr_p, ph_p)
            fake_scores = self.d_z(z_a_mapped.detach(), sr_a, ph_a)
            loss = hinge_d_loss(real_scores, fake_scores)
            return loss, {'D_z_loss': loss}

        # --- Optimiser 2: update PatchNCE projection head ----------------
        if opt_idx == 2:
            # Already optimised alongside M via shared loss; this slot is used
            # so the Lightning-style step counter stays consistent. Re-compute
            # a small L2 to keep optimiser state warm.
            z_a_mapped = self.m(z_a).detach()
            loss = self.patchnce(z_a, z_a_mapped, mask_a) * 0.1
            return loss, {'patchnce_warm': loss}

        # --- Optimiser 3: D_mel ------------------------------------------
        if opt_idx == 3 and hparams.get('lambda_adv_mel', 0.2) > 0 and hasattr(self, 'd_mel'):
            with torch.no_grad():
                z_a_mapped = self.m(z_a)
                mel_a_mapped = self.decode_mel(z_a_mapped, f0_a)
                mel_p = self.decode_mel(z_p, f0_p)
            real_s = self.d_mel(mel_p)
            fake_s = self.d_mel(mel_a_mapped.detach())
            loss = hinge_d_loss(real_s, fake_s)
            return loss, {'D_mel_loss': loss}

        return None

    # ---------------------------------------------------------------------
    # Dispatch
    # ---------------------------------------------------------------------
    def training_step(self, sample, batch_idx, opt_idx=0):
        stage = hparams.get('task_stage', 'stage1')
        if stage == 'stage1':
            return self._training_step_stage1(sample, batch_idx, opt_idx)
        return self._training_step_stage2(sample, batch_idx, opt_idx)

    # ---------------------------------------------------------------------
    # Encoder / decoder helpers (to be adapted to your fork's API)
    # ---------------------------------------------------------------------
    def run_encoder(self, model, sample):
        """Run CVAE encoder, return dict with 'z' and 'mel_mask'.

        YOU MAY NEED TO ADJUST the call signature for your fork of NSVB.
        The original SVBVAE forward returns more; we only need z and mask.
        """
        mel = sample['mels']                       # (B, T, M)
        f0 = sample['f0']
        mel_mask = (mel.abs().sum(-1) > 0).float()
        z = model.fvae.encoder(mel.transpose(1, 2))
        if isinstance(z, tuple):
            z = z[0]
        return {'z': z, 'mel_mask': mel_mask}

    def decode_mel(self, z: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Decode mel from latent + F0 through the (frozen) CVAE decoder."""
        return self.model.fvae.decoder(z, f0=f0)

    def run_model(self, model, sample, return_z: bool = False):
        """Stage 1 forward. Delegates to parent when possible."""
        out = super().run_model(model, sample) if hasattr(super(), 'run_model') else {}
        loss_dict = out if isinstance(out, dict) else {'loss': out}
        if return_z:
            enc = self.run_encoder(model, sample)
            return {
                'loss_dict': loss_dict,
                'z': enc['z'],
                'mel_mask': enc['mel_mask'],
            }
        return loss_dict

    # ---------------------------------------------------------------------
    # Validation — keep simple: forward through current stage, log.
    # ---------------------------------------------------------------------
    def validation_step(self, sample, batch_idx):
        stage = hparams.get('task_stage', 'stage1')
        out = {}
        if stage == 'stage2':
            enc = self.run_encoder(self.model, sample)
            z = enc['z']
            z_mapped = self.m(z)
            out['val_delta_ratio'] = (self.m.delta_only(z).norm()
                                      / (z.norm() + 1e-6)).detach()
        return super().validation_step(sample, batch_idx) if \
            hasattr(super(), 'validation_step') else out
