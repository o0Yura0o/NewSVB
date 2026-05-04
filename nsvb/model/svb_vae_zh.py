"""
nsvb/model/svb_vae_zh.py
==========================

【這支檔案做什麼】
NSVB-ZH 的 Stage 1 model：CVAE（FVAE backbone）+ 條件構造模組。

外部介面：
    SVBVAEZh(num_mels, ppg_dim, spk_emb_dim, latent_size, hidden_size, ...)
    .forward(mel, ppg, f0, spk_emb, mel_mask) → dict {
        mel_recon, kl, m_q, logs_q
    }

【為什麼把條件構造從 NSVB FastSpeech2VAE 拆出來】
原 NSVB 用 FastSpeech2 把 (text, duration, pitch) 編碼成 hidden state 餵 FVAE
作為條件。我們不走文字路徑：
    - PPG (Whisper hidden state) 替代 FastSpeech2 phoneme encoder
    - F0 直接編碼（log-Hz embedding）替代 FastSpeech2 pitch predictor
    - spk_emb (Resemblyzer) 替代 FastSpeech2 speaker embedding 表
所以 SVBVAEZh 自己負責把這幾個來源拼成 g [B, gin_channels, T_mel] 餵 FVAE。

【條件 g 的構造】
    pitch_emb     = LogF0Embed(f0)                  → [B, T, pitch_emb_dim]
    ppg_proj      = Linear(ppg)                      → [B, T, ppg_proj_dim]
    spk_proj      = Linear(spk_emb).expand(T)        → [B, T, spk_proj_dim]
    g_BTH         = concat(pitch, ppg_proj, spk)     → [B, T, gin_channels]
    g             = g_BTH.transpose(1, 2)            → [B, gin_channels, T]

預設維度配置（總和 = 256，**對齊 NSVB 1030_vae_mle ckpt** 的 gin_channels=256
讓 transfer learning 可行）：
    pitch_emb_dim = 32      （F0 用 log-Hz linear projection 即可，不需高維）
    ppg_proj_dim  = 128     （PPG 是最大宗資訊源，給最多容量）
    spk_proj_dim  = 96      （speaker 較大 channel 提升音色錨穩定性）

【為什麼把 F0 用 LogF0Embed 而非餵 raw Hz】
- raw Hz 範圍 [0, 1100] 動態大、unvoiced=0 與 voiced 數值跨度差數十倍
- log-Hz 把半音差變成定值（每 +1 semitone = 固定 log delta）
- unvoiced 用 mask + bias，避免影響 voiced 的 log embedding

【為什麼 spk_emb 用 expand 而非 attention conditioning】
- Speaker identity 在一首歌內不變，attention 浪費；單向量 broadcast 已是常見做法
- 與 NSVB 原版 use_spk_embed 一致

【為什麼 mel 在輸入 FVAE 前 transpose】
- FVAE 期望 [B, num_mels, T_mel]（channel-first）
- 我們 dataset 給的是 [B, T_mel, num_mels]（time-first）
- 在 model forward 開頭 transpose，內部全走 channel-first，輸出再轉回 time-first
"""

from typing import Dict

import torch
import torch.nn as nn

from nsvb.backbone.fvae import FVAE


class LogF0Embed(nn.Module):
    """
    F0 (Hz) → log-Hz linear embedding。

    為什麼這樣設計：
        F0 = 0 (unvoiced) 與 voiced 範圍 [80, 1000] Hz 的 log 差距巨大；
        直接 log(f0+eps) 會給 unvoiced 一個遠離 voiced 的 sentinel 值，
        embedding 學到「unvoiced 自有空間」。
    """

    def __init__(self, embed_dim: int = 32, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Linear from 1D (log-Hz) to embed_dim
        # 為什麼 1D in：F0 是 scalar per frame，不需要先 bin
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: [B, T] Hz (unvoiced=0)
        Returns:
            emb: [B, T, embed_dim]
        """
        log_f0 = torch.log2(f0 + self.eps).unsqueeze(-1)  # [B, T, 1]
        return self.linear(log_f0)


class ConditionBuilder(nn.Module):
    """
    把 (PPG, F0, spk_emb) 拼成 FVAE 的 condition g。
    """

    def __init__(self,
                 ppg_dim: int,
                 spk_emb_dim: int = 256,
                 ppg_proj_dim: int = 128,
                 pitch_emb_dim: int = 32,
                 spk_proj_dim: int = 96):
        super().__init__()
        # 為什麼要記下 gin_channels：FVAE 構造時要 explicit 傳這個值
        self.gin_channels = ppg_proj_dim + pitch_emb_dim + spk_proj_dim

        self.ppg_proj = nn.Linear(ppg_dim, ppg_proj_dim)
        self.pitch_emb = LogF0Embed(embed_dim=pitch_emb_dim)
        self.spk_proj = nn.Linear(spk_emb_dim, spk_proj_dim)

    def forward(
        self,
        ppg: torch.Tensor,        # [B, T, ppg_dim]
        f0: torch.Tensor,          # [B, T]
        spk_emb: torch.Tensor,    # [B, spk_emb_dim]
    ) -> torch.Tensor:
        """
        Returns:
            g: [B, gin_channels, T]   channel-first (FVAE 期望)
        """
        T = ppg.shape[1]
        ppg_e = self.ppg_proj(ppg)                      # [B, T, ppg_proj]
        pitch_e = self.pitch_emb(f0)                    # [B, T, pitch_emb]
        spk_e = self.spk_proj(spk_emb)                  # [B, spk_proj]
        # broadcast spk 到 time 軸
        spk_e = spk_e.unsqueeze(1).expand(-1, T, -1)    # [B, T, spk_proj]

        g_BTH = torch.cat([ppg_e, pitch_e, spk_e], dim=-1)  # [B, T, gin]
        return g_BTH.transpose(1, 2)                        # [B, gin, T]


class SVBVAEZh(nn.Module):
    """
    Stage 1 NSVB-ZH model：condition (PPG + F0 + spk_emb) + FVAE。

    Args:
        num_mels:         mel 通道數（80）
        ppg_dim:          PPG 通道數（whisper-large-v3=1280, whisper-tiny=384）
        spk_emb_dim:      Resemblyzer embedding dim（256）
        latent_size:      z 通道數（128, 與 NSVB 一致）
        hidden_size:      FVAE 內部 hidden（192, 與 NSVB 一致）
        kernel_size:      FVAE WN dilated conv kernel（5）
        enc_n_layers:     encoder WN 層數（8, 與 NSVB 一致）
        dec_n_layers:     decoder WN 層數（4, 與 NSVB 一致）
        latent_strides:   FVAE 下採 stride tuple，預設 (8,) 對齊 hop=128 / mel172fps → latent 21fps
        ppg_proj_dim / pitch_emb_dim / spk_proj_dim:  條件子模組通道分配
    """

    def __init__(
        self,
        num_mels: int = 80,
        ppg_dim: int = 1280,
        spk_emb_dim: int = 256,
        latent_size: int = 128,
        hidden_size: int = 192,
        kernel_size: int = 5,
        enc_n_layers: int = 8,
        dec_n_layers: int = 4,
        latent_strides=(4,),
        ppg_proj_dim: int = 128,
        pitch_emb_dim: int = 32,
        spk_proj_dim: int = 96,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_mels = num_mels
        self.condition = ConditionBuilder(
            ppg_dim=ppg_dim, spk_emb_dim=spk_emb_dim,
            ppg_proj_dim=ppg_proj_dim, pitch_emb_dim=pitch_emb_dim,
            spk_proj_dim=spk_proj_dim,
        )
        self.fvae = FVAE(
            in_out_channels=num_mels,
            hidden_channels=hidden_size,
            latent_size=latent_size,
            kernel_size=kernel_size,
            enc_n_layers=enc_n_layers,
            dec_n_layers=dec_n_layers,
            gin_channels=self.condition.gin_channels,
            strides=latent_strides,
            p_dropout=p_dropout,
        )

    def forward(
        self,
        mel: torch.Tensor,        # [B, T, num_mels]
        ppg: torch.Tensor,         # [B, T, ppg_dim]
        f0: torch.Tensor,          # [B, T]
        spk_emb: torch.Tensor,    # [B, spk_emb_dim]
        mel_mask: torch.Tensor,    # [B, T] (1=valid, 0=pad)
        infer: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            mel_recon: [B, T, num_mels]      (training+infer)
            kl:        scalar                  (training only)
            m_q:       [B, latent, T_z]       posterior mean (training only)
            logs_q:    [B, latent, T_z]       posterior log-std (training only)
            z_p:       [B, latent, T_z]       prior sample (infer only)
        """
        # 條件 g
        g = self.condition(ppg, f0, spk_emb)  # [B, gin, T]

        # FVAE 期望 channel-first input/mask
        x = mel.transpose(1, 2)               # [B, num_mels, T]
        x_mask = mel_mask.unsqueeze(1)        # [B, 1, T]

        if not infer:
            x_recon, kl, _z_p, m_q, logs_q = self.fvae(
                x=x, x_mask=x_mask, g=g, infer=False,
            )
            return {
                "mel_recon": x_recon.transpose(1, 2),  # [B, T, num_mels]
                "kl": kl,
                "m_q": m_q,
                "logs_q": logs_q,
            }
        else:
            x_recon, z_p = self.fvae(g=g, infer=True)
            return {
                "mel_recon": x_recon.transpose(1, 2),
                "z_p": z_p,
            }


if __name__ == "__main__":
    # 自我測試 shape
    B, T, NUM_MELS, PPG_DIM, SPK = 2, 320, 80, 1280, 256

    model = SVBVAEZh(
        num_mels=NUM_MELS, ppg_dim=PPG_DIM, spk_emb_dim=SPK,
        latent_size=128, hidden_size=192,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SVBVAEZh params: {n_params/1e6:.2f}M")
    print(f"  condition.gin_channels = {model.condition.gin_channels}  (expect 256)")

    mel = torch.randn(B, T, NUM_MELS)
    ppg = torch.randn(B, T, PPG_DIM)
    f0 = torch.rand(B, T) * 500 + 100  # 100~600 Hz
    spk_emb = torch.randn(B, SPK)
    mel_mask = torch.ones(B, T)

    out = model(mel, ppg, f0, spk_emb, mel_mask, infer=False)
    print(f"  mel_recon: {out['mel_recon'].shape}  (expect [{B},{T},{NUM_MELS}])")
    print(f"  kl: {out['kl'].item():.4f}")
    print(f"  m_q: {out['m_q'].shape}  (expect [{B},128,{T // 4}])")

    # 推理路徑
    model.eval()
    with torch.no_grad():
        out_inf = model(mel, ppg, f0, spk_emb, mel_mask, infer=True)
    print(f"  infer mel: {out_inf['mel_recon'].shape}")