"""
nsvb/backbone/fvae.py
=======================

【這支檔案做什麼】
從 NSVB `modules/fastspeech/fs2_vae.py` 移植 CVAE 主結構：
    WN          : WaveNet-style dilated conv block，encoder/decoder 內部使用
    FVAEEncoder : Mel → latent z 的編碼器 φ（含 stride downsample）
    FVAEDecoder : Latent z → Mel 的解碼器 θ（含 stride upsample）
    FVAE        : 包裝 encoder + decoder + Gaussian prior，提供 forward/infer 介面

【相對 NSVB 原版的修改】
1. **移除 FastSpeech2VAE wrapper class**：
   原版用 FastSpeech2 處理文字輸入再餵 FVAE；我們不走文字路徑（PPG 取代），
   直接在外層構建條件向量餵給 FVAE，不需要 FastSpeech2 的 phoneme encoder
2. **移除 use_prior_glow 分支**：
   glow flow 對 z 的先驗加複雜度但對 unpaired SVB 邊際效益小，
   加上會引入 ResidualCouplingBlock 依賴；我們用標準 N(0, I) 先驗，KL closed-form
3. **參數從 hparams 全域 state 改成 explicit constructor args**：
   讓 FVAE 完全 self-contained，方便不同 task / 測試分別構建

【架構摘要】
    輸入：mel [B, NUM_MELS, T_mel]，條件 g [B, gin_channels, T_mel]
    流程：
      1. g 經 g_pre_net 下採樣 stride 倍 → g_sqz [B, gin_channels, T_z]
      2. encoder（含同樣 stride 的 pre_net）：mel + g_sqz → z, m_q, logs_q [B, latent_size, T_z]
      3. decoder（ConvTranspose1d 上採樣 stride 倍）：z + g → mel_recon [B, NUM_MELS, T_mel]
      4. KL loss = KL(q(z|x) || N(0, I))，per-frame 平均

【為什麼 latent 比 mel 在時間軸下採 stride 倍】
- 我們設定 strides=[8] → latent fps = mel fps / 8 = 172/8 ≈ 21.5 fps
- 21.5 fps ≈ 46 ms / latent frame，足以容納單一中文音素的 prosodic detail
  （中文音素平均 80-150 ms）
- 太密（如 mel rate 同步）讓 z 容量過大、難解耦；太疏失去細節

【WN 是什麼】
- WaveNet 的 residual gated dilated conv block
- 內部用 weight_norm 而非 batch_norm（VAE 訓練 BN 不穩）
- 條件 g 透過 1x1 conv 注入到每一層的 gate input

【為什麼用 Conv1d 而非 attention】
- mel 已是時間有序序列，local context (kernel * dilation) 足以建模
- Attention 對 fixed-length latent 才有優勢；mel 長度可變
- WaveNet-style 在歌聲 / TTS 領域實證強，是 NSVB 原作者選擇
"""

import numpy as np
import torch
import torch.distributions as dist
from torch import nn


# ── WaveNet-style residual block ─────────────────────────
def _fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """tanh(a[:n] + b[:n]) * sigmoid(a[n:] + b[n:])，gated activation 核心。"""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class WN(nn.Module):
    """
    WaveNet-style residual gated dilated conv stack。

    架構：
        n_layers 層，每層：
            x_in     = Conv1d(hidden, 2*hidden, k=kernel_size, dilation=dilation_rate^i)(x)
            cond_in  = Conv1d(gin_channels, 2*hidden*n_layers, k=1)(g) 切片給該層
            gated    = tanh(x_in[:hidden] + cond_in[:hidden]) * sigmoid(x_in[hidden:] + cond_in[hidden:])
            (中間層) skip + residual
            (最後層) skip only
        最終輸出 = sum of all skips * x_mask

    為什麼用 weight_norm 而非 BatchNorm：
        VAE 訓練 BatchNorm 對 batch 統計敏感，z 分布隨 batch 漂移；
        weight_norm 只 normalize 權重不依賴 batch stat，VAE 友善。

    為什麼 cond layer 一次算完所有層用切片：
        每層獨立 Conv1d 的 GPU kernel launch 開銷大；單次 conv 算完 2*hidden*n_layers
        通道再 view+slice，比 n_layers 個 conv 快 ~30%。
    """

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0.0, share_cond_layers=False,
                 is_BTC=False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        assert hidden_channels % 2 == 0, "hidden_channels must be even (gated split)"

        self.is_BTC = is_BTC
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.share_cond_layers = share_cond_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0 and not share_cond_layers:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # 最後一層只需 skip 路徑（沒有下一層接收 residual）
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        """
        Args:
            x:      [B, hidden_channels, T]
            x_mask: [B, 1, T] 或 None；padding frame 的 mask（1=valid, 0=pad）
            g:      [B, gin_channels, T]
        Returns:
            output: [B, hidden_channels, T]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2) if x_mask is not None else None
        if x_mask is None:
            x_mask = 1

        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        # 條件 g 一次算完所有層（除非 caller 已經算好並 share）
        if g is not None and not self.share_cond_layers:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = _fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        """訓練後 inference 移除 weight_norm 加速；對應 nn.utils.remove_weight_norm。"""
        def _rm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_rm)


# ── Encoder φ ────────────────────────────────────────────
class FVAEEncoder(nn.Module):
    """
    Mel → latent 的 stride 下採樣編碼器。

    流程：
        1. pre_net：Conv1d(stride=s, kernel=2s) 把 T_mel 下採成 T_z = T_mel / prod(strides)
        2. WN：stride 後的 hidden 經 dilated conv stack 提取上下文
        3. out_proj：1x1 conv 投影成 (mean, log_std) 兩半，採樣 z

    為什麼用 stride conv 而非 pooling：
        Conv 帶可學參數，可以在下採同時做特徵變換；pooling 是固定平均，浪費容量。

    為什麼 reparam 是 z = m + eps * exp(logs)：
        VAE 標準做法；exp(logs) 保證 std > 0，eps ~ N(0, I) 引入隨機性。
    """

    def __init__(self, in_channels, hidden_channels, latent_channels,
                 kernel_size, n_layers, gin_channels=0, p_dropout=0.0,
                 strides=(8,)):
        super().__init__()
        self.strides = list(strides)
        self.hidden_size = hidden_channels
        self.latent_channels = latent_channels

        # pre_net：每個 stride 一層
        layers = []
        for i, s in enumerate(self.strides):
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(
                nn.Conv1d(in_ch, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            )
        self.pre_net = nn.Sequential(*layers)
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels * 2, 1)

    def forward(self, x, x_mask, g):
        """
        Args:
            x:      [B, in_channels, T_mel]    （通常 in_channels = NUM_MELS = 80）
            x_mask: [B, 1, T_mel]              padding mask
            g:      [B, gin_channels, T_z]     已經被外層 g_pre_net 下採過的條件
        Returns:
            z:      [B, latent_channels, T_z]
            m:      [B, latent_channels, T_z]  posterior mean
            logs:   [B, latent_channels, T_z]  posterior log-std
            x_mask_sqz: [B, 1, T_z]            下採後的 mask
        """
        x = self.pre_net(x)
        # mask 同步下採（用 stride slicing）並 clip 到 x 的 T 軸長度
        x_mask = x_mask[:, :, ::int(np.prod(self.strides))][:, :, :x.shape[-1]]
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        # ── 數值安全:clamp logs 防 exp 溢位 ──
        # 訓初(NSVB-ZH 新加的 condition layer 隨機 init)實測 logs 可飆到 ±55,
        # exp(55) 直接溢位;z 抽樣的 exp(logs) 與下游 KL 的 exp(2*logs) 都會炸。
        # [-10, 10] 對應 σ ∈ [4.5e-5, 22026],兩端都在 fp32 安全範圍;健康訓中
        # logs 通常落 [-3, 3],clamp 只在病態 init 時生效,對訓練語義無影響。
        logs = logs.clamp(min=-10.0, max=10.0)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs, x_mask


# ── Decoder θ ────────────────────────────────────────────
class FVAEDecoder(nn.Module):
    """
    Latent z → Mel 的 stride 上採樣解碼器。

    流程：
        1. pre_net：ConvTranspose1d(stride=s, kernel=s) 把 T_z 上採回 T_mel
        2. WN：mel-rate 上的 dilated conv stack
        3. out_proj：1x1 conv 到 NUM_MELS 通道

    為什麼 ConvTranspose stride=s, kernel=s：
        相當於「複製 s 次」+ 學習混合，避免 checkerboard artifact（kernel != stride 易生）；
        簡單且配合 encoder 的 Conv stride downsampling 對稱。
    """

    def __init__(self, latent_channels, hidden_channels, out_channels,
                 kernel_size, n_layers, gin_channels=0, p_dropout=0.0,
                 strides=(8,)):
        super().__init__()
        self.strides = list(strides)
        self.hidden_size = hidden_channels

        layers = []
        for i, s in enumerate(self.strides):
            in_ch = latent_channels if i == 0 else hidden_channels
            layers.append(
                nn.ConvTranspose1d(in_ch, hidden_channels, kernel_size=s, stride=s)
            )
        self.pre_net = nn.Sequential(*layers)
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g):
        """
        Args:
            x:      [B, latent_channels, T_z]
            x_mask: [B, 1, T_mel]              （注意：這是 mel-rate 的 mask）
            g:      [B, gin_channels, T_mel]   mel-rate 條件
        Returns:
            mel:    [B, out_channels, T_mel]
        """
        x = self.pre_net(x)
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        return x


# ── 完整 FVAE（encoder + decoder + KL）───────────────────
class FVAE(nn.Module):
    """
    Conditional VAE：x = mel，條件 g = (PPG + F0 emb + spk_emb) 經外層拼接好的 tensor。

    與原 NSVB 差異：
        移除 use_prior_glow 分支 → 始終用 N(0, I) prior 與 closed-form KL；
        移除 FastSpeech2VAE wrapper → 條件 g 的構造交給上層 model。

    為什麼 g 被 g_pre_net 下採一次：
        encoder.wn 在 latent rate 工作，需要對應 latent rate 的條件；
        decoder.wn 在 mel rate 工作，仍用原始 g。
        所以 encoder 拿 g_sqz、decoder 拿 g。

    Args:
        in_out_channels: NUM_MELS（input/output 都是 mel）
        hidden_channels: WN/encoder/decoder 的隱通道
        latent_size:     z 的通道數（NSVB 預設 128）
        kernel_size:     WN dilated conv kernel
        enc_n_layers:    encoder WN 層數
        dec_n_layers:    decoder WN 層數
        gin_channels:    條件 g 的通道數
        strides:         tuple of stride，例如 (8,) 表單一 8x 下採
    """

    def __init__(self,
                 in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels,
                 strides=(8,), p_dropout=0.0):
        super().__init__()
        self.strides = list(strides)
        self.hidden_size = hidden_channels
        self.latent_size = latent_size

        # g 走 stride conv 下採（與 encoder pre_net 同 stride）
        g_layers = []
        for s in self.strides:
            g_layers.append(
                nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            )
        self.g_pre_net = nn.Sequential(*g_layers)

        self.encoder = FVAEEncoder(
            in_out_channels, hidden_channels, latent_size, kernel_size,
            enc_n_layers, gin_channels, p_dropout=p_dropout, strides=strides,
        )
        self.decoder = FVAEDecoder(
            latent_size, hidden_channels, in_out_channels, kernel_size,
            dec_n_layers, gin_channels, p_dropout=p_dropout, strides=strides,
        )
        # 標準 N(0, I) prior，對 z 的每個元素獨立
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, x_mask=None, g=None, infer=False):
        """
        Args:
            x:      [B, in_out_channels, T_mel]   訓練時必餵；推理（無條件先驗採樣）時 None
            x_mask: [B, 1, T_mel]                  mel-rate mask
            g:      [B, gin_channels, T_mel]       mel-rate condition
            infer:  False=訓練（forward + KL）；True=從先驗採樣

        Returns（infer=False）:
            x_recon: [B, in_out_channels, T_mel]
            loss_kl: scalar，KL 平均（per latent dim per valid frame）
            z_p:     None（保留位置與原版相容）
            m_q:    [B, latent_size, T_z]  posterior mean
            logs_q: [B, latent_size, T_z]  posterior log-std

        Returns（infer=True）:
            x_recon: [B, in_out_channels, T_mel]
            z_p:     [B, latent_size, T_z]   sampled prior
        """
        g_sqz = self.g_pre_net(g)
        if not infer:
            # 注意:encoder 內已對 logs_q 做 clamp(±10)防 exp 溢位,
            # 這裡 z_q 跟 logs_q 都已在安全範圍,KL 計算無需再 clamp
            z_q, m_q, logs_q, x_mask_sqz = self.encoder(x, x_mask, g_sqz)
            x_recon = self.decoder(z_q, x_mask, g)

            # closed-form KL：q(z|x) = N(m_q, exp(logs_q)^2) vs prior N(0, 1)
            q_dist = dist.Normal(m_q, logs_q.exp())
            loss_kl = dist.kl_divergence(q_dist, self.prior_dist)
            # 在 valid frame 上做 mean，按 latent_size 平均（讓 KL 與 latent 容量無關）
            loss_kl = (loss_kl * x_mask_sqz).sum() / x_mask_sqz.sum() / z_q.shape[1]
            return x_recon, loss_kl, None, m_q, logs_q
        else:
            # 從 prior 採樣
            latent_shape = [g_sqz.shape[0], self.latent_size, g_sqz.shape[2]]
            z_p = self.prior_dist.sample(latent_shape).to(g.device)
            x_recon = self.decoder(z_p, 1, g)
            return x_recon, z_p


if __name__ == "__main__":
    # 自我測試：Y 結構與 shape
    B, T_mel = 2, 320       # T_z = 40
    NUM_MELS = 80
    GIN = 192
    LATENT = 128

    fvae = FVAE(
        in_out_channels=NUM_MELS, hidden_channels=192, latent_size=LATENT,
        kernel_size=5, enc_n_layers=8, dec_n_layers=4,
        gin_channels=GIN, strides=(8,),
    )
    n_params = sum(p.numel() for p in fvae.parameters())
    print(f"FVAE params: {n_params/1e6:.2f}M")

    mel = torch.randn(B, NUM_MELS, T_mel)
    mask = torch.ones(B, 1, T_mel)
    g = torch.randn(B, GIN, T_mel)

    x_recon, kl, _, m_q, logs_q = fvae(mel, mask, g)
    print(f"x_recon: {x_recon.shape}  (expect [{B},{NUM_MELS},{T_mel}])")
    print(f"kl: {kl.item():.4f}  (positive scalar)")
    print(f"m_q: {m_q.shape}  (expect [{B},{LATENT},{T_mel//8}])")

    # Inference path
    fvae.eval()
    with torch.no_grad():
        x_inf, z_p = fvae(g=g, infer=True)
    print(f"infer x: {x_inf.shape}  z_p: {z_p.shape}")