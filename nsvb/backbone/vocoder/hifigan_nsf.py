"""
nsvb/backbone/vocoder/hifigan_nsf.py
=====================================

【這支檔案做什麼】
從 NSVB `modules/hifigan/hifigan.py` 移植真正的 HifiGAN-NSF generator，
對應 ckpt `1012_hifigan_all_songs_nsf/` 的實際架構。

HifiGanNSFGenerator: (mel, F0_hz) → wav
    - mel 經 conv_pre → 上採 4 stage（rates [8, 4, 2, 2], prod=128=hop）
    - F0 同步上採 audio rate → SourceModuleHnNSF 生成 harmonic source
    - 每個 upsample stage：x = ConvTranspose1d(x); x += noise_convs(har_source)
    - 每 stage 後過 3 個 MRF ResBlock1（kernels [3,7,11]，dilations [[1,3,5]] x3）求平均
    - 最後 conv_post + tanh 出 wav [-1, 1]

【相對 NSVB 原版的修改】
1. **建構介面從 hparams dict 改為 explicit args**：原版 __init__(self, h, c_out=1)
   讀 dict；我們把所有需要的 keys 拆成 named args，方便構建與測試
2. **移除 Discriminator 類**：DiscriminatorP / MultiPeriodDiscriminator /
   DiscriminatorS / MultiScaleDiscriminator — 我們用 frozen pretrained，不訓練 vocoder
3. **移除 use_pitch_embed=False 的純 mel 分支**：1012 ckpt 必有 NSF source；
   避免 conditional branch 增加閱讀負擔
4. **forward 輸入 F0 維度從 [B, T_mel] 統一為 [B, T_mel] (Hz)**：
   原作者寫法 `f0[:, None]`，我們維持一致

【為什麼 ResBlock1 而非 ResBlock2】
1012 ckpt config: `resblock='1'`。ResBlock1 用 (1,3,5) dilation 三層，
ResBlock2 用 (1,3) 兩層。ResBlock1 感受野更大，HifiGAN paper 預設選擇。

【架構參數（hardcode 自 ckpt config）】
    upsample_rates           = [8, 4, 2, 2]    （prod = 128 = hop_size）
    upsample_kernel_sizes    = [16, 16, 4, 4]
    upsample_initial_channel = 512             （第一層 conv_pre 輸出通道）
    resblock_kernel_sizes    = [3, 7, 11]      （MRF 三 kernel）
    resblock_dilation_sizes  = [[1,3,5]] * 3  （每個 kernel 都用 (1,3,5)）
    harmonic_num             = 8               （SourceModuleHnNSF 的諧波數）
    audio_sample_rate        = 22050

upsample 後 channel：512 → 256 → 128 → 64 → 32（每 stage 折半）

【LRELU_SLOPE = 0.1】
HifiGAN paper 預設；比 0.2 更平滑、訓練早期梯度更穩。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from nsvb.backbone.vocoder.source import SourceModuleHnNSF


LRELU_SLOPE = 0.1


def _init_weights(m, mean=0.0, std=0.01):
    """Conv 層權重初始化為 N(mean, std)；HifiGAN 原作者選擇。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    """SAME padding 對應公式（kernel * dilation - dilation) // 2。"""
    return int((kernel_size * dilation - dilation) / 2)


# ── ResBlock1（MRF 元件）──────────────────────────────
class ResBlock1(nn.Module):
    """
    HifiGAN ResBlock1：
        對輸入 x，並列 3 條（dilation=1, 3, 5）：
            xt = LeakyReLU(x); xt = Conv1d_d(xt); xt = LeakyReLU(xt); xt = Conv1d_1(xt)
            x  = x + xt   （residual）
        三條依序累積到 x（不是平行）

    論文中 ResBlock1 對應 V1 版本，和歌聲場景 `kernel=3` + dilation [1,3,5]
    平衡 receptive field 與參數量。

    為什麼每對 conv 是 (dilated → 1)：
        第一個 dilated conv 抓寬感受野，第二個 conv (dilation=1) 做 local refine；
        若兩個都 dilated 結果空洞、純 local 又無上下文。
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=d, padding=_get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs1.apply(_init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=_get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs2.apply(_init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in list(self.convs1) + list(self.convs2):
            remove_weight_norm(layer)


# ── HifiGAN-NSF Generator ────────────────────────────
class HifiGanNSFGenerator(nn.Module):
    """
    對應 NSVB ckpt 1012_hifigan_all_songs_nsf 的真實 generator。

    輸入：
        mel: [B, 80, T_mel]                 log-mel
        f0:  [B, T_mel] (Hz, 連續值, 0=unvoiced)

    輸出：
        wav: [B, 1, T_audio]                T_audio = T_mel * hop_size
    """

    # 全部 hardcode 自 ckpt config.yaml（這些不該變動，動了就跟 ckpt 不相容）
    UPSAMPLE_RATES = [8, 4, 2, 2]
    UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
    UPSAMPLE_INITIAL_CHANNEL = 512
    RESBLOCK_KERNEL_SIZES = [3, 7, 11]
    RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    HARMONIC_NUM = 8
    DEFAULT_SAMPLE_RATE = 22050
    DEFAULT_NUM_MELS = 80

    def __init__(self,
                 c_out=1,
                 num_mels=DEFAULT_NUM_MELS,
                 audio_sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__()
        self.num_kernels = len(self.RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(self.UPSAMPLE_RATES)
        self.harmonic_num = self.HARMONIC_NUM

        # F0 上採（最近鄰，乘 hop_size）
        # 為什麼用 nn.Upsample 而非 F.interpolate：模組化，weight 不用、scale 固定
        self.f0_upsamp = nn.Upsample(scale_factor=int(np.prod(self.UPSAMPLE_RATES)))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=audio_sample_rate, harmonic_num=self.harmonic_num,
        )

        # mel → upsample_initial_channel
        self.conv_pre = weight_norm(
            Conv1d(num_mels, self.UPSAMPLE_INITIAL_CHANNEL, 7, 1, padding=3)
        )

        # 4 個 upsample stage
        self.ups = nn.ModuleList()
        # noise_convs：每個 stage 一個，把 audio-rate harmonic source down 到該 stage 的 channel
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.UPSAMPLE_RATES, self.UPSAMPLE_KERNEL_SIZES)):
            c_cur = self.UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(c_cur * 2, c_cur, k, u, padding=(k - u) // 2)
            ))
            # noise_convs：對 har_source（已是 audio rate）下採到該 stage 的 rate
            # stride_f0 = prod(後續 upsample) — 把 audio rate 壓縮到 stage 的 rate
            if i + 1 < len(self.UPSAMPLE_RATES):
                stride_f0 = int(np.prod(self.UPSAMPLE_RATES[i + 1:]))
                self.noise_convs.append(
                    Conv1d(1, c_cur, kernel_size=stride_f0 * 2,
                           stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                # 最後 stage 已到 audio rate，noise_conv 是 1x1
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        # MRF：每個 upsample stage 配 num_kernels (3) 個 ResBlock
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(self.RESBLOCK_KERNEL_SIZES,
                                           self.RESBLOCK_DILATION_SIZES)):
                self.resblocks.append(ResBlock1(ch, k, d))

        # 最後輸出：channel → 1，過 tanh 限幅
        self.conv_post = weight_norm(Conv1d(ch, c_out, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, x, f0):
        """
        Args:
            x:  [B, num_mels, T_mel]   mel
            f0: [B, T_mel]              F0 in Hz (連續值, 0=unvoiced)

        Returns:
            wav: [B, c_out, T_audio]   T_audio = T_mel * 128
        """
        # 1. F0 上採到 audio rate → SourceModuleHnNSF 生成 harmonic excitation
        # f0[:, None] = [B, 1, T_mel] → upsamp → [B, 1, T_audio] → transpose → [B, T_audio, 1]
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [B, 1, T_audio]

        # 2. mel pre-conv
        x = self.conv_pre(x)  # [B, 512, T_mel]

        # 3. 4 個 upsample stage：(LeakyReLU → ConvTranspose1d → + noise_source → MRF avg)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)  # [B, c_cur, T_stage_i]
            x = x + x_source

            # MRF：3 個 ResBlock 並列累加再除 num_kernels
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels

        # 4. 出 audio
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """推理前移除 weight_norm 以加速。"""
        for layer in self.ups:
            remove_weight_norm(layer)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


if __name__ == "__main__":
    # 自我測試：shape 驗證；不載 ckpt
    HOP = 128
    B, T_mel = 2, 64
    T_audio = T_mel * HOP

    g = HifiGanNSFGenerator(c_out=1, num_mels=80, audio_sample_rate=22050)
    n_params = sum(p.numel() for p in g.parameters())
    print(f"HifiGAN-NSF generator params: {n_params/1e6:.2f}M")

    mel = torch.randn(B, 80, T_mel)
    f0 = torch.full((B, T_mel), 220.0)  # all-voiced 220 Hz
    g.eval()
    with torch.no_grad():
        out = g(mel, f0)
    print(f"out shape: {out.shape}  (expect [{B}, 1, {T_audio}])")
    print(f"out range: [{out.min().item():.3f}, {out.max().item():.3f}]  (expect within [-1, 1])")