"""
nsvb/backbone/vocoder/source.py
=================================

【這支檔案做什麼】
從 NSVB `modules/parallel_wavegan/models/source.py` 移植 NSF
（Neural Source-Filter, Wang 2019）的 source signal 產生器：
    SineGen          : F0 → 多諧波 sine waveforms（fundamental + N overtones）
    SourceModuleHnNSF: SineGen 包裝，把多諧波 merge 成單一 excitation signal

【為什麼 vocoder 需要這個】
真實 HifiGAN-NSF 的關鍵差別：除了 mel 條件，還在 audio rate 直接生成
sin(2πf0t) + 多諧波 sin 訊號當作 source（excitation），讓 generator 不必
從零學習 harmonic 結構，只需學 envelope / filter。
對歌聲（sustained note + vibrato）這個架構效果遠好於純 mel HifiGAN：
- mel 在歌聲穩定段難穩定保留 phase coherence → 純 mel HifiGAN 易 fuzzy
- NSF 把 sin source 餵到每個 upsample stage，phase 由 F0 直接決定，乾淨

【為什麼用 cumsum trick 算瞬時相位】
公式：sine[t] = sin(2π * Σ_{k=1..t} f_k / sr)
直接 cumsum 對長序列會數值溢位（rad 累積到很大）。
原作者用 modulo 1 + 抓「過 1 的 step」減回去的技巧，讓 cumulative phase
保持在 [0, 1] 範圍，數值穩定。

【SineGen 的 forward 為什麼用 torch.no_grad】
Source signal 是「給 vocoder 用的固定訊號」，不需要對 SineGen 的內部變數
（rad accumulator、隨機初始相位）算 gradient；no_grad 節省記憶體。
"""

import numpy as np
import torch
import torch.nn as nn


class SineGen(nn.Module):
    """
    F0 → 多諧波 sine waveforms。

    輸入 F0 [B, T_audio, 1]（已上採到 audio rate），輸出 sine_waves [B, T_audio, 1+harmonic_num]。

    Args:
        samp_rate:        sample rate (Hz), 用來把 F0 (Hz) 轉相位
        harmonic_num:     額外諧波數；輸出 dim = 1 + harmonic_num
                          1012 ckpt 用 8（基頻 + 8 overtones, 共 9 諧波）
        sine_amp:         sine wave 振幅（unvoiced 段噪聲也用此 scale）
        noise_std:        voiced 段疊加的 Gaussian noise std
        voiced_threshold: F0 > 此值才算 voiced；1012 ckpt 用 0
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1,
                 noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        """F0 → UV mask；shape 同 f0。"""
        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        """
        F0 [B, T, dim] → sine waves [B, T, dim]。

        為什麼有 cumsum_shift trick：
            純 cumsum(rad / sr) 在長 wav 會線性增長到很大（10^4+）；
            對 2*pi*x 的 sin 來說，整數部分不影響結果但浮點精度會散失。
            trick：cumsum 結果取 mod 1 後抓「相鄰差為負」（過 1 點）的 step，
                   在那些 step 把 rad 減 1，讓累計恆在 [0, 1)，精度穩定。

        為什麼 batch 0 通道（fundamental）給隨機初始相位：
            純從 0 開始 cumulative phase 在 unvoiced 段會出現「人工同步」，
            voiced 段聽起來不自然；隨機初始相位破除這對稱性。
            其他諧波繼承 fundamental 的相位，所以 rand_ini 只給 channel 0。
        """
        # rad in [0, 1)；對 fundamental 與每個 harmonic 都各算一份
        rad_values = (f0_values / self.sampling_rate) % 1

        # 隨機初始相位（只給 fundamental, channel 0）
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # cumsum 並抓「過 1」的位置
        tmp_over_one = torch.cumsum(rad_values, dim=1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(
            torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
        )
        return sines

    def forward(self, f0):
        """
        Args:
            f0: [B, T_audio, 1]，audio-rate F0（已用 nearest-Upsample 從 mel-rate 上採）

        Returns:
            sine_waves: [B, T_audio, 1 + harmonic_num]
            uv:         [B, T_audio, 1]
            noise:      [B, T_audio, 1 + harmonic_num]
        """
        with torch.no_grad():
            # 為每個 harmonic 構造 F0 buffer：channel 0 = fundamental，
            # channel k>=1 = (k+1) * fundamental
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)

            # voiced 段加微小 noise，unvoiced 段加大 noise（amp ~ sine_amp/3）
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # voiced 處：sine + small noise；unvoiced 處：純 noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    NSVB / RVC HifiGAN-NSF 的 harmonic-noise source module。

    把 SineGen 的多諧波輸出（[B, T, 1 + harmonic_num]）經過：
        l_linear : Linear(harmonic_num+1 → 1)  讓網路自己學各諧波的權重
        l_tanh   : 限幅到 [-1, 1]
    回傳：
        sine_merge: [B, T, 1]   harmonic excitation
        noise:      [B, T, 1]   white noise（單獨頻道，給 generator 的 noise branch）
        uv:         [B, T, 1]   voiced flag

    為什麼把諧波合成分離成 Linear + tanh：
        每個 harmonic 對最終訊號的貢獻可學（不是固定平均）；
        tanh 確保 source 的最終 amplitude 不爆，generator 後續 receptive 較穩。
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: [B, T_audio, 1]   audio-rate F0（已上採）

        Returns:
            sine_merge: [B, T_audio, 1]
            noise:      [B, T_audio, 1]
            uv:         [B, T_audio, 1]
        """
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # noise branch（與 SineGen 內部 noise 不同；給 generator noise_convs 用）
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


if __name__ == "__main__":
    # 自我測試：餵 440 Hz F0，驗證 SineGen 確實生成 440 Hz sin
    sr = 22050
    duration = 0.1
    T = int(sr * duration)
    B = 1

    f0 = torch.full((B, T, 1), 440.0)
    src = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=8)
    src.eval()
    with torch.no_grad():
        merge, noise, uv = src(f0)
    print(f"sine_merge shape: {merge.shape}  (expect [{B},{T},1])")
    print(f"noise shape:      {noise.shape}")
    print(f"uv shape:         {uv.shape}, all-voiced: {bool(uv.all())}")
    print(f"merge range: [{merge.min().item():.3f}, {merge.max().item():.3f}]")
    print(f"merge std:   {merge.std().item():.4f}")