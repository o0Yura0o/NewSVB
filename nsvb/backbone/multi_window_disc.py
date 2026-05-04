"""
nsvb/backbone/multi_window_disc.py
====================================

【這支檔案做什麼】
從 NSVB `modules/fastspeech/multi_window_disc.py` 移植 mel 層判別器 D_mel。

D_mel 接收 [B, T, NUM_MELS] mel spectrogram，輸出 real/fake 判別分數。
架構是「multi-window 2D CNN」：
    - 對每個指定的 time_length（例如 32/64/128 frame）各跑一個 2D PatchGAN-style
      sub-discriminator，每個 sub-discriminator 隨機從 batch 的 mel 切該長度的窗
    - 三種 reduction：'sum'（單一 scalar 分數）、'stack'（每窗一分數）、'none'（per-frame 分數）

【為什麼用 multi-window 而非單一固定長度】
- 不同 window 看不同尺度的 mel artifact：
    32 frame ~ 0.19s（音素級 spectral artifact）
    64 frame ~ 0.37s（短音節級）
   128 frame ~ 0.74s（vibrato cycle 級）
- 訓練時隨機抽樣 window 起點，避免 D 過 fit 單一段落
- 比單一大 window 的 D 訓練更穩、收斂更快（NSVB 原作者實證）

【為什麼 2D CNN 而非 1D】
- mel 是 2D 結構（time × freq bin），2D conv 可以同時看頻譜峰位置 + 時間延展
- 單一 1D conv 在每個 time slice 上看不到頻率關係（除非整個 freq dim 攤平成 channel，
  但通道數 80 對 1D conv 來說太大）

【相對 NSVB 原版的修改】
- 純結構移植，沒有改演算法
- 只是註解風格調整、import 簡化
- 為什麼幾乎不改：D_mel 結構穩定，原版 NSVB 訓練了上千小時資料證明 work，沒理由動

【Stage 2 調整】
- Stage 1：D_mel 看 amateur + pro 都當 real（學「自然人聲」）
- Stage 2：D_mel real 只餵 pro mel（升級成「pro 自然度」判別器）
- 結構不變，只改外層 task 的 loss 計算
"""

import numpy as np
import torch
import torch.nn as nn


class Discriminator2DFactory(nn.Module):
    """
    單一 2D CNN sub-discriminator。

    架構：3 層 stride-2 2D conv 把 [B, c_in, T, freq] 下採成 [B, hidden, T/8, freq/8]，
          再接一個 Linear 出 scalar / per-frame 分數。

    為什麼 stride=2 + kernel=3：
        PatchGAN 標準寫法；每層感受野翻倍但通道不變，
        最後總感受野 ~ 8*kernel = 24 frames（覆蓋 ~140ms 的 spectral block）。

    為什麼第一層不放 BN/IN：
        BatchNorm 在 D 第一層接收 raw mel 統計差異大、不穩；first=True 跳過
        是 Pix2Pix / NSVB 的慣例。
    """

    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1,
                 hidden_size=128, norm_type="bn", reduction="sum"):
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)

        def block(in_filters, out_filters, first=False):
            """
            一個 [Conv2d, LeakyReLU, Dropout2d] (+optional norm) 組合。
            Input  : (B, in_filters, 2H, 2W)
            Output : (B, out_filters, H, W)（stride=2 下採）
            """
            conv = nn.Conv2d(in_filters, out_filters, kernel, (2, 2), padding)
            if norm_type == "sn":
                conv = nn.utils.spectral_norm(conv)
            layers = [
                conv,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if norm_type == "bn" and not first:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            if norm_type == "in" and not first:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            return nn.Sequential(*layers)

        self.model = nn.ModuleList([
            block(c_in, hidden_size, first=True),
            block(hidden_size, hidden_size),
            block(hidden_size, hidden_size),
        ])

        self.reduction = reduction
        # 三層 stride 2 → 下採 8x；freq 用 (freq+7)//8 保護不滿 8 的尾段
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        if reduction != "none":
            # 全域 flatten 出單一 scalar
            self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)
        else:
            # 保留 time 維 → 每個下採 frame 出一個分數
            self.adv_layer = nn.Linear(hidden_size * ds_size[1], 1)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, freq]
        Returns:
            validity: [B, 1] (reduction!=none) 或 [B, T_ds] (reduction=none)
            h:        list of intermediate feature maps（給 feature matching loss 用）
        """
        h = []
        for layer in self.model:
            x = layer(x)
            h.append(x)

        if self.reduction != "none":
            x = x.view(x.shape[0], -1)
            validity = self.adv_layer(x)  # [B, 1]
        else:
            B, _, T_, _ = x.shape
            x = x.transpose(1, 2).reshape(B, T_, -1)
            validity = self.adv_layer(x)[:, :, 0]  # [B, T_]
        return validity, h


class MultiWindowDiscriminator(nn.Module):
    """
    Multi-window 包裝：對每個 time_length 開一個 Discriminator2DFactory，
    forward 時對 input batch 隨機切該長度窗送進去。

    為什麼隨機抽窗起點：
        若每次都從頭開始切，D 只看到歌曲開頭；隨機切讓 D 看遍整首歌的所有段落。

    為什麼支援 cond：
        若 D 要做 conditional discriminator（輸入 mel + 條件向量），
        cond 經 Linear → 與 mel 相加（注入到頻軸）。
        本專案 D_mel 預設 unconditional（cond_size=0），但保留接口。
    """

    def __init__(self, time_lengths, cond_size=0, freq_length=80, kernel=(3, 3),
                 c_in=1, hidden_size=128, norm_type="bn", reduction="sum"):
        super().__init__()
        self.win_lengths = time_lengths
        self.reduction = reduction

        self.conv_layers = nn.ModuleList()
        if cond_size > 0:
            self.cond_proj_layers = nn.ModuleList()
            self.mel_proj_layers = nn.ModuleList()

        for time_length in time_lengths:
            self.conv_layers.append(
                Discriminator2DFactory(
                    time_length, freq_length, kernel, c_in=c_in,
                    hidden_size=hidden_size, norm_type=norm_type, reduction=reduction,
                )
            )
            if cond_size > 0:
                self.cond_proj_layers.append(nn.Linear(cond_size, freq_length))
                self.mel_proj_layers.append(nn.Linear(freq_length, freq_length))

    def forward(self, x, x_len, cond=None, start_frames_wins=None):
        """
        Args:
            x:                [B, c_in, T, freq]
            x_len:            [B] 每筆有效長度
            cond:             [B, T, cond_size] 或 None
            start_frames_wins: list of int 或 None
                              len = num windows；若 None 則隨機；否則固定起點（給 G/D 同步切窗用）

        Returns:
            validity:          依 reduction：sum=[B] / stack=[B, n_win] / none=[B, T_sum_ds]
            start_frames_wins: 此次每窗實際的起點（給 G 同步用）
            h:                 list of feature maps
        """
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.conv_layers)

        validity = []
        h = []
        for i, start_frames in zip(range(len(self.conv_layers)), start_frames_wins):
            x_clip, c_clip, start_frames = self._clip(
                x, cond, x_len, self.win_lengths[i], start_frames
            )
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            if cond is not None:
                x_clip = self.mel_proj_layers[i](x_clip)
                c_clip = self.cond_proj_layers[i](c_clip)[:, None]
                x_clip = x_clip + c_clip
            x_clip, h_ = self.conv_layers[i](x_clip)
            h += h_
            validity.append(x_clip)

        if len(validity) != len(self.conv_layers):
            return None, start_frames_wins, h

        if self.reduction == "sum":
            validity = sum(validity)
        elif self.reduction == "stack":
            validity = torch.stack(validity, -1)
        elif self.reduction == "none":
            validity = torch.cat(validity, -1)

        return validity, start_frames_wins, h

    @staticmethod
    def _clip(x, cond, x_len, win_length, start_frames=None):
        """
        從 x 隨機切 win_length 長度的窗。

        為什麼用 batch 的 max len 推 T_end：
          一個 batch 內每筆長度可能不一，但隨機切窗對齊用 batch max 簡化邏輯；
          padding 部分後續 mask 會處理。

        Returns:
            x_batch:     [B, c_in, win_length, freq] 或 None（batch 全部 < win_length）
            c_batch:     [B, win_length, cond_size] 或 None
            start_frames: list[int]
        """
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()

        if start_frames is None:
            start_frame = np.random.randint(low=0, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]

        x_batch = x[:, :, start_frame:start_frame + win_length]
        c_batch = cond[:, start_frame:start_frame + win_length] if cond is not None else None
        return x_batch, c_batch, start_frames


class Discriminator(nn.Module):
    """
    對外的 D_mel 介面。包裝 unconditional + (optional) conditional 兩個分支。

    為什麼把 uncond/cond 分成兩個 sub-disc：
        若同一個 D 既看 mel 又看 cond，學到的 cond 表徵會被 mel 主導；
        分兩個獨立 D 各管一邊，訓練訊號更乾淨。

    本專案 Stage 2 預設只啟用 unconditional 部分（uncond_disc=True, cond_size=0），
    與 NSVB 原版 stage2 一致；保留接口給未來實驗。
    """

    def __init__(self, time_lengths=(32, 64, 128), freq_length=80, cond_size=0,
                 kernel=(3, 3), c_in=1, hidden_size=128, norm_type="bn",
                 reduction="sum", uncond_disc=True):
        super().__init__()
        self.time_lengths = list(time_lengths)
        self.cond_size = cond_size
        self.reduction = reduction
        self.uncond_disc = uncond_disc

        if uncond_disc:
            self.discriminator = MultiWindowDiscriminator(
                freq_length=freq_length, time_lengths=self.time_lengths,
                kernel=kernel, c_in=c_in, hidden_size=hidden_size,
                norm_type=norm_type, reduction=reduction,
            )
        if cond_size > 0:
            self.cond_disc = MultiWindowDiscriminator(
                freq_length=freq_length, time_lengths=self.time_lengths,
                cond_size=cond_size, kernel=kernel, c_in=c_in,
                hidden_size=hidden_size, norm_type=norm_type, reduction=reduction,
            )

    def forward(self, x, cond=None, start_frames_wins=None):
        """
        Args:
            x:                [B, T, freq]    （沒 channel 維時自動 unsqueeze）
            cond:             [B, T, cond_size] 或 None
            start_frames_wins: 給 G/D 切相同窗用
        Returns:
            ret: dict { 'y': uncond logits, 'y_c': cond logits, 'h': features,
                        'h_c': cond features, 'start_frames_wins': ...}
        """
        if x.dim() == 3:
            x = x[:, None, :, :]  # [B, 1, T, freq]

        # 自動推 valid 長度：對 (channel + freq) 軸求和不為 0 的時間步
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])

        ret = {"y_c": None, "y": None}
        if self.uncond_disc:
            ret["y"], start_frames_wins, ret["h"] = self.discriminator(
                x, x_len, start_frames_wins=start_frames_wins
            )
        if self.cond_size > 0 and cond is not None:
            ret["y_c"], start_frames_wins, ret["h_c"] = self.cond_disc(
                x, x_len, cond, start_frames_wins=start_frames_wins
            )
        ret["start_frames_wins"] = start_frames_wins
        return ret


if __name__ == "__main__":
    # 自我測試：D_mel 對隨機 mel 是否回傳合理 shape
    B, T, FREQ = 4, 256, 80
    D = Discriminator(
        time_lengths=(32, 64, 128), freq_length=FREQ,
        cond_size=0, hidden_size=128, norm_type="bn", reduction="sum",
        uncond_disc=True,
    )
    n_params = sum(p.numel() for p in D.parameters())
    print(f"D_mel params: {n_params/1e6:.2f}M")

    mel = torch.randn(B, T, FREQ)
    ret = D(mel)
    print(f"y shape: {ret['y'].shape}  (reduction=sum 預期 [B, 1])")
    print(f"start_frames_wins: {ret['start_frames_wins']}")
    print(f"hidden feature maps: {len(ret['h'])} entries")