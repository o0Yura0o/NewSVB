"""
nsvb/backbone/vocoder/
========================

Vocoder generator（從 NSVB modules/hifigan/hifigan.py +
modules/parallel_wavegan/models/source.py 移植，
對應 NSVB 作者提供的 1012_hifigan_all_songs_nsf ckpt 真實架構）。

【為什麼名稱叫 "1012_hifigan_all_songs_nsf" 而 NSVB config 看起來是 PWG】
- NSVB 作者的 config.yaml 同時保留兩種訓練設定（PWG-style generator_params
  + HifiGAN-style top-level keys），最終實際訓練的是 HifiGAN-NSF
- 用 ckpt 的 state_dict keys 為準 — 這是不會說謊的 ground truth：
    `m_source.l_linear`、`noise_convs.0..3` ← SourceModuleHnNSF（真 NSF）
    `conv_pre`、`ups.0..3`、`conv_post` ← HifiGAN 標準命名
    `resblocks.0..11.convs1/convs2` ← MRF (4 upsample × 3 resblock = 12)

【真實架構參數（來自 ckpt config.yaml 頂層）】
    audio_sample_rate=22050
    hop_size=128                 (= prod(upsample_rates) = 8*4*2*2)
    resblock='1'                 (用 ResBlock1, kernel=3 with dilations)
    resblock_kernel_sizes=[3, 7, 11]      (HifiGAN MRF 標準三 kernel)
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]]
    upsample_rates=[8, 4, 2, 2]
    upsample_kernel_sizes=[16, 16, 4, 4]
    upsample_initial_channel=512
    use_pitch_embed=true         (use SourceModuleHnNSF, 不是 nn.Embedding!)

【為什麼放在獨立子目錄】
- vocoder 推理時是 frozen pretrained，與訓練端解耦
- 訓練時 import nsvb.backbone 不會自動拖入 vocoder（避免 import 開銷）
- 推理 / vocoder identity test 需要時才 import

【Phase 0 後決策點】
若 vocoder identity test 失敗（SSIM < 0.85 或 F0 RMSE > 20Hz），
評估切到 RVC NSF-HifiGAN（重 align sr=40000 + re-binarize）。
細節見 rebuild_checklist.md §I 與 risk.md Monitor 1b。
"""

from nsvb.backbone.vocoder.hifigan_nsf import HifiGanNSFGenerator

__all__ = ["HifiGanNSFGenerator"]