"""
nsvb/utils/f0_utils.py
========================

【這支檔案做什麼】
F0 處理共用工具，主要兩件事：
    1. interp_f0_unvoiced(f0) — 對 unvoiced (== 0) 段做 log-space 線性內插
    2. f0_to_log_hz(f0)        — 訓練時把 Hz F0 轉 log-Hz（含 unvoiced 處理）

【為什麼需要 interp_f0_unvoiced】
HifiGAN-NSF vocoder 的 SineGen 對連續 F0 trajectory 工作；若 voiced=Hz, unvoiced=0
直接餵入，邊界處 sin wave 從正弦突然變 0 → 高頻 transient → 重建出電音。
NSVB 訓練 pipeline 的 norm_interp_f0 + denorm_f0(use_uv=False) 路徑等價於：
    1. log2(f0) — 進 log space
    2. np.interp at unvoiced positions — 連續內插
    3. 2^x — 還原 Hz
    結果：unvoiced 段獲得 voiced 兩端的幾何平均 Hz，無 0 gap

【這個 helper 在哪裡用】
- vocoder identity test（已用，從 vocoder_identity_test.py 抽出來）
- Mode A / Mode B 推理時餵 vocoder 前
- Stage 1 / Stage 2 training 時若要對 F0 做內插處理（通常存 raw 內插值與 uv mask
  分離，由 task 決定是否再合併）

【為什麼 log-space 線性內插而非 Hz-space】
- F0 是對數知覺的（半音差是固定 Hz ratio）
- log-space 線性 = Hz-space 幾何平均，過 unvoiced 段 pitch 緩變、自然
- Hz-space 直接內插會在大跨度（低音 → 高音 unvoiced gap）出現「卡半音中間」異常
"""

import numpy as np


def interp_f0_unvoiced(f0: np.ndarray) -> np.ndarray:
    """
    對 F0 的 unvoiced 段做 log-space 線性內插，回傳「無 0 gap」的連續 F0。

    為什麼回傳新 array 而非 in-place：
      F0 在 dataset 通常以 voiced=Hz, unvoiced=0 為儲存格式，原始資料保留
      這個格式有意義（uv mask 由 == 0 推得）；內插是「餵 vocoder 前」的預處理。

    Args:
        f0: [T] np.ndarray, voiced=Hz (>0), unvoiced=0

    Returns:
        f0_interp: [T] float32, 連續 Hz 值（unvoiced 段被 voiced 兩端 log 線性內插填補）
    """
    f0 = f0.copy().astype(np.float32)
    uv = f0 == 0
    if uv.all():
        # 整段都 unvoiced，無從內插
        return f0
    if uv.any():
        # 在 log space 做線性內插。np.maximum 防 log2(0) = -inf
        log_f0 = np.log2(np.maximum(f0, 1e-8))
        log_f0[uv] = np.interp(
            np.where(uv)[0], np.where(~uv)[0], log_f0[~uv],
        )
        f0 = (2.0 ** log_f0).astype(np.float32)
    return f0


def f0_to_log_hz(f0: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    把 Hz F0 轉成 log2-Hz space（訓練時 condition embedding 用）。

    為什麼 log2 而非自然 log：
      log2 對音高半音差是線性的（每 +1 = 倍頻）；
      neural network 的 linear layer 在 log2 space 學到的會更接近「半音偏移」這種音樂直覺

    為什麼加 eps 而非 maximum：
      訓練期不希望 unvoiced 留下 log2(0) = -inf 之類的 NaN；
      log2(eps) ≈ -26.6 給網絡一個遠離 voiced 範圍的 sentinel 值

    Args:
        f0:  [T] Hz (voiced > 0, unvoiced = 0)
        eps: log argument 下限

    Returns:
        log_f0: [T] float32 (log2 scale)
    """
    return np.log2(f0 + eps).astype(np.float32)
