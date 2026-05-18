"""F0 smoothing utilities for Plan B (f0_support).

【為什麼存在】
Risk §二.3 amateur F0 conditioning confound:Stage 2 D_mel real 是 pro mel(由 pro
F0 生),fake 是 decoder 出來的 mel(由 amateur F0 生)。D_mel 可能用「F0 軌跡平滑度」
當判別捷徑 → M 學不到真正的 pro 化方向。

f0_support 想法:在 fake mel 生成時,把 amateur F0 平滑化(去掉 jitter,保留 note
center + vibrato),讓 D_mel 不能用「F0 微抖動」當 amateur 簽名。

【方法】
- `median`: 中位數濾波,robust 對單點 outlier;簡單可靠
- `savgol`: Savitzky-Golay 多項式擬合,保留局部曲率(對 vibrato 友好)
- `none`: 不平滑(v2 行為)

【在 log-F0 space 平滑】
音樂感知是 log-pitch,半音間距在 log 內等距。直接在 Hz space 平滑會偏向高音域,
log space 才對。
"""
from typing import Literal

import numpy as np
import torch
from scipy.signal import medfilt, savgol_filter


def smooth_f0_numpy(
    f0: np.ndarray,
    method: Literal["median", "savgol", "none"] = "median",
    window: int = 5,
) -> np.ndarray:
    """單條 F0 平滑(numpy,給 batch=1 / debug 用)。

    Args:
        f0:     [T] Hz, voiced=正值, unvoiced=0
        method: 平滑方法
        window: 窗口大小(自動補成奇數)

    Returns:
        smoothed F0 [T] Hz,unvoiced 段仍 0
    """
    if method == "none":
        return f0.astype(np.float32)
    if window % 2 == 0:
        window += 1

    f0_in = f0.astype(np.float32)
    voiced = f0_in > 0
    if not voiced.any():
        return f0_in

    # 在 voiced 連續段內平滑(boundary 不跨,避免 unvoiced=0 拉低 voiced 平均)
    log_f0 = np.zeros_like(f0_in)
    log_f0[voiced] = np.log2(f0_in[voiced] + 1e-5)

    out_log = log_f0.copy()
    for start, end in _find_runs(voiced):
        seg_len = end - start
        if seg_len < window:
            continue  # 太短跳過
        seg = log_f0[start:end]
        if method == "median":
            smoothed = medfilt(seg, kernel_size=window)
        elif method == "savgol":
            poly_order = min(3, window - 1)
            smoothed = savgol_filter(seg, window_length=window, polyorder=poly_order)
        else:
            raise ValueError(f"unknown method: {method}")
        out_log[start:end] = smoothed

    out = np.zeros_like(f0_in)
    out[voiced] = np.exp2(out_log[voiced])
    return out.astype(np.float32)


def smooth_f0_batch_torch(
    f0_batch: torch.Tensor,
    method: Literal["median", "savgol", "none"] = "median",
    window: int = 5,
) -> torch.Tensor:
    """Batch 版,GPU friendly:detach → CPU → numpy → smooth per-sample → 回 GPU。

    為什麼不全 GPU vectorized:
      median filter 在 GPU 沒原生 op(要自寫 unfold + sort);savgol 是 conv1d 可以,
      但 voiced/unvoiced boundary 條件處理在 vectorized 路徑很麻煩。
    每步 ~0.05s overhead(B=16, T=1500),50K 步 ~40min 總 overhead,可接受。

    Args:
        f0_batch: [B, T] Hz on any device
        method:   平滑方法
        window:   窗口大小

    Returns:
        smoothed [B, T] 同 device 同 dtype
    """
    if method == "none":
        return f0_batch

    orig_device = f0_batch.device
    orig_dtype = f0_batch.dtype
    f0_np = f0_batch.detach().cpu().numpy().astype(np.float32)
    out = np.empty_like(f0_np)
    for b in range(f0_np.shape[0]):
        out[b] = smooth_f0_numpy(f0_np[b], method=method, window=window)
    return torch.from_numpy(out).to(device=orig_device, dtype=orig_dtype)


def _find_runs(mask: np.ndarray) -> list:
    """Find runs of True in boolean array. Returns [(start, end), ...] (end exclusive)。"""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs
