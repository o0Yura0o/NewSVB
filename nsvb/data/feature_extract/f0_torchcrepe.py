"""
nsvb/data/feature_extract/f0_torchcrepe.py
============================================

【這支檔案做什麼】
從原始音檔抽取 frame-level F0（Hz），並把輸出對齊到 mel frame rate（86.13 fps）。
回傳：
    f0:        np.float32 [T_mel]，單位 Hz；unvoiced frame = 0.0
    voicing:   np.float32 [T_mel]，每 frame 的 voicing 信心度 [0,1]

【為什麼用 torchcrepe 取代原 NSVB 的 CREPE / parselmouth / pyworld】
- 原 NSVB 用 parselmouth (Praat) 或 pyworld 抽 F0，這兩者對歌聲（vibrato、glide、
  快速 portamento）容易判斷錯誤
- torchcrepe 是 CREPE 模型的 PyTorch 移植：
    1. 純 PyTorch、可走 GPU，比 parselmouth/pyworld 快 5-10x
    2. CREPE 對歌聲場景訓練比例高，voicing 邊界比 pyworld 準
    3. API 與原 CREPE 相容，未來若要換 RMVPE / FCPE 等更新模型，介面易換
- 為什麼不用 RMVPE：RMVPE 對伴奏歌聲更強，但 NSVB-ZH 是純人聲輸入，CREPE 已足夠，
  且 RMVPE 模型分發無 pip 套件，部署成本高

【為什麼要把 voicing 一起回傳】
- 訓練端（mel 重建）：voicing 不直接用，但 binarizer 把它存進 dataset 供將來
  視需要使用（例如 voicing-aware loss）
- 推理端（DSP 音準修正）：MIDI snap 演算法需要區分 voiced/unvoiced 段，避免在
  氣聲/輔音段做音準量化
- Soft register bucket：本來只用 F0 即可判斷 voicing（F0=0），voicing 是 backup

【為什麼把 frame 對齊到 mel rate 而非 CREPE 預設】
- D_z 的 register 條件必須與 z 的 frame 對齊；z 來自 mel encoder，frame rate = mel rate
- CREPE 原生 hop 與 mel hop 對齊很方便：直接設 torchcrepe.predict(hop_length=HOP_SIZE)
  就能取得同 fps 的 F0，不需要 post-resample
"""

from typing import Tuple

import numpy as np
import torch
import torchcrepe

from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    HOP_SIZE,
    F0_FMIN,
    F0_FMAX,
    CREPE_MODEL,
    CREPE_VITERBI,
    CREPE_CONFIDENCE_THRESHOLD,
)


# Phase 0 抽特徵時的 batch size（torchcrepe 內部分塊推理）
# 為什麼 2048：torchcrepe 推理是 frame-wise CNN，batch=2048 frames 在 8GB VRAM 跑得動，
#              更大 batch 邊際效益有限
TORCHCREPE_BATCH_SIZE = 2048


def extract_f0(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    confidence_threshold: float = CREPE_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    主入口：從一段音訊抽 F0 + voicing。

    Args:
        audio:                np.ndarray [N_samples]，float32，已正規化到 [-1, 1]
        sample_rate:          輸入音訊取樣率（會傳給 torchcrepe，內部自行 resample 到 16k）
        device:               'cuda' 或 'cpu'
        confidence_threshold: voicing 閾值；< 此值 frame F0 設 0

    Returns:
        f0:       [T_mel] float32，Hz；unvoiced=0
        voicing:  [T_mel] float32，[0, 1]

    Frame 對齊保證：
        T_mel = ceil(N_samples / HOP_SIZE)
        torchcrepe 對 (audio, hop_length=HOP_SIZE) 會輸出長度約 N_samples//hop+1 的 F0；
        若與 mel 略差 1 frame，下游 binarizer 會 pad/trim 對齊（這是 STFT/CREPE 邊界差異
        的常見處理）。

    為什麼 audio 期望 numpy 而非 tensor：
        Phase 0 的 audio loading 用 librosa.load 直接拿到 numpy；轉 tensor 留給此函式內部
        統一處理（避免 caller 重複 to(device)）。
    """
    # 1. numpy → torch (CPU)；torchcrepe 內部會 send to device
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)  # [1, N]

    # 2. 跑 CREPE
    # torchcrepe.predict 回傳 (pitch [1, T_crepe], periodicity [1, T_crepe])
    # pitch 是 Hz, periodicity 是 [0,1] voicing-like 信心度
    pitch, periodicity = torchcrepe.predict(
        audio_t,
        sample_rate=sample_rate,
        hop_length=HOP_SIZE,
        fmin=F0_FMIN,
        fmax=F0_FMAX,
        model=CREPE_MODEL,
        return_periodicity=True,
        batch_size=TORCHCREPE_BATCH_SIZE,
        device=device,
    )

    # 3. 視需要做 viterbi smoothing（預設不做，保留歌聲細節）
    # 為什麼留這個分支：未來若發現某些資料集 F0 噪聲過多，可從 audio_config 開啟
    if CREPE_VITERBI:
        pitch = torchcrepe.filter.viterbi(pitch, periodicity)

    # 4. 用 periodicity threshold 把 unvoiced frame 的 F0 設為 0
    # 這個操作 inline 在 GPU 上完成，避免拉回 CPU 再 numpy 操作
    voiced = periodicity > confidence_threshold
    pitch = torch.where(voiced, pitch, torch.zeros_like(pitch))

    # 5. 拉回 CPU + numpy
    f0 = pitch.squeeze(0).cpu().numpy().astype(np.float32)
    voicing = periodicity.squeeze(0).cpu().numpy().astype(np.float32)

    return f0, voicing


def trim_or_pad_to_length(arr: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """
    把 1D array 對齊到指定長度（CREPE 與 mel STFT 邊界處理可能差 ±1 frame）。

    為什麼需要：
      mel = librosa.stft(...) 通常用 'center=True' padding，輸出 ceil(N/hop)+1 frames；
      torchcrepe 在邊界處可能少一個 frame。binarizer 統一用「mel frame 數」當基準長度，
      F0/PPG 都對齊到此長度。

    為什麼用 0 pad：
      F0 用 0 = unvoiced，PPG 用 0 = 無內容向量，都是合法的「無資訊」訊號，
      不會在訓練時誤導模型。
    """
    cur_len = arr.shape[0]
    if cur_len == target_len:
        return arr
    if cur_len > target_len:
        return arr[:target_len]
    # cur_len < target_len → pad
    pad = np.full((target_len - cur_len,) + arr.shape[1:], pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


if __name__ == "__main__":
    # 自我測試：合成一段 A4 (440 Hz) 純正弦 1 秒，CREPE 應抽出 ~440 Hz
    import time

    sr = SAMPLE_RATE
    duration_sec = 1.0
    freq = 440.0

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    print(f"Test: {freq} Hz sine wave, {duration_sec}s @ {sr} Hz")
    print(f"  Expected F0 = {freq} Hz across all voiced frames")
    print(f"  Expected frames ≈ {int(sr * duration_sec / HOP_SIZE)}")

    start = time.time()
    f0, voicing = extract_f0(audio, sample_rate=sr)
    elapsed = time.time() - start

    voiced_mask = f0 > 0
    print(f"\nResult:")
    print(f"  Frames returned: {len(f0)}")
    print(f"  Voiced frames:   {voiced_mask.sum()} / {len(f0)}")
    print(f"  Mean F0 (voiced): {f0[voiced_mask].mean():.2f} Hz "
          f"(error {abs(f0[voiced_mask].mean() - freq):.2f} Hz)")
    print(f"  Mean voicing conf: {voicing[voiced_mask].mean():.3f}")
    print(f"  Inference time:  {elapsed:.2f}s")
