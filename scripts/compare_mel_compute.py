"""
scripts/compare_mel_compute.py
================================

對比兩種 mel 計算的數值差異：
    A) `nsvb.utils.audio_io.compute_mel`             (我們的版本)
    B) NSVB 訓練 vocoder 真正用的 mel 公式
       — port `data_gen/tts/data_gen_utils.py:process_utterance`
         (vocoder='pwg', eps=1e-10) 的純 mel 計算部分

【為什麼要 port 而非直接 import】
process_utterance 依賴 NSVB 的 hparams 全域 state、`utils/audio.py` 的 padding helper、
以及 librosa 對 wav2spec_eps 的處理；直接 import 會拖入太多東西。
這裡逐字 port 計算核心，保證 bit-exact 對齊我們關心的 hyperparameters
（hop=128, fft=512, win=512, fmin=0, fmax=8000, eps=1e-10, log10）。

執行：
    PYTHONPATH=. python scripts/compare_mel_compute.py [--wav PATH]
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

from nsvb.utils.audio_config import (
    SAMPLE_RATE, HOP_SIZE, FFT_SIZE, WIN_SIZE, NUM_MELS, MEL_FMIN, MEL_FMAX,
)
from nsvb.utils.audio_io import compute_mel as compute_mel_ours


def compute_mel_nsvb(y_np: np.ndarray) -> np.ndarray:
    """
    Bit-exact port 自 NSVB `data_gen/tts/data_gen_utils.py:process_utterance`
    （vocoder='pwg' 分支，wav2spec_eps=1e-10），這是訓練 1012_hifigan_all_songs_nsf
    時實際走的 mel 公式。
    """
    # 1. STFT — librosa 預設 center=True, window='hann'
    stft = librosa.stft(
        y=y_np,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
        window="hann",
        pad_mode="constant",
    )
    spc = np.abs(stft)  # [n_fft//2+1, T]

    # 2. mel basis
    # 為什麼這裡 fmin/fmax 也走 librosa.filters.mel：與我們 audio_io 完全相同實作
    fmin_resolved = 0 if MEL_FMIN == -1 else MEL_FMIN
    fmax_resolved = SAMPLE_RATE / 2 if MEL_FMAX == -1 else MEL_FMAX
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        n_mels=NUM_MELS,
        fmin=fmin_resolved,
        fmax=fmax_resolved,
    )
    mel = mel_basis @ spc  # [NUM_MELS, T]

    # 3. log10 with eps=1e-10 (NSVB pwg.py 預設值)
    mel = np.log10(np.maximum(1e-10, mel))

    # NSVB 回傳 [NUM_MELS, T]；我們 transpose 成 [T, NUM_MELS]
    return mel.T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/m4singer/Alto-1#newboy/0000.wav")
    args = ap.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: {wav_path} not found")
        sys.exit(1)

    print(f"=== Mel computation comparison ===")
    print(f"Wav: {wav_path}")
    print(f"Config: sr={SAMPLE_RATE} hop={HOP_SIZE} fft={FFT_SIZE} "
          f"win={WIN_SIZE} mels={NUM_MELS} fmin={MEL_FMIN} fmax={MEL_FMAX}")
    print()

    # 為什麼直接 librosa.load：這是 NSVB 訓練時的 wav 載入方式（process_utterance line 111）
    wav, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    print(f"wav: shape={wav.shape}, range=[{wav.min():.4f}, {wav.max():.4f}], sr={SAMPLE_RATE}")
    print()

    mel_ours = compute_mel_ours(wav)
    print(f"A) audio_io.compute_mel:")
    print(f"   shape={mel_ours.shape}  "
          f"range=[{mel_ours.min():.3f}, {mel_ours.max():.3f}]  "
          f"mean={mel_ours.mean():.3f}  std={mel_ours.std():.3f}")

    mel_nsvb = compute_mel_nsvb(wav)
    print(f"B) NSVB process_utterance port:")
    print(f"   shape={mel_nsvb.shape}  "
          f"range=[{mel_nsvb.min():.3f}, {mel_nsvb.max():.3f}]  "
          f"mean={mel_nsvb.mean():.3f}  std={mel_nsvb.std():.3f}")
    print()

    T = min(mel_ours.shape[0], mel_nsvb.shape[0])
    a, b = mel_ours[:T], mel_nsvb[:T]
    abs_diff = np.abs(a - b)
    print(f"--- diff |A - B| (T={T}) ---")
    print(f"  max  = {abs_diff.max():.6e}")
    print(f"  mean = {abs_diff.mean():.6e}")
    print(f"  p99  = {np.percentile(abs_diff, 99):.6e}")
    print()

    if abs_diff.max() < 1e-5:
        print("✅ Mel 計算與 NSVB process_utterance bit-exact 對齊。")
        print("   電音問題在於別處（F0 / vocoder forward / weight 載入 norm 不對）。")
    else:
        print(f"❌ 仍有數值差，需修 audio_io.compute_mel。")


if __name__ == "__main__":
    main()
