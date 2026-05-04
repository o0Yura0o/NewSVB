"""
scripts/sanity_vocoder_nsvb.py
================================

Sanity check：直接調用 NSVB **原 repo** 的 vocoder code 來重建 wav。

【為什麼需要這個】
我們的 `nsvb/backbone/vocoder/hifigan_nsf.py` 與 1012 ckpt 244 keys strict-load 通過，
但 vocoder identity test 的重建仍有電音與音高跑掉。可能是：
  (A) 我們的 hifigan_nsf.py forward 路徑與 NSVB 原版有微小差異（forward pass 不同）
  (B) NSVB 原版 vocoder 在 spec2wav 之前對 mel/F0 做了我們沒做的處理

這支腳本繞過我們的 backbone，直接 sys.path 把 NSVB 原 repo 接進來，
用它的 `vocoders.hifigan.HifiGAN().spec2wav(mel, f0=f0)` 跑重建。

如果原版聽起來也電音 → 問題在 mel/F0 餵法（與 hifigan_nsf 無關）
如果原版聽起來正常 → 問題在我們的 forward 實作

【執行】
    PYTHONPATH=. python scripts/sanity_vocoder_nsvb.py \
        --wav data/m4singer/Alto-1#newboy/0011.wav \
        --out-dir outputs/sanity_vocoder_nsvb
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

# 我們自己的 audio_io 算 mel（與 NSVB process_utterance bit-exact）
from nsvb.utils.audio_config import SAMPLE_RATE
from nsvb.utils.audio_io import compute_mel
from nsvb.data.feature_extract.f0_torchcrepe import extract_f0


# 為什麼把 NSVB repo 路徑加到 sys.path：
# 直接 import nsvb 自己的程式不夠 — 我們要 import NSVB **原版** 的 vocoders.hifigan，
# 才能驗證問題不在我們的 hifigan_nsf 移植。
# 透過環境變數 NSVB_REPO_ROOT 指定原 NSVB repo 的位置（跨平台）。
NSVB_REPO_ROOT = os.environ.get("NSVB_REPO_ROOT")
if NSVB_REPO_ROOT and Path(NSVB_REPO_ROOT).is_dir():
    sys.path.insert(0, NSVB_REPO_ROOT)
else:
    print("[nsvb-sanity] WARNING: 環境變數 NSVB_REPO_ROOT 未設或不存在，"
          "將無法 import NSVB 原版 modules.hifigan.hifigan", file=sys.stderr)


def load_nsvb_vocoder(ckpt_dir: Path, device: str):
    """
    用 NSVB 原 repo 的 HifiGanGenerator 載入 1012 ckpt。

    為什麼自己直接 instantiate 而不走 NSVB 的 PWG class wrapper：
    那個 class 依賴 hparams 全域 state；自己 load 簡單可控。
    """
    import yaml
    from modules.hifigan.hifigan import HifiGanGenerator

    config_path = ckpt_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        h = yaml.safe_load(f)

    # 找最新的 ckpt
    ckpts = sorted(ckpt_dir.glob("model_ckpt_steps_*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No ckpt under {ckpt_dir}")
    ckpt_path = ckpts[-1]

    # NSVB 原版 HifiGanGenerator 構造方式：吃整個 hparams dict，自己讀 keys
    g = HifiGanGenerator(h, c_out=1)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state["state_dict"]["model_gen"]
    g.load_state_dict(sd, strict=True)
    g.remove_weight_norm()
    g = g.to(device).eval()
    print(f"[nsvb-sanity] loaded NSVB original HifiGanGenerator from {ckpt_path}")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/m4singer/Alto-1#newboy/0011.wav")
    # 預設從環境變數推導；訓練 / debug 機路徑各異時用 --ckpt-dir override
    default_ckpt_dir = (
        Path(NSVB_REPO_ROOT) / "checkpoints" / "1012_hifigan_all_songs_nsf"
        if NSVB_REPO_ROOT else None
    )
    ap.add_argument("--ckpt-dir",
                    default=str(default_ckpt_dir) if default_ckpt_dir else None,
                    required=default_ckpt_dir is None,
                    help="NSVB 1012_hifigan_all_songs_nsf ckpt dir；"
                         "未設時從 $NSVB_REPO_ROOT/checkpoints/1012_... 推導")
    ap.add_argument("--out-dir", default="outputs/sanity_vocoder_nsvb")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: wav not found: {wav_path}")
        sys.exit(1)

    # 1. 載入原 NSVB vocoder
    g_nsvb = load_nsvb_vocoder(Path(args.ckpt_dir), args.device)

    # 2. mel + F0：用 NSVB 原版邏輯（process_utterance + get_pitch）抽
    # 由於 NSVB 用 librosa 0.7 的 positional API、新版 librosa 不相容，
    # 這裡 inline 重寫等價邏輯，與 NSVB process_utterance / get_pitch 數值一致
    import librosa
    import parselmouth
    from nsvb.utils.audio_config import (
        SAMPLE_RATE, HOP_SIZE, FFT_SIZE, WIN_SIZE, NUM_MELS, MEL_FMIN, MEL_FMAX,
    )
    print("[nsvb-sanity] inlined NSVB process_utterance + get_pitch ...")
    wav_np, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)

    # process_utterance（vocoder='pwg' 分支，eps=1e-10）
    stft = librosa.stft(y=wav_np, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
                        win_length=WIN_SIZE, window="hann", pad_mode="constant")
    spc = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE,
                                    n_mels=NUM_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)
    mel_nsvb_native = (mel_basis @ spc)
    mel_nsvb_native = np.log10(np.maximum(1e-10, mel_nsvb_native)).T.astype(np.float32)
    # 這正是 NSVB process_utterance 的 vocoder='pwg' 分支等價

    # get_pitch（parselmouth, f0_min=80, f0_max=750, voicing_threshold=0.6）
    time_step = HOP_SIZE / SAMPLE_RATE * 1000  # ms
    f0_min, f0_max = 80, 750
    pad_size = 4 if HOP_SIZE == 128 else 2
    f0_raw = parselmouth.Sound(wav_np, SAMPLE_RATE).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max,
    ).selected_array["frequency"]
    lpad = pad_size * 2
    rpad = max(0, len(mel_nsvb_native) - len(f0_raw) - lpad)
    f0_nsvb_native = np.pad(f0_raw, [[lpad, rpad]], mode="constant")
    delta_l = len(mel_nsvb_native) - len(f0_nsvb_native)
    if delta_l > 0:
        f0_nsvb_native = np.concatenate([f0_nsvb_native, [f0_nsvb_native[-1]] * delta_l], 0)
    f0_nsvb_native = f0_nsvb_native[: len(mel_nsvb_native)].astype(np.float32)

    # 也跑我們的版本對比
    import librosa
    wav_ours, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    mel_ours = compute_mel(wav_ours)
    f0_ours, _ = extract_f0(wav_ours, sample_rate=SAMPLE_RATE, device=args.device)
    T_mel_o = mel_ours.shape[0]
    if f0_ours.shape[0] < T_mel_o:
        f0_ours = np.concatenate([f0_ours, np.zeros(T_mel_o - f0_ours.shape[0], dtype=f0_ours.dtype)])
    else:
        f0_ours = f0_ours[:T_mel_o]

    print(f"\n--- mel comparison ---")
    print(f"  NSVB native: shape={mel_nsvb_native.shape}, range=[{mel_nsvb_native.min():.3f}, {mel_nsvb_native.max():.3f}]")
    print(f"  Ours:        shape={mel_ours.shape}, range=[{mel_ours.min():.3f}, {mel_ours.max():.3f}]")
    T_min = min(mel_nsvb_native.shape[0], mel_ours.shape[0])
    print(f"  mel diff max  = {np.abs(mel_nsvb_native[:T_min] - mel_ours[:T_min]).max():.6e}")

    print(f"\n--- F0 comparison ---")
    print(f"  NSVB native (parselmouth): shape={f0_nsvb_native.shape}, voiced={(f0_nsvb_native>0).mean():.2%}")
    print(f"  Ours (torchcrepe):         shape={f0_ours.shape}, voiced={(f0_ours>0).mean():.2%}")

    # 用 NSVB 抽的 mel+f0 跑 vocoder
    mel = mel_nsvb_native
    f0 = f0_nsvb_native
    print(f"\n[nsvb-sanity] feeding NSVB-extracted mel+f0 to vocoder")

    # 3. 餵 NSVB 原版 vocoder（介面：mel [B, 80, T], f0 [B, T] in Hz）
    mel_t = torch.from_numpy(mel).float().T.unsqueeze(0).to(args.device)  # [1, 80, T]
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(args.device)       # [1, T]

    # 固定 seed 確保 SineGen 內部的 random initial phase 兩次調用相同
    # 為什麼必要：SineGen.forward 用 torch.rand 生 rand_ini，不固定 seed → 兩次
    # forward 必然不同，但這只是隨機相位差，非真正 forward 邏輯差異
    torch.manual_seed(42)
    with torch.no_grad():
        wav_recon_nsvb = g_nsvb(mel_t, f0_t).squeeze().cpu().numpy()

    print(f"NSVB recon: {wav_recon_nsvb.shape}, "
          f"range [{wav_recon_nsvb.min():.3f}, {wav_recon_nsvb.max():.3f}]")

    # 4. 同時也跑我們的 hifigan_nsf 比對
    from nsvb.backbone.vocoder import HifiGanNSFGenerator
    g_ours = HifiGanNSFGenerator(c_out=1, num_mels=80, audio_sample_rate=22050)
    state = torch.load(
        str(Path(args.ckpt_dir) / "model_ckpt_steps_1170000.ckpt"),
        map_location="cpu", weights_only=False,
    )
    g_ours.load_state_dict(state["state_dict"]["model_gen"], strict=True)
    g_ours.remove_weight_norm()
    g_ours = g_ours.to(args.device).eval()
    torch.manual_seed(42)  # 同樣 seed 跑 ours
    with torch.no_grad():
        wav_recon_ours = g_ours(mel_t, f0_t).squeeze().cpu().numpy()

    print(f"Ours recon: {wav_recon_ours.shape}, "
          f"range [{wav_recon_ours.min():.3f}, {wav_recon_ours.max():.3f}]")

    # 5. 直接比兩 wav 的數值差（forward 是否 bit-exact 等價）
    T = min(len(wav_recon_nsvb), len(wav_recon_ours))
    a = wav_recon_nsvb[:T]
    b = wav_recon_ours[:T]
    abs_diff = np.abs(a - b)
    print(f"\n--- NSVB vs ours wav diff ---")
    print(f"  max  = {abs_diff.max():.6e}")
    print(f"  mean = {abs_diff.mean():.6e}")
    if abs_diff.max() < 1e-4:
        print("✅ 兩個 forward 數值幾乎一致")
    else:
        print("❌ 兩個 forward 不同，我們的 hifigan_nsf 移植有差")

    # 6. 存 wav 給人耳聽測
    stem = wav_path.stem
    gt_wav = wav_np[: len(wav_recon_nsvb)]
    for name, w in [("gt", gt_wav),
                    ("nsvb_orig", wav_recon_nsvb),
                    ("ours_port", wav_recon_ours)]:
        clipped = np.clip(w, -1.0, 1.0) * 32767
        path = out_dir / f"{stem}__{name}.wav"
        wavfile.write(path, SAMPLE_RATE, clipped.astype(np.int16))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
