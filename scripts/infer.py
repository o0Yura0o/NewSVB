"""
scripts/infer.py
==================

【這支檔案做什麼】
NSVB-ZH 推理 CLI。對應 rebuild_checklist §H 的兩種推理模式：

    Mode A：python -m scripts.infer
              --stage2-ckpt checkpoints/stage2/stage2_latest.pt
              --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt
              --input-a path/to/amateur.wav
              --output  path/to/out.wav

    Mode B：python -m scripts.infer
              --stage2-ckpt ...
              --vocoder-ckpt ...
              --input-a    path/to/amateur.wav
              --pro-ref    path/to/professional.wav
              --output     path/to/out.wav

【為什麼 CLI 而非 jupyter notebook】
- Linux 訓練機 deploy 後跑推理多走 ssh + bash 環境，notebook 不便
- 自動化批次推理（一晚跑 100 首做 listening test）需要可 scriptable 的 entry
- 與 vocoder_identity_test / audio_quality_probe 等其他 CLI 風格一致

【為什麼預設 dereverb=True / loudness=True】
與訓練 binarize 端對齊（Risk 2 主防線）；user 真的要關（已是乾淨 studio 音檔）才下
--no-dereverb / --no-loudness。

【可選 --kmeans-centroids】
推理本身不需要 phoneme_id（D_z 用，這裡不會跑 D_z），所以這個參數預設 None；
保留 entry 是為了 debug 場景（要視覺化 register / phoneme 分布時用）。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nsvb.inference import (
    InferenceFeatureExtractor,
    load_inference_models,
    run_mode_a,
    run_mode_b,
)
from nsvb.utils.audio_config import SAMPLE_RATE


def _save_wav(path: Path, wav: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    寫 16-bit int wav。

    為什麼 int16：
      多數播放器 / DAW 最相容；推理輸出 [-1, 1] float32 直接寫 float wav 雖然
      技術上可行，但部分舊播放器無法讀。clip 後乘 32767 是 industry standard。
    """
    from scipy.io import wavfile
    wav_clipped = np.clip(wav, -1.0, 1.0)
    wav_i16 = (wav_clipped * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, wav_i16)


def main():
    parser = argparse.ArgumentParser(
        description="NSVB-ZH inference (Mode A / Mode B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required: ckpt 路徑 ──
    parser.add_argument("--stage2-ckpt", required=True, type=Path,
                        help="Stage 2 ckpt (含 M + 對 Stage 1 ckpt 的引用)")
    parser.add_argument("--vocoder-ckpt", required=True, type=Path,
                        help="HifiGAN-NSF vocoder ckpt")
    parser.add_argument("--stage1-ckpt", default=None, type=Path,
                        help="覆寫 Stage 2 ckpt 內紀錄的 Stage 1 路徑（跨機器推理時用）")

    # ── Required: 輸入音檔 ──
    parser.add_argument("--input-a", required=True, type=Path,
                        help="業餘音檔 x_a")
    parser.add_argument("--pro-ref", default=None, type=Path,
                        help="專業 reference 音檔 x_p_ref（提供 → Mode B；不提供 → Mode A）")

    # ── 輸出 ──
    parser.add_argument("--output", required=True, type=Path,
                        help="輸出 wav 路徑")
    parser.add_argument("--save-mel", action="store_true",
                        help="同時把 decoder 出來的 mel 存成 .npy 供視覺化")

    # ── 前處理選項（與訓練端對齊預設）──
    parser.add_argument("--no-dereverb", action="store_true",
                        help="不跑 DeepFilterNet3 dereverb（已是乾淨 studio 才用）")
    parser.add_argument("--no-loudness", action="store_true",
                        help="不跑 BS.1770 loudness norm")
    parser.add_argument("--no-f0-interp", action="store_true",
                        help="vocoder 前不做 F0 log-interp（debug 用，會產生電音）")

    # ── DTW 選項（Mode B 才用）──
    parser.add_argument("--dtw-metric", default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Mode B DTW 距離度量")

    # ── 其他 ──
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                        help="預設 auto")
    parser.add_argument("--kmeans-centroids", default=None, type=Path,
                        help="PPG k-means centroids (.npy)；提供則一併抽 phoneme_id "
                             "供 debug；推理本身不需要")
    parser.add_argument("--whisper-model", default=None,
                        help="覆寫 audio_config 預設 whisper model "
                             "(e.g. openai/whisper-tiny for smoke test)")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mode = "B" if args.pro_ref is not None else "A"

    print(f"[infer] device={device}  mode={mode}", flush=True)
    print(f"[infer] input_a={args.input_a}", flush=True)
    if mode == "B":
        print(f"[infer] pro_ref={args.pro_ref}", flush=True)

    # ── 1. 載 model ──
    t0 = time.time()
    models = load_inference_models(
        stage2_ckpt=args.stage2_ckpt,
        vocoder_ckpt=args.vocoder_ckpt,
        device=device,
        stage1_ckpt=args.stage1_ckpt,
    )
    print(f"[infer] model load: {time.time()-t0:.1f}s", flush=True)

    # ── 2. 載 k-means centroids (optional) ──
    centroids = None
    if args.kmeans_centroids:
        centroids = np.load(str(args.kmeans_centroids))
        print(f"[infer] loaded centroids {centroids.shape}", flush=True)

    # ── 3. 抽特徵 ──
    extractor = InferenceFeatureExtractor(
        device=device,
        kmeans_centroids=centroids,
        whisper_model_name=args.whisper_model,
    )

    t0 = time.time()
    feat_a = extractor.extract(
        args.input_a,
        dereverb=not args.no_dereverb,
        loudness=not args.no_loudness,
    )
    print(f"[infer] features_a: T_mel={feat_a.t_mel}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    feat_p: Optional = None
    if mode == "B":
        t0 = time.time()
        feat_p = extractor.extract(
            args.pro_ref,
            dereverb=not args.no_dereverb,
            loudness=not args.no_loudness,
        )
        print(f"[infer] features_p_ref: T_mel={feat_p.t_mel}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ── 4. 推理 ──
    t0 = time.time()
    if mode == "A":
        result = run_mode_a(
            models, feat_a,
            apply_f0_interp=not args.no_f0_interp,
        )
    else:
        result = run_mode_b(
            models, feat_a, feat_p,
            apply_f0_interp=not args.no_f0_interp,
            dtw_metric=args.dtw_metric,
        )
    elapsed = time.time() - t0
    duration = len(result.wav) / SAMPLE_RATE
    print(f"[infer] forward: {elapsed:.2f}s  "
          f"output_duration={duration:.2f}s  "
          f"RTF={elapsed/duration:.2f}x", flush=True)
    if result.dtw_cost is not None:
        print(f"[infer] DTW cost (normalized): {result.dtw_cost:.4f}", flush=True)

    # ── 5. 輸出 ──
    _save_wav(args.output, result.wav)
    print(f"[infer] wav saved: {args.output}", flush=True)

    if args.save_mel:
        mel_path = args.output.with_suffix(".mel.npy")
        np.save(str(mel_path), result.mel)
        print(f"[infer] mel saved: {mel_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())