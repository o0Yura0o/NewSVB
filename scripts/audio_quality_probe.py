"""
scripts/audio_quality_probe.py
=================================

【這支檔案做什麼】
量化兩個 dataset (m4singer, VocalVerse) 在「**錄音環境品質維度**」上的分布差異，
作為 Phase 0 gate 的補充指標（與 phoneme/register JSD 並列）。

針對 Risk 2（音質域與技術域混淆）的訓前驗證：
    若兩 dataset 的「環境差異」（殘響 / 噪聲 / 高頻能量）分布過大，
    D_z 會把這個差異當捷徑，M 學成「降噪 / 去殘響濾波器」而非「修技術」。

【量化指標（4 個）】
1. **Spectral Flatness Mean (SFM)** — Wiener 熵；訊號越偏「白噪」越接近 1，越偏「純音」越接近 0
   - 對「環境噪聲」非常敏感：noisy room 高 SFM、anechoic 低 SFM
   - 範圍 [0, 1]
2. **Reverb Estimate (T60-like)** — 用 spectral subtraction + decay slope 估計殘響時間
   - 殘響長 → 大值；乾 anechoic → 小值
   - 單位：秒（粗略，僅相對比較）
3. **High-Freq Energy Ratio** — 4-8 kHz / 全頻能量比例
   - 高品質錄音保留更多 4-8 kHz 細節；手機錄音常 high-cut
   - 範圍 [0, 1]
4. **SNR Estimate** — 用 voiced/unvoiced 段能量比近似（torchcrepe periodicity threshold）
   - voiced 是 signal、unvoiced + 靜音段視為 noise floor
   - 單位：dB

【為什麼這 4 個】
- SFM 對「白噪 / 環境噪聲」最敏感（最容易 D_z 抓的捷徑訊號）
- Reverb 對「家庭錄音 vs 錄音室」最 discriminative
- High-Freq Ratio 對「裝備品質」（mic / 取樣 / 壓縮）有強訊號
- SNR 對「總體錄音純淨度」有訊號

四個一起涵蓋音質維度的主要面向；單一 metric 不足。

【輸出】
- 每樣本 4 個 metric 值
- 每 dataset 的 metric 分布 histogram + mean/std
- 兩 dataset 對應 metric 的 JSD（histogram bin freq）
- 是否 PASS：所有 4 個 JSD < 0.10（比 phoneme/register 鬆，因為環境本來就會有差）

執行：
    PYTHONPATH=. python scripts/audio_quality_probe.py \\
        --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \\
        --n-per-dir 100 \\
        --out-dir outputs/phase0_audio_quality
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from nsvb.utils.audio_config import SAMPLE_RATE
from nsvb.utils.jsd_check import jensen_shannon_divergence


# ── 通過閾值（4 個 JSD 都要過）─────────────────────────
JSD_PASS_THRESHOLD = 0.10  # 比 phoneme/register JSD 寬鬆（環境差異本身就難完全消除）


# ── Metric 計算 ─────────────────────────────────────────
def spectral_flatness_mean(wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Wiener 熵：geometric_mean(spec) / arithmetic_mean(spec) per frame，再對 frame 求 mean。

    為什麼 mean 而非 max：
        環境噪聲是持續性的，平均能反映整體；max 容易被瞬時尖峰污染
    """
    sf = librosa.feature.spectral_flatness(y=wav)  # [1, T]
    return float(sf.mean())


def reverb_estimate(wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    粗略殘響時間估計：用 energy decay 在 silent 段的衰減斜率。

    為什麼不用標準 RT60（Schroeder integration）：
        標準 RT60 需要明確的 impulse response 起點；歌聲段沒有 impulse 邊界。
        這裡用 simpler proxy：把 spectrogram 取 db，找 energy 從 max 衰減 -30 dB 的時間。

    返回值：秒（粗略；同 dataset 內可比較，跨 dataset 數值不要太較真）
    """
    # spectrogram in dB
    n_fft = 512
    S = np.abs(librosa.stft(y=wav, n_fft=n_fft, hop_length=128)) ** 2
    S_db = librosa.power_to_db(S.sum(axis=0), ref=np.max)  # [T] frame-level energy in dB

    # 找局部峰值，量該峰之後 energy 衰減 -30 dB 所花時間
    # 為什麼 -30 dB：標準 RT60 是 -60 dB，但歌聲段難達到 -60 dB（會被下一音符切斷）；
    # -30 dB 在歌聲 phrase 之間 silent 段較好量
    peaks_idx = np.argsort(S_db)[-min(10, len(S_db) // 100):]
    decay_times_sec = []
    for p in peaks_idx:
        # 從 peak 往後找 -30 dB
        target = S_db[p] - 30
        for j in range(p + 1, min(p + 200, len(S_db))):
            if S_db[j] < target:
                decay_times_sec.append((j - p) * 128 / sr)
                break
    if not decay_times_sec:
        return 0.0
    # 為什麼 median 而非 mean：殘響估計受極端值（短 phrase 結尾）影響大，median 更穩
    return float(np.median(decay_times_sec))


def high_freq_energy_ratio(wav: np.ndarray, sr: int = SAMPLE_RATE,
                           f_low: float = 4000, f_high: float = 8000) -> float:
    """
    4-8 kHz 能量 / 全頻能量。手機錄音常 < 4kHz cutoff，比例會明顯較低。
    """
    n_fft = 1024
    S = np.abs(librosa.stft(y=wav, n_fft=n_fft, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # 為什麼用 sum 而非 mean：能量比例本來就要 sum
    band_mask = (freqs >= f_low) & (freqs < f_high)
    total = S.sum() + 1e-10
    band = S[band_mask].sum()
    return float(band / total)


def snr_estimate(wav: np.ndarray, sr: int = SAMPLE_RATE,
                 voicing_threshold: float = 0.5) -> float:
    """
    粗略 SNR：voiced 段平均能量 / unvoiced(noise floor) 段平均能量，dB。

    為什麼用 voicing 而非靜音段：
        歌聲樣本可能沒有「明確的純靜音」（呼吸聲也算 unvoiced），voicing periodicity
        threshold 可以區分「有 fundamental」與「無 fundamental」段，後者作 noise proxy。
    """
    import torchcrepe
    import torch
    audio_t = torch.from_numpy(wav).float().unsqueeze(0)
    pitch, periodicity = torchcrepe.predict(
        audio_t, sample_rate=sr, hop_length=256,
        fmin=50.0, fmax=1100.0, model="tiny",   # tiny 加速：SNR 不需高精度 F0
        return_periodicity=True, batch_size=2048,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    voiced_mask = (periodicity.squeeze(0) > voicing_threshold).cpu().numpy()  # [T]
    # frame-level energy
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(y=wav, n_fft=n_fft, hop_length=hop)) ** 2
    energy_per_frame = S.sum(axis=0)  # [T_frames]

    T = min(len(voiced_mask), len(energy_per_frame))
    voiced_mask = voiced_mask[:T]
    energy_per_frame = energy_per_frame[:T]

    voiced_e = energy_per_frame[voiced_mask].mean() if voiced_mask.sum() > 0 else 1e-10
    unvoiced_e = energy_per_frame[~voiced_mask].mean() if (~voiced_mask).sum() > 0 else 1e-10
    return float(10 * np.log10(voiced_e / max(unvoiced_e, 1e-10)))


# ── per-dataset 抽樣 + metric 計算 ──────────────────────
def compute_dataset_metrics(
    root: Path,
    n: int,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    從 root 遞迴抽 n 個 wav，對每個算 4 個 metric，回傳 dict of lists。
    """
    candidates = sorted(root.rglob("*.wav"))
    if not candidates:
        raise RuntimeError(f"No .wav under {root}")
    rng = np.random.default_rng(seed)
    n = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=n, replace=False)

    metrics: Dict[str, List[float]] = {
        "sfm": [], "reverb_sec": [], "hf_ratio": [], "snr_db": [],
    }
    for i, k in enumerate(sorted(idx)):
        path = candidates[k]
        try:
            wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            metrics["sfm"].append(spectral_flatness_mean(wav))
            metrics["reverb_sec"].append(reverb_estimate(wav))
            metrics["hf_ratio"].append(high_freq_energy_ratio(wav))
            metrics["snr_db"].append(snr_estimate(wav))
        except Exception as e:
            print(f"[{root.name}] WARN: {path.name} skipped: {e}", flush=True)

        if (i + 1) % 20 == 0:
            print(f"[{root.name}] processed {i+1}/{len(idx)}", flush=True)

    return metrics


# ── JSD per metric ───────────────────────────────────────
def metric_jsd(values_a: List[float], values_b: List[float],
               n_bins: int = 30) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    把兩個 metric value list 各自做 histogram，然後算 JSD。

    為什麼共用同 bin edges：
        若兩邊各自 auto bin 不對齊，JSD 沒意義；用聯合 min/max 算統一 bin。

    Returns:
        jsd, hist_a, hist_b, bin_edges
    """
    all_vals = np.concatenate([values_a, values_b])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    hist_a, _ = np.histogram(values_a, bins=bin_edges)
    hist_b, _ = np.histogram(values_b, bins=bin_edges)
    # 轉概率
    pa = hist_a / max(hist_a.sum(), 1)
    pb = hist_b / max(hist_b.sum(), 1)
    jsd = jensen_shannon_divergence(pa, pb)
    return jsd, hist_a, hist_b, bin_edges


# ── 主 verdict ───────────────────────────────────────────
def verdict(metric_jsds: Dict[str, float]) -> str:
    """全部 metric 的 JSD 都要 < threshold 才 PASS。"""
    all_pass = all(j < JSD_PASS_THRESHOLD for j in metric_jsds.values())
    return "PASS" if all_pass else "FAIL"


# ── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 audio quality probe (Risk 2 補強 gate)"
    )
    parser.add_argument(
        "--wav-dirs", nargs="+", required=True,
        help='格式 "name=path"；至少給兩個 dataset 才能算 JSD',
    )
    parser.add_argument(
        "--n-per-dir", type=int, default=100,
        help="每個資料集抽多少 wav（預設 100；要降到 30 也能 work）",
    )
    parser.add_argument(
        "--out-dir", default="outputs/phase0_audio_quality",
        help="report.json 輸出位置",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 name=path
    dirs: Dict[str, Path] = {}
    for spec in args.wav_dirs:
        if "=" not in spec:
            print(f"ERROR: --wav-dirs 應為 'name=path' 格式，得到 {spec!r}")
            sys.exit(1)
        name, path = spec.split("=", 1)
        dirs[name] = Path(path)

    if len(dirs) < 2:
        print("ERROR: 需要至少 2 個 dataset 才能算 JSD")
        sys.exit(1)

    # 對每個 dataset 抽樣 + 算 metrics
    per_dataset_metrics: Dict[str, Dict[str, List[float]]] = {}
    for name, root in dirs.items():
        print(f"\n[probe] === {name} @ {root} ===", flush=True)
        if not root.exists():
            print(f"[probe] WARNING: {root} not found, skipping")
            continue
        per_dataset_metrics[name] = compute_dataset_metrics(
            root, n=args.n_per_dir, seed=args.seed,
        )

    if len(per_dataset_metrics) < 2:
        print("ERROR: 有效 dataset 不足 2 個，無法計算 JSD")
        sys.exit(1)

    # 計算兩 dataset 兩兩 JSD（只支援第 1, 2 個）
    names = list(per_dataset_metrics.keys())
    a_name, b_name = names[0], names[1]
    metric_keys = ["sfm", "reverb_sec", "hf_ratio", "snr_db"]

    print(f"\n[probe] === JSD between '{a_name}' and '{b_name}' ===", flush=True)
    metric_jsds: Dict[str, float] = {}
    summary = {a_name: {}, b_name: {}, "jsds": {}}
    for mk in metric_keys:
        a_vals = per_dataset_metrics[a_name][mk]
        b_vals = per_dataset_metrics[b_name][mk]

        a_mean, a_std = float(np.mean(a_vals)), float(np.std(a_vals))
        b_mean, b_std = float(np.mean(b_vals)), float(np.std(b_vals))
        summary[a_name][mk] = {"mean": a_mean, "std": a_std, "n": len(a_vals)}
        summary[b_name][mk] = {"mean": b_mean, "std": b_std, "n": len(b_vals)}

        jsd, _, _, _ = metric_jsd(a_vals, b_vals)
        metric_jsds[mk] = jsd
        summary["jsds"][mk] = jsd

        flag = "✅" if jsd < JSD_PASS_THRESHOLD else "❌"
        print(f"  {mk:14s}  {a_name}: {a_mean:.4f}±{a_std:.4f}   "
              f"{b_name}: {b_mean:.4f}±{b_std:.4f}   "
              f"JSD={jsd:.4f}  {flag}")

    v = verdict(metric_jsds)
    summary["verdict"] = v
    summary["jsd_threshold"] = JSD_PASS_THRESHOLD

    # 報告
    report_path = out_dir / "audio_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[probe] report saved to {report_path}", flush=True)

    if v == "PASS":
        print(f"[probe] ✅ ALL METRICS JSD < {JSD_PASS_THRESHOLD} — 音質分布對齊，可進 Stage 1")
        sys.exit(0)
    else:
        bad = [k for k, j in metric_jsds.items() if j >= JSD_PASS_THRESHOLD]
        print(f"[probe] ❌ FAIL on metrics: {bad}")
        print(f"  → 建議：(a) 加強 dereverb/denoise 預處理；")
        print(f"           (b) 重採樣或丟棄極端音質樣本；")
        print(f"           (c) 替換 dataset")
        sys.exit(2)


if __name__ == "__main__":
    main()