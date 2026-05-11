"""
scripts/cluster_ppg_inspect.py
=================================

【這支檔案做什麼】
Phase 0 完成 cluster_ppg.py 後的 sanity check：檢查 Whisper-PPG → k-means 的
`phoneme_id` 是否真的是「純 content cluster」，還是被 pitch 污染。

對應的擔憂見 [risk.md](../risk.md) 監控項，背景：
- Whisper 是 speech-heavy 模型；唱歌的延音、vibrato、melisma、非語音段可能讓
  hidden state 混入 pitch / prosody 資訊
- k-means 對這種「同一母音但不同 pitch」會產生獨立的 cluster
- → `phoneme_id` 序列與 `register_id` 序列高度相關 → D_z 的「phoneme + register」
   實質上變成單一強 condition（隱形 F0 shortcut）

【兩種診斷】
1. **量化**：
   - `cluster_register_mi`：phoneme_id 與 register_id 的 mutual information（bit）
     健康 < 0.3 bit（cluster 不太能預測 register）
     警訊 > 0.6 bit（cluster 強烈依賴 register → 被 pitch 污染）
   - `mean_dwell_frames_voiced`：voiced 段內 cluster 的平均連續長度（frame）
     健康 ≥ 8 frames（~90 ms，跨單一發音穩定）
     警訊 < 3 frames（cluster 切太細，可能 frame-by-frame 跟 pitch 跳）

2. **視覺化**：
   - 抽 N 首歌，畫 phoneme_id timeseries + F0 + voicing 疊圖
   - 直觀看「sustained 音段」「vibrato 段」「換氣段」cluster 是否穩定

【為什麼放在 cluster_ppg 之後 vs phase 1 之前】
這個 check 是純讀已 binarize 的 .npz 計算，不跑 model；訓練前發現 cluster 不健康
可以提早重 fit k-means（換 K、換 layer、加正規化），比訓到 Phase 2 才聽出 D_z 不穩
便宜很多。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# ── 健康閾值 ────────────────────────────────────────
# 為什麼選 0.3 / 0.6 bit：
#   完全獨立 → MI=0；完全依賴 → MI=min(H(p), H(r))，K=200 phoneme × 5 register
#   理論 max ~ log2(5) = 2.32 bit
#   經驗：合理 SVC 系統 MI 應在 0.1-0.3 bit；> 0.6 表 cluster 強烈跟 register 綁定
MI_HEALTHY = 0.3
MI_WARNING = 0.6
# 為什麼選 8 / 3 frames（mel-rate 172 fps）：
#   8 frames ≈ 46 ms ≈ 一個短促音節的時長（健康）
#   3 frames ≈ 17 ms（< 一個塞音時長，cluster 切太細，明顯不穩）
DWELL_HEALTHY = 8.0
DWELL_WARNING = 3.0


def mutual_information(
    a: np.ndarray, b: np.ndarray,
    n_a: int, n_b: int,
) -> float:
    """
    計算離散變數 a, b 的 mutual information（bit）。

    為什麼自寫而不用 sklearn.mutual_info_score：
        sklearn 用 nats（自然對數），我們報 bits 更直覺；
        自寫順便能 print 邊際分布做 debug
    """
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    N = len(a)
    if N == 0:
        return 0.0

    # 2D contingency table
    joint = np.bincount(a * n_b + b, minlength=n_a * n_b).reshape(n_a, n_b)
    p_joint = joint.astype(np.float64) / max(N, 1)
    p_a = p_joint.sum(axis=1, keepdims=True)
    p_b = p_joint.sum(axis=0, keepdims=True)

    # MI = sum p_joint * log2(p_joint / (p_a * p_b))，0 處跳過
    # 為什麼 mask：log2(0) = -inf，會污染 sum
    valid = (p_joint > 0) & (p_a > 0) & (p_b > 0)
    log_ratio = np.zeros_like(p_joint)
    log_ratio[valid] = np.log2(p_joint[valid] / (p_a * p_b)[valid])
    mi = float((p_joint * log_ratio).sum())
    return mi


def cluster_dwell_lengths(ids: np.ndarray, mask: Optional[np.ndarray] = None) -> List[int]:
    """
    對單一序列計算每個 cluster 「連續同 id」的 run length list。

    Args:
        ids:  [T] int  cluster id 序列
        mask: [T] bool 只計入 mask=True 的 frame；連續性按原序列；遇 mask=False
              段中斷現有 run

    Returns:
        list of run lengths

    為什麼分段：voiced segment 內部 cluster 才有意義；unvoiced silence 無 phoneme
    """
    if len(ids) == 0:
        return []
    runs: List[int] = []
    cur_id = ids[0]
    cur_len = 0
    for i, x in enumerate(ids):
        in_mask = True if mask is None else bool(mask[i])
        if not in_mask:
            # 中斷
            if cur_len > 0:
                runs.append(cur_len)
                cur_len = 0
            continue
        if x == cur_id:
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append(cur_len)
            cur_id = x
            cur_len = 1
    if cur_len > 0:
        runs.append(cur_len)
    return runs


# ── 載入 ───────────────────────────────────────────
def load_npz_fields(npz_path: Path):
    """讀單一 .npz 的 phoneme_id / register_id / voicing / f0。"""
    with np.load(npz_path) as data:
        keys = list(data.files)
        if "phoneme_id" not in keys:
            raise KeyError(f"{npz_path}: phoneme_id missing；先跑 cluster_ppg.py")
        out = {
            "phoneme_id": data["phoneme_id"].astype(np.int64),
            "register_id": data["register_id"].astype(np.int64),
            "voicing": data["voicing"].astype(np.float32),
            "f0": data["f0"].astype(np.float32),
        }
    return out


def sample_npz(root: Path, n: int, seed: int = 42) -> List[Path]:
    """從 binarized dataset 目錄抽 n 個 .npz。"""
    candidates = sorted(root.rglob("*.npz"))
    # 排除 centroids file 等非 sample 檔
    candidates = [p for p in candidates if "centroids" not in p.name]
    if not candidates:
        raise RuntimeError(f"No .npz under {root}; 先跑 binarizer + cluster_ppg")
    rng = np.random.default_rng(seed)
    n = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(idx)]


# ── 量化指標 ─────────────────────────────────────────
def compute_quantitative_metrics(
    binarized_root: Path,
    dataset_name: str,
    n_samples: int = 200,
    phoneme_vocab_size: int = 200,
    register_vocab_size: int = 5,
    seed: int = 42,
) -> dict:
    """
    對抽樣的 .npz 計算 MI + dwell length 統計。

    為什麼用 sample 而非全部：
        全部 binarized 可能 ~20k files；抽 200 已能穩定估計 MI / dwell 統計，
        速度快 100x。Phase 0 sanity check 不需要嚴格全量。
    """
    dataset_dir = binarized_root / dataset_name
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset dir not found: {dataset_dir}")
    files = sample_npz(dataset_dir, n_samples, seed=seed)
    print(f"[inspect] {dataset_name}: sampling {len(files)} files for stats", flush=True)

    # 累積：
    all_phoneme = []        # for MI（含 voiced/unvoiced）
    all_register = []
    all_voiced_dwell = []   # voiced 段 dwell lengths

    for p in files:
        d = load_npz_fields(p)
        ph = d["phoneme_id"]
        reg = d["register_id"]
        # voicing：register_id = -1 視為 unvoiced（binarizer 內 sentinel）
        voiced = reg >= 0
        # MI 累積（只算 voiced，否則 unvoiced 大量 register=-1 偏 MI）
        if voiced.any():
            all_phoneme.append(ph[voiced])
            all_register.append(reg[voiced])
        # dwell：分 voiced 段算
        all_voiced_dwell.extend(cluster_dwell_lengths(ph, voiced))

    # MI
    if all_phoneme:
        ph_concat = np.concatenate(all_phoneme)
        reg_concat = np.concatenate(all_register)
        mi = mutual_information(ph_concat, reg_concat, phoneme_vocab_size, register_vocab_size)
    else:
        mi = float("nan")

    # Dwell stats
    dwell_arr = np.array(all_voiced_dwell) if all_voiced_dwell else np.array([0])

    return {
        "dataset": dataset_name,
        "n_files": len(files),
        "n_voiced_frames": int(sum(len(p) for p in all_phoneme)),
        "cluster_register_mi_bit": float(mi),
        "mean_dwell_frames_voiced": float(dwell_arr.mean()),
        "median_dwell_frames_voiced": float(np.median(dwell_arr)),
        "p10_dwell_frames_voiced": float(np.percentile(dwell_arr, 10)),
        "p90_dwell_frames_voiced": float(np.percentile(dwell_arr, 90)),
        "verdict_mi": (
            "HEALTHY" if mi < MI_HEALTHY else
            "MARGINAL" if mi < MI_WARNING else
            "WARNING (pitch-confounded)"
        ),
        "verdict_dwell": (
            "HEALTHY" if dwell_arr.mean() >= DWELL_HEALTHY else
            "MARGINAL" if dwell_arr.mean() >= DWELL_WARNING else
            "WARNING (cluster fragmenting)"
        ),
    }


# ── 視覺化 ─────────────────────────────────────────
def plot_timeseries(
    npz_path: Path,
    out_path: Path,
):
    """
    對單一檔案畫 phoneme_id + F0 + voicing 疊圖。

    為什麼把三張圖疊在一個 figure：
        三條時間序列 frame index 同步；疊圖才能用眼睛交叉檢查
        「sustained 音 / vibrato / 換氣」對應的 cluster 是否穩定
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_npz_fields(npz_path)
    ph = d["phoneme_id"]
    f0 = d["f0"]
    voicing = d["voicing"]
    T = len(ph)
    t = np.arange(T) / 172.27  # convert to seconds (mel fps)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    # subplot 1: phoneme_id (use scatter so changes are visible at frame resolution)
    axes[0].scatter(t, ph, s=2, c="steelblue", alpha=0.7)
    axes[0].set_ylabel("phoneme_id\n(k-means cluster)")
    axes[0].set_title(f"phoneme_id timeseries: {npz_path.stem}")
    axes[0].grid(True, alpha=0.3)

    # subplot 2: F0 (Hz), grey for unvoiced
    voiced = f0 > 0
    axes[1].plot(t[voiced], f0[voiced], "-", color="darkred", linewidth=0.7)
    axes[1].set_ylabel("F0 (Hz)\nvoiced only")
    axes[1].grid(True, alpha=0.3)

    # subplot 3: voicing confidence
    axes[2].plot(t, voicing, "-", color="darkgreen", linewidth=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("voicing\nconfidence")
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_dwell_histogram(
    dwell_lengths: List[int],
    out_path: Path,
    dataset_name: str,
):
    """畫 voiced 段內 cluster dwell length 的 histogram + 健康閾值線。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dwell_lengths:
        return
    arr = np.array(dwell_lengths)
    mean = arr.mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    # 用 log scale x 軸看分布
    bins = np.logspace(0, np.log10(max(arr.max(), 10)), 50)
    ax.hist(arr, bins=bins, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(mean, color="red", linewidth=1.5, label=f"mean={mean:.1f}")
    ax.axvline(DWELL_HEALTHY, color="green", linestyle="--", linewidth=1, label=f"healthy ≥ {DWELL_HEALTHY}")
    ax.axvline(DWELL_WARNING, color="orange", linestyle="--", linewidth=1, label=f"warning < {DWELL_WARNING}")
    ax.set_xlabel("dwell length (frames; mel-rate 172 fps)")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_name}: voiced-segment cluster dwell distribution")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()


# ── Main ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Inspect phoneme_id quality")
    ap.add_argument("--binarized-root", default="data/binarized",
                    help="cluster_ppg.py 寫回 phoneme_id 的根目錄")
    ap.add_argument("--datasets", nargs="+", default=["m4singer", "vocalverse"],
                    help="要檢查的 dataset 子目錄名")
    ap.add_argument("--n-stat-samples", type=int, default=200,
                    help="量化指標的抽樣數（MI / dwell 統計用）")
    ap.add_argument("--n-plot-samples", type=int, default=5,
                    help="畫 timeseries 圖的抽樣數（每 dataset）")
    ap.add_argument("--phoneme-vocab-size", type=int, default=200)
    ap.add_argument("--register-vocab-size", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/phase0_cluster_inspect")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.binarized_root)

    overall = {
        "binarized_root": str(root),
        "thresholds": {
            "mi_healthy": MI_HEALTHY, "mi_warning": MI_WARNING,
            "dwell_healthy": DWELL_HEALTHY, "dwell_warning": DWELL_WARNING,
        },
        "datasets": {},
    }

    for name in args.datasets:
        dataset_dir = root / name
        if not dataset_dir.is_dir():
            print(f"[inspect] skip {name}: dir not found {dataset_dir}", flush=True)
            continue

        print(f"\n[inspect] === {name} ===", flush=True)
        metrics = compute_quantitative_metrics(
            root, name,
            n_samples=args.n_stat_samples,
            phoneme_vocab_size=args.phoneme_vocab_size,
            register_vocab_size=args.register_vocab_size,
            seed=args.seed,
        )
        print(f"  MI(phoneme; register)  = {metrics['cluster_register_mi_bit']:.3f} bit "
              f"[{metrics['verdict_mi']}]")
        print(f"  mean voiced dwell     = {metrics['mean_dwell_frames_voiced']:.1f} frames "
              f"[{metrics['verdict_dwell']}]")
        print(f"  median voiced dwell   = {metrics['median_dwell_frames_voiced']:.1f} frames")
        print(f"  dwell p10/p90        = {metrics['p10_dwell_frames_voiced']:.1f} / "
              f"{metrics['p90_dwell_frames_voiced']:.1f} frames")

        # 視覺化抽樣
        plot_dataset_dir = out_dir / name
        files = sample_npz(root / name, args.n_plot_samples, seed=args.seed)
        for i, p in enumerate(files):
            png_path = plot_dataset_dir / f"timeseries_{i:02d}_{p.stem}.png"
            plot_timeseries(p, png_path)

        # Dwell histogram（用 stat sample 數據）
        files_stat = sample_npz(root / name, args.n_stat_samples, seed=args.seed)
        all_dwells: List[int] = []
        for p in files_stat:
            d = load_npz_fields(p)
            all_dwells.extend(cluster_dwell_lengths(d["phoneme_id"], d["register_id"] >= 0))
        plot_dwell_histogram(all_dwells, plot_dataset_dir / "dwell_histogram.png", name)

        overall["datasets"][name] = metrics
        print(f"  plots saved → {plot_dataset_dir}/")

    # 寫 report
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\n[inspect] report → {report_path}")

    # 整體 gate
    all_pass = all(
        d["verdict_mi"] == "HEALTHY" and d["verdict_dwell"] == "HEALTHY"
        for d in overall["datasets"].values()
    )
    if all_pass:
        print("[inspect] ALL HEALTHY — k-means cluster 表現良好，可進 Phase 1")
        sys.exit(0)
    print("[inspect] HAS WARNINGS — 看圖確認；常見補救：")
    print("  - MI WARNING（cluster 被 pitch 污染）：")
    print("    * 降 K（e.g. 200 → 100）讓 cluster 變粗")
    print("    * 換 Whisper layer（試 layer 6 或 10）")
    print("    * binarize 時對 PPG 做 per-utterance 去 DC：減去全曲均值")
    print("  - DWELL WARNING（cluster 切太細）：")
    print("    * 同上降 K 為主")
    print("    * 或 binarize 後對 phoneme_id 做 mode filter (window=5)")
    sys.exit(2)


if __name__ == "__main__":
    main()