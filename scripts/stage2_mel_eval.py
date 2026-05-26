"""Stage 2 mel-domain evaluation across multiple step ckpts.

跟 stage2_ckpts_listening 同 inference flow,但 **跳過 vocoder** + 著重 mel 域指標 +
NSVB spec_f0_to_figure 風格視覺化。給「vocoder 電音影響聽測判斷」場景用。

【為什麼存在】
Stage 2 訓出來後人耳聽測會被 amateur-specific 的電音遮蔽(Stage 1 端 FVAE/F0 跟 amateur
mel 不夠 clean,跟 M 無關)。所以在 mel 域繞過 vocoder 看 M 到底改了什麼。

【設計重點】
- 不重跑 inference 模式:若 --listening-dir 給了 (含 .mel.npy 檔),直接讀,無需 GPU
- 自己跑 inference 模式:給 --stage1-ckpt + --stage2-ckpt-dir,fresh forward(需 GPU)
- pro 參考:從 binarized/m4singer/ 採樣 N 個 mels 算 mean envelope([NUM_MELS]),
            給 mel_pro_dist 與 modification direction alignment 用

【產出】
{out_dir}/
├── pro_mean_env.npy                       # cached pro 參考 envelope
├── {ds}_{sample}/
│   ├── mel_grid.png                       # NSVB spec_f0_to_figure stack
│   └── metrics.json                       # 該 sample 每 step 的 metric 值
├── metrics_aggregate.csv                  # 全 sample × step × metric 矩陣
└── report.md                              # per-step aggregate + 健康判定 + M4 vs amateur 比

Usage:
    # 模式 A:重用 listening dump 的 mel(無 GPU)
    python scripts/stage2_mel_eval.py \\
        --listening-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v2_listening \\
        --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized \\
        --pro-dataset m4singer \\
        --out-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v2_eval \\
        --pro-mean-n 100

    # 模式 B:fresh inference(等同 listening 但跳 vocoder)
    python scripts/stage2_mel_eval.py \\
        --stage1-ckpt /content/.../stage1_best.pt \\
        --stage2-ckpt-dir /content/.../stage2_v2 \\
        --binarized-root /content/.../binarized \\
        --val-split /content/.../splits/val.txt \\
        --pro-dataset m4singer \\
        --out-dir /content/.../stage2_v2_eval \\
        --n-samples 6 \\
        --steps "5000,15000,30000,50000,70000,90000,110000,120000" \\
        --pro-mean-n 100
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.utils.audio_config import NUM_MELS

# matplotlib 只在 render_mel_grid / render_zspace_tsne 等繪圖函式內 lazy import,
# 讓 aggregate_and_report 等純文字處理函式可被 rerender 工具復用(無需裝 matplotlib)


def _lazy_import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# 健康閾值(對齊 training_flow.md §3.6.1 / stage2.py monitor)
#
# 注意:`temporal_diff_ratio_mel` 跟 `hf_energy_increase` 的 baseline 已含 Stage 1
# VAE 重建本身的差異(典型 tdr_mel_baseline ~1.0, hf_baseline ~-0.01)。
# 直接套絕對閾值會把 VAE 噪音底歸咎給 M。
# 改看 `tdr_mel_extra` / `hf_energy_increase_extra`(扣掉 baseline 後)才能正確判 M。
HEALTHY = {
    "unvoiced_concentration": {"good": 0.55, "warn": 0.65},
    "voiced_spectral_ratio":  {"good": 0.70, "warn": 0.40},  # ≥0.7 健康, <0.4 警訊
    # _extra 才是 M 的真實貢獻(baseline 已 ~1.0,M 多加 < 0.3 才算健康)
    "temporal_diff_ratio_mel_extra": {"good": 0.30, "warn": 1.0},
    # 高頻能量變化:±0.20 健康內,-0.35~-0.20 marginal(高頻略削),
    # > +0.50 hiss/金屬感警訊,< -0.35 抹掉齒音/氣聲/亮度警訊
    "hf_energy_increase_extra": {"good_abs": 0.20, "warn_low": -0.35, "warn_high": 0.50},
    "pro_direction_alignment":{"good": 0.30, "warn": 0.0},   # >0.3 健康(正方向), <0 反方向
}


# ── 計算指標 ─────────────────────────────────────────────────────

def compute_metrics(mel_out: np.ndarray, mel_orig: np.ndarray,
                    mel_baseline: np.ndarray, f0: np.ndarray,
                    pro_mean_env: np.ndarray,
                    baseline_metrics: dict = None) -> dict:
    """計算單一 (sample, step) 的所有 mel-domain 指標。

    Args:
        mel_out:       [T, NUM_MELS] - M+decoder 出來的 mel
        mel_orig:      [T, NUM_MELS] - .npz['mel'] (binarize GT)
        mel_baseline:  [T, NUM_MELS] - stage1 only (M=None) 重建的 mel
        f0:            [T] - f0 (Hz);voiced = (f0 > 0)
        pro_mean_env:  [NUM_MELS] - 全 pro 集的 mean envelope (axis=0 reduction)
        baseline_metrics: 可選,baseline (_1_stage1_recon) 的 metrics dict。給了會額外
                          算 `_extra` 版指標 = current - baseline,扣掉 VAE 重建底噪後
                          M 的真實貢獻。對 `temporal_diff_ratio_mel` / `hf_energy_increase`
                          / `mel_l1_vs_orig` 有意義(它們參考 mel_orig,含 VAE noise)。

    Returns: dict of metric_name -> float
    """
    T = mel_out.shape[0]
    voiced = (f0 > 0).astype(np.float32)
    unvoiced = (1 - voiced)
    n_voiced = max(voiced.sum(), 1)
    n_unvoiced = max(unvoiced.sum(), 1)

    # ── 內容保留 ──────────────────────────────────────────
    # mel_l1_vs_orig: 總改變量(對 GT,含 VAE 重建噪音)
    mel_l1_vs_orig = float(np.mean(np.abs(mel_out - mel_orig)))
    # mel_l1_vs_recon: 把 VAE recon 噪音扣掉,只剩 M 的純貢獻
    mel_l1_vs_recon = float(np.mean(np.abs(mel_out - mel_baseline)))

    # ── 修飾方向 ──────────────────────────────────────────
    # 每首歌的 envelope (mean over time);用 baseline 作為「未修飾起點」
    env_out      = mel_out.mean(axis=0)          # [NUM_MELS]
    env_baseline = mel_baseline.mean(axis=0)
    env_orig     = mel_orig.mean(axis=0)

    # 各自到 pro_mean 的 L1 距離
    pro_dist_orig     = float(np.mean(np.abs(env_orig     - pro_mean_env)))
    pro_dist_baseline = float(np.mean(np.abs(env_baseline - pro_mean_env)))
    pro_dist_out      = float(np.mean(np.abs(env_out      - pro_mean_env)))
    # 修飾把 envelope 拉了多少 pro_dist:正號=更接近 pro,負號=遠離 pro
    pro_dist_delta    = pro_dist_baseline - pro_dist_out

    # direction alignment:cos(modification_vec, pro_direction_vec)
    #   modification = env_out - env_baseline    [M 對 baseline 改的 envelope 方向]
    #   direction    = pro_mean - env_baseline   [從 baseline 走到 pro 的理想方向]
    #   alignment ∈ [-1, 1];1=完美往 pro 走,-1=完全反向,0=正交
    mod_vec = env_out - env_baseline
    dir_vec = pro_mean_env - env_baseline
    mod_norm = np.linalg.norm(mod_vec) + 1e-10
    dir_norm = np.linalg.norm(dir_vec) + 1e-10
    pro_direction_alignment = float(np.dot(mod_vec, dir_vec) / (mod_norm * dir_norm))

    # ── 健康指標 (port 自 stage2.py audio_quality_monitor)──
    delta_mel = mel_out - mel_baseline                  # [T, NUM_MELS]
    delta_energy = (delta_mel ** 2).sum(axis=-1)        # [T]

    voiced_e   = float((delta_energy * voiced).sum() / n_voiced)
    unvoiced_e = float((delta_energy * unvoiced).sum() / n_unvoiced)
    unvoiced_concentration = unvoiced_e / (voiced_e + unvoiced_e + 1e-10)

    # voiced_spectral_ratio: voiced 段 Δ 的低時間頻率成分比例
    delta_hf = delta_mel[1:] - delta_mel[:-1]           # [T-1, NUM_MELS] HF 時間分量
    delta_lf = (delta_mel[1:] + delta_mel[:-1]) / 2     # [T-1, NUM_MELS] LF 時間分量
    valid_voiced_pair = voiced[1:] * voiced[:-1]        # [T-1]
    n_pair = max(valid_voiced_pair.sum(), 1)
    voiced_hf_e = float(((delta_hf ** 2).sum(-1) * valid_voiced_pair).sum() / n_pair)
    voiced_lf_e = float(((delta_lf ** 2).sum(-1) * valid_voiced_pair).sum() / n_pair)
    voiced_spectral_ratio = voiced_lf_e / (voiced_lf_e + voiced_hf_e + 1e-10)

    # ── Artifact 檢查 ─────────────────────────────────────
    # HF energy increase:bin >= 50 (~5kHz 以上) 的能量 vs orig
    hf_e_out  = float((mel_out[:, 50:] ** 2).sum())
    hf_e_orig = float((mel_orig[:, 50:] ** 2).sum() + 1e-10)
    hf_energy_increase = (hf_e_out - hf_e_orig) / hf_e_orig

    # temporal_diff_ratio_mel:‖Δ_t(mel_out) - Δ_t(mel_orig)‖₁ / ‖Δ_t(mel_orig)‖₁
    #   M 是否把 mel 的時間導數改太多 — 時間軌跡破壞 proxy(mel 域版,latent 之外)
    dt_out  = mel_out[1:]  - mel_out[:-1]
    dt_orig = mel_orig[1:] - mel_orig[:-1]
    tdr_mel = float(np.mean(np.abs(dt_out - dt_orig)) /
                    (np.mean(np.abs(dt_orig)) + 1e-10))

    result = {
        # 修飾幅度(M 改了多少;**不**等於 content preservation 證明,真的 content 保留
        # 要靠 PPG similarity / ASR / phoneme posterior 等,目前 backlog)
        "mel_l1_vs_orig":           mel_l1_vs_orig,
        "mel_l1_vs_recon":          mel_l1_vs_recon,
        # 修飾方向(僅看 time-averaged envelope,不證明 pitch / vibrato / 咬字 更好)
        "pro_dist_orig":            pro_dist_orig,
        "pro_dist_baseline":        pro_dist_baseline,
        "pro_dist_out":             pro_dist_out,
        "pro_dist_delta":           pro_dist_delta,
        "pro_direction_alignment":  pro_direction_alignment,
        # 健康
        "unvoiced_concentration":   unvoiced_concentration,
        "voiced_spectral_ratio":    voiced_spectral_ratio,
        # Artifact(原始版,baseline 含 VAE 重建底噪,看 _extra 比較精準)
        "hf_energy_increase":       hf_energy_increase,
        "temporal_diff_ratio_mel":  tdr_mel,
    }

    # ── _extra 版指標(扣 baseline 噪音底,M 的真實貢獻)──
    # 三個用 mel_orig 當參考的指標都受 VAE 重建噪音影響;_extra = current - baseline
    # 給定 baseline_metrics(self-vs-self)時:_extra 應該全 0
    # 給定 step_N(M+decoder)時:_extra 反映 M 在 baseline 之上又加/減了多少
    if baseline_metrics is not None:
        for k in ("temporal_diff_ratio_mel", "hf_energy_increase", "mel_l1_vs_orig"):
            base_v = baseline_metrics.get(k, 0.0)
            result[f"{k}_extra"] = result[k] - base_v
    else:
        # baseline 自己跟自己比,_extra 定義為 0(語意一致性)
        for k in ("temporal_diff_ratio_mel", "hf_energy_increase", "mel_l1_vs_orig"):
            result[f"{k}_extra"] = 0.0

    return result


def compute_pro_mean_env(binarized_root: Path, pro_dataset: str, n: int,
                         cache_path: Path, seed: int = 42,
                         split_file: Path = None) -> np.ndarray:
    """採樣 n 個 pro mel,算每首的 envelope(time-mean),再平均得到 pro mean env。

    為什麼 envelope 而不是 raw mel:不同 sample 長度不同,沒辦法直接平均 [T, NUM_MELS];
        envelope 是 [NUM_MELS] 固定 shape,且代表「該 sample 的 spectral 重心」,
        平均後得到 pro 群體的「典型 envelope」。

    Args:
        split_file: 可選,只從 split file 內 item_id 抽 pro_mean(避免 test 自己被
                    含進 reference)。預設 None = 全 pro_dataset。
    """
    if cache_path.exists():
        cached = np.load(cache_path)
        print(f"[pro_mean] cached envelope loaded from {cache_path} (shape={cached.shape})")
        return cached

    pro_dir = binarized_root / pro_dataset
    if split_file is not None:
        # 只取 split file 內列出的 pro samples(避免 test self-contamination)
        items = [s.strip() for s in Path(split_file).read_text().splitlines() if s.strip()]
        npz_files = [pro_dir / f"{item}.npz" for item in items]
        npz_files = [p for p in npz_files if p.exists()]
        print(f"[pro_mean] split-filtered ({split_file.name}): "
              f"{len(npz_files)} pro samples available")
    else:
        npz_files = sorted(pro_dir.glob("*.npz"))
        print(f"[pro_mean] using all {len(npz_files)} samples in {pro_dataset}/ "
              f"(⚠ 含 test 樣本,有輕微 self-contamination)")
    if not npz_files:
        raise SystemExit(f"no .npz available for pro_mean computation")

    rng = random.Random(seed)
    sampled = rng.sample(npz_files, min(n, len(npz_files)))

    print(f"[pro_mean] computing envelope from {len(sampled)} pro samples")
    envs = []
    for p in sampled:
        with np.load(p, allow_pickle=True) as d:
            mel = d["mel"].astype(np.float32)  # [T, NUM_MELS]
        envs.append(mel.mean(axis=0))          # [NUM_MELS]
    pro_mean_env = np.stack(envs).mean(axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pro_mean_env)
    print(f"[pro_mean] saved to {cache_path}, shape={pro_mean_env.shape}, "
          f"range=[{pro_mean_env.min():.2f}, {pro_mean_env.max():.2f}]")
    return pro_mean_env


# ── 視覺化 ───────────────────────────────────────────────────────

def render_mel_grid(mels: dict, f0: np.ndarray, metrics: dict,
                    out_path: Path, sample_title: str):
    """NSVB-style spec_f0_to_figure stack。

    Args:
        mels: ordered dict {label: mel [T, NUM_MELS]} - 含 orig / stage1_recon / step{N}
        f0:   [T] Hz - 從 .npz 讀,同首歌共用
        metrics: {label: {metric: float}} - 標題上顯示
        out_path: PNG 輸出
        sample_title: 整圖 super-title(e.g. "vocalverse / user01_song")
    """
    plt = _lazy_import_plt()
    n_rows = len(mels)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 1.6 * n_rows),
        sharex=True, constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # 計算共同 vmin/vmax(從所有 mel 取 1-99 percentile,fair across rows)
    all_mel = np.concatenate([m.flatten() for m in mels.values()])
    vmin, vmax = np.percentile(all_mel, [1, 99])

    # F0 線:跟 NSVB spec_f0_to_figure 一樣 clip 到 mel range
    f0_clip = np.clip(f0, 0, NUM_MELS).astype(np.float32)
    # F0 標尺:把 0-1500 Hz 大致映射到 0-NUM_MELS(80) — NSVB 用 /10 但對 singing F0 偏低
    # 改用線性 map:F0/20 落 0-75,大致跟 mel bin 對齊
    f0_for_overlay = np.clip(f0 / 20.0, 0, NUM_MELS - 1)

    for ax, (label, mel) in zip(axes, mels.items()):
        # mel: [T, NUM_MELS] -> 轉置 [NUM_MELS, T] 給 pcolor
        ax.pcolormesh(mel.T, vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
        ax.plot(f0_for_overlay, color="r", linewidth=0.8, alpha=0.7, label="F0/20")
        ax.set_ylabel(label, fontsize=8)
        ax.set_yticks([])
        if label in metrics and metrics[label]:
            m = metrics[label]
            tag = (
                f"L1_recon={m.get('mel_l1_vs_recon', 0):.3f}  "
                f"uv_conc={m.get('unvoiced_concentration', 0):.2f}  "
                f"vsr={m.get('voiced_spectral_ratio', 0):.2f}  "
                f"pro_dir={m.get('pro_direction_alignment', 0):+.2f}  "
                f"tdr_extra={m.get('temporal_diff_ratio_mel_extra', 0):+.2f}  "
                f"hf_extra={m.get('hf_energy_increase_extra', 0):+.2f}"
            )
            ax.set_title(tag, fontsize=7, loc="left", pad=2)

    axes[-1].set_xlabel("mel frame")
    fig.suptitle(sample_title, fontsize=10)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── 載入 mels(兩種模式)─────────────────────────────────────────

def load_mels_from_listening_dir(sample_dir: Path) -> tuple[dict, np.ndarray]:
    """從 listening output dir 讀 .mel.npy + f0.npy(無 GPU)。"""
    files = sorted(sample_dir.glob("*.mel.npy"))
    mels = {}
    for f in files:
        # name e.g. "_0_orig.mel.npy" / "_1_stage1_recon.mel.npy" / "step005000.mel.npy"
        label = f.name.replace(".mel.npy", "")
        mels[label] = np.load(f)
    f0_path = sample_dir / "f0.npy"
    if not f0_path.exists():
        raise FileNotFoundError(f"{f0_path} 不存在 — listening 跑時要加 --dump-mel")
    f0 = np.load(f0_path)
    return mels, f0


def run_fresh_inference(args) -> dict:
    """模式 B:fresh inference。回傳 {sample_dir_name: (mels_dict, f0)}。需 GPU + ckpts。"""
    import torch
    from scripts.stage2_ckpts_listening import (
        load_svbvae, load_M, load_vocoder, run_mode_a, find_npz,
    )
    binarized_root = Path(args.binarized_root)
    val_list = [s.strip() for s in Path(args.val_split).read_text().splitlines() if s.strip()]
    val_vv = [v for v in val_list if "__c" in v]
    val_m4 = [v for v in val_list if "#" in v and v not in val_vv]

    rng = random.Random(args.seed)
    n_vv = max(args.n_samples - 1, 1)
    picks = rng.sample(val_vv, min(n_vv, len(val_vv)))
    if val_m4 and len(picks) < args.n_samples:
        picks += rng.sample(val_m4, args.n_samples - len(picks))

    device = args.device
    svbvae = load_svbvae(args.stage1_ckpt, device)
    # vocoder 仍然要載(載入後棄置不用),因為 run_mode_a 內 vocoder forward 一定會跑。
    # 為什麼不重寫 inference path:用同一條跟 listening 完全等價,確保 mel 值與 listening 一致
    vocoder = load_vocoder(args.vocoder_ckpt, device)

    step_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    out: dict = {}
    for item_id in picks:
        npz_path, ds = find_npz(item_id, binarized_root)
        if npz_path is None:
            print(f"✗ {item_id}: .npz not found")
            continue
        safe = item_id.replace("#", "_").replace("__", "-")
        sample_key = f"{ds}_{safe}"
        print(f"[infer] {sample_key}")

        with np.load(npz_path, allow_pickle=True) as d:
            mel_orig = d["mel"].astype(np.float32)
            f0 = d["f0"].astype(np.float32)

        _, mel_s1 = run_mode_a(svbvae, None, vocoder, npz_path, device)
        mels = {"_0_orig": mel_orig, "_1_stage1_recon": mel_s1}

        for step in step_list:
            ckpt = Path(args.stage2_ckpt_dir) / f"stage2_step{step}.pt"
            if not ckpt.exists():
                continue
            M = load_M(str(ckpt), device)
            _, mel_step = run_mode_a(svbvae, M, vocoder, npz_path, device)
            mels[f"step{step:06d}"] = mel_step
            del M
        out[sample_key] = (mels, f0, item_id, ds)
    return out


# ── 報告 ─────────────────────────────────────────────────────────

def aggregate_and_report(per_sample: dict, out_dir: Path):
    """彙整 per-sample metrics 成 aggregate CSV + report.md。

    per_sample: {sample_key: {label: metrics_dict, ...}}
        label 為 "_0_orig" / "_1_stage1_recon" / "step{NNNNNN}"
    """
    # 找所有 step labels(orig/stage1_recon 不算 step)
    all_labels = set()
    for d in per_sample.values():
        all_labels.update(d.keys())
    step_labels = sorted([l for l in all_labels if l.startswith("step")])
    base_labels = ["_1_stage1_recon"] + step_labels  # orig 沒 metric

    metric_keys = [
        "mel_l1_vs_orig", "mel_l1_vs_orig_extra", "mel_l1_vs_recon",
        "pro_dist_baseline", "pro_dist_out", "pro_dist_delta",
        "pro_direction_alignment",
        "unvoiced_concentration", "voiced_spectral_ratio",
        "hf_energy_increase", "hf_energy_increase_extra",
        "temporal_diff_ratio_mel", "temporal_diff_ratio_mel_extra",
    ]

    # ── aggregate CSV(用 csv.writer 自動處理含逗號的 sample 名)──
    # 為什麼用 csv.writer:M4 部份歌名含逗號(例如「想你,零点零一分」),手動 join
    # 不做 quoting 會讓 row 多出一欄,後續 reload(rerender 工具)會 parse 出錯
    import csv as _csv
    with (out_dir / "metrics_aggregate.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["sample", "label"] + metric_keys)
        for s_key in sorted(per_sample.keys()):
            for label in base_labels:
                m = per_sample[s_key].get(label)
                if m is None:
                    continue
                row = [s_key, label] + [f"{m.get(k, 0):.4f}" for k in metric_keys]
                w.writerow(row)

    # ── M4 vs VV 分組 ──
    m4_keys = [k for k in per_sample if k.startswith("m4singer_")]
    vv_keys = [k for k in per_sample if k.startswith("vocalverse_")]

    def mean_over_samples(group_keys: list, label: str) -> dict:
        items = [per_sample[k][label] for k in group_keys if label in per_sample[k]]
        if not items:
            return {}
        return {mk: float(np.mean([it[mk] for it in items])) for mk in metric_keys}

    # ── verdict helper ──
    def verdict(label_pref: str, metric_value: float) -> str:
        spec = HEALTHY.get(label_pref)
        if not spec:
            return ""
        if label_pref == "voiced_spectral_ratio":
            good, warn = spec["good"], spec["warn"]
            if metric_value >= good:
                return "✅"
            return "❌" if metric_value < warn else "⚠️"
        if label_pref == "pro_direction_alignment":
            good, warn = spec["good"], spec["warn"]
            if metric_value > good:
                return "✅"
            return "❌" if metric_value < warn else "⚠️"
        if label_pref == "hf_energy_increase_extra":
            good_abs = spec["good_abs"]      # ±0.20 內 healthy
            warn_low = spec["warn_low"]      # < -0.35 嚴重削掉高頻
            warn_high = spec["warn_high"]    # > +0.50 hiss/金屬感
            if abs(metric_value) <= good_abs:
                return "✅"
            if metric_value < warn_low or metric_value > warn_high:
                return "❌"
            return "⚠️"
        # 其他(unvoiced_concentration, tdr_mel_extra):越小越好
        good, warn = spec["good"], spec["warn"]
        if metric_value <= good:
            return "✅"
        return "❌" if metric_value > warn else "⚠️"

    def render_table(headers, rows):
        widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
                  for i, h in enumerate(headers)]
        out = []
        out.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for r in rows:
            out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
        return out

    # ── report.md ──
    lines = ["# Stage 2 mel-domain evaluation\n"]
    lines.append(f"- samples: {len(per_sample)}  (m4={len(m4_keys)}, vv={len(vv_keys)})")
    lines.append(f"- steps:   {len(step_labels)}")
    lines.append("")
    lines.append("> **報告結構**:VV(amateur)是主要推理對象,M4(pro)只是 control "
                 "看 M 沒過度修飾;故主表用 VV-only 平均算指標跟 verdict。")
    lines.append("> baseline `_1_stage1_recon` 列出供 reference,**它的指標是 self-vs-self**"
                 "(0 或 undefined),不打 ✅/❌。")
    lines.append("")

    # ── §1. VV-only per-step aggregate(主表)──
    lines.append("## 1. VV(amateur)per-step aggregate — 主要推理品質\n")
    if not vv_keys:
        lines.append("(無 VV samples,跳過 VV 表)\n")
    else:
        headers = ["step", "L1_recon", "uv_conc", "vsr", "pro_dir", "pro_dist_Δ",
                   "tdr_extra", "hf_extra"]
        rows = []
        for label in base_labels:
            m = mean_over_samples(vv_keys, label)
            if not m:
                continue
            if label == "_1_stage1_recon":
                # baseline 列:self-vs-self,verdict 沒意義 → 不打符號
                row = [
                    label + " (ref)",
                    f"{m['mel_l1_vs_recon']:.3f}",
                    f"{m['unvoiced_concentration']:.2f}",
                    f"{m['voiced_spectral_ratio']:.2f}",
                    f"{m['pro_direction_alignment']:+.2f}",
                    f"{m['pro_dist_delta']:+.3f}",
                    f"{m['temporal_diff_ratio_mel_extra']:+.2f}",
                    f"{m['hf_energy_increase_extra']:+.2f}",
                ]
            else:
                row = [
                    label,
                    f"{m['mel_l1_vs_recon']:.3f}",
                    f"{m['unvoiced_concentration']:.2f} {verdict('unvoiced_concentration', m['unvoiced_concentration'])}",
                    f"{m['voiced_spectral_ratio']:.2f} {verdict('voiced_spectral_ratio', m['voiced_spectral_ratio'])}",
                    f"{m['pro_direction_alignment']:+.2f} {verdict('pro_direction_alignment', m['pro_direction_alignment'])}",
                    f"{m['pro_dist_delta']:+.3f}",
                    f"{m['temporal_diff_ratio_mel_extra']:+.2f} {verdict('temporal_diff_ratio_mel_extra', m['temporal_diff_ratio_mel_extra'])}",
                    f"{m['hf_energy_increase_extra']:+.2f} {verdict('hf_energy_increase_extra', m['hf_energy_increase_extra'])}",
                ]
            rows.append(row)
        lines.extend(render_table(headers, rows))
        lines.append("")

    # ── §2. M4 control:只看 L1_recon 是否保持小 + pro_dir 接近 0 ──
    lines.append("## 2. M4(pro control)per-step — 只驗 M 沒過度修飾 pro\n")
    if not m4_keys:
        lines.append("(無 M4 samples,跳過 M4 control 表)\n")
    else:
        lines.append("此表的健康判定**標準不同**:M 對 pro 應該接近 identity。")
        lines.append("- `L1_recon` 應極小(< 0.10);若 ≥ 0.20 表示 L_id_pro 失效")
        lines.append("- `pro_dir` 接近 0(M 沒拉 pro,因為 pro 已經是 pro);"
                     "若 > +0.30 表示 M 對 pro 也在「再 pro 化」,可能是 mode collapse")
        lines.append("")
        m4_headers = ["step", "L1_recon", "pro_dir", "tdr_extra", "hf_extra", "備註"]
        m4_rows = []
        for label in base_labels:
            m = mean_over_samples(m4_keys, label)
            if not m:
                continue
            l1 = m["mel_l1_vs_recon"]
            pro_dir = m["pro_direction_alignment"]
            note_l1 = "✅" if l1 < 0.10 else ("⚠️" if l1 < 0.20 else "❌")
            note_dir = "✅" if abs(pro_dir) < 0.30 else "⚠️"
            note = f"{note_l1}{note_dir}"
            display = label + (" (ref)" if label == "_1_stage1_recon" else "")
            if label == "_1_stage1_recon":
                note = "(ref)"
            m4_rows.append([
                display,
                f"{l1:.3f}",
                f"{pro_dir:+.2f}",
                f"{m['temporal_diff_ratio_mel_extra']:+.2f}",
                f"{m['hf_energy_increase_extra']:+.2f}",
                note,
            ])
        lines.extend(render_table(m4_headers, m4_rows))
        lines.append("")

    # ── §3. Best step 推薦(以 VV 為準)──
    if vv_keys and step_labels:
        # 排除 _1_stage1_recon 跟 step005000(post-warmup chaos)
        candidate_steps = [s for s in step_labels if s != "step005000"]
        scored = []
        for s in candidate_steps:
            m = mean_over_samples(vv_keys, s)
            if not m:
                continue
            scored.append((s, m["pro_direction_alignment"], m))
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            best_step, best_align, best_m = scored[0]
            lines.append("## 3. Best step 推薦(以 VV `pro_direction_alignment` 排序)\n")
            lines.append(f"⭐ **`{best_step}` 為 VV 上 alignment 最高**:")
            lines.append(f"- `pro_direction_alignment` = **{best_align:+.3f}**")
            lines.append(f"- `unvoiced_concentration` = {best_m['unvoiced_concentration']:.3f} "
                         f"{verdict('unvoiced_concentration', best_m['unvoiced_concentration'])}")
            lines.append(f"- `voiced_spectral_ratio`  = {best_m['voiced_spectral_ratio']:.3f} "
                         f"{verdict('voiced_spectral_ratio', best_m['voiced_spectral_ratio'])}")
            lines.append(f"- `tdr_extra` = {best_m['temporal_diff_ratio_mel_extra']:+.3f} "
                         f"{verdict('temporal_diff_ratio_mel_extra', best_m['temporal_diff_ratio_mel_extra'])}")
            lines.append(f"- `hf_extra`  = {best_m['hf_energy_increase_extra']:+.3f} "
                         f"{verdict('hf_energy_increase_extra', best_m['hf_energy_increase_extra'])}")
            # 也列前 3 名給比較
            if len(scored) >= 2:
                lines.append("")
                lines.append("Top 3 by VV alignment:")
                for s, a, _ in scored[:3]:
                    lines.append(f"  - {s}: {a:+.3f}")
            lines.append("")
            lines.append("> 排除了 `_1_stage1_recon`(baseline)跟 `step005000`(D_z warmup 結束 post-chaos)。")
            lines.append("> 「best」之外仍需聽測佐證:`pro_direction_alignment` 只看 envelope 平均方向,"
                         "不包括時間動態與咬字(詳 [eval_metrics_guide.md §2.B](../../docs/eval_metrics_guide.md))。")
            lines.append("")

    # ── §4. M4 vs VV ratio(amateur-specificity sanity check)──
    if m4_keys and vv_keys and step_labels:
        last_step = step_labels[-1]
        m_m4 = mean_over_samples(m4_keys, last_step)
        m_vv = mean_over_samples(vv_keys, last_step)
        if m_m4 and m_vv:
            lines.append(f"## 4. M4 vs VV L1_recon ratio @ {last_step} — amateur-specificity\n")
            lines.append("M 對 pro / amateur 修飾量比,確認 M 不是「對所有輸入都做固定修飾」:")
            lines.append(f"- M4 (pro control) L1_recon = **{m_m4['mel_l1_vs_recon']:.3f}**(M 對 pro 的 noise floor)")
            lines.append(f"- VV (amateur)     L1_recon = **{m_vv['mel_l1_vs_recon']:.3f}**(M 對 amateur 的整體修飾)")
            ratio = m_vv["mel_l1_vs_recon"] / max(m_m4["mel_l1_vs_recon"], 1e-6)
            lines.append(f"- **ratio (VV / M4) = {ratio:.2f}×**")
            if ratio > 2.0:
                lines.append("\n→ M 對 amateur 修飾 **顯著大於** pro noise floor:"
                             "**amateur-specific ✅**(L_id_pro 工作中)")
            elif ratio > 1.2:
                lines.append("\n→ ratio 微弱:M 對 amateur 稍重於 pro,但**邊界 ⚠️**")
            else:
                lines.append("\n→ ratio ≈ 1:M 對 amateur 跟 pro 修飾差不多 → "
                             "**mode collapse 雛形或 L_id_pro 失效 ❌**")
            lines.append("")

    # ── per-sample 索引 ──
    lines.append("## Per-sample detail\n")
    for s_key in sorted(per_sample.keys()):
        lines.append(f"- [{s_key}/mel_grid.png](./{s_key}/mel_grid.png) — "
                     f"see [{s_key}/metrics.json](./{s_key}/metrics.json)")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] aggregate written: {out_dir / 'metrics_aggregate.csv'}")
    print(f"[OK] report  written:   {out_dir / 'report.md'}")


# ── Main ────────────────────────────────────────────────────────

def main():
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Stage 2 mel-domain evaluation")
    # 兩種模式擇一
    ap.add_argument("--listening-dir", type=Path, default=None,
                    help="模式 A: 讀 stage2_ckpts_listening.py --dump-mel 產的 .mel.npy(無 GPU)")
    ap.add_argument("--stage1-ckpt", default=None,
                    help="模式 B: fresh inference 用,SVBVAEZh backbone ckpt")
    ap.add_argument("--stage2-ckpt-dir", default=None,
                    help="模式 B: stage2_step*.pt 目錄")
    ap.add_argument("--vocoder-ckpt", default=None,
                    help="模式 B: 跑 inference 需 vocoder ckpt(雖然 eval 不用 wav,run_mode_a 共用此 path)")
    ap.add_argument("--val-split", default=None,
                    help="模式 B: splits/val.txt")
    ap.add_argument("--n-samples", type=int, default=6, help="模式 B: 取幾個 val 樣本")
    ap.add_argument("--steps", default="5000,15000,30000,50000,70000,90000,110000,120000",
                    help="模式 B: comma-separated step numbers")
    ap.add_argument("--seed", type=int, default=42, help="模式 B: 樣本選擇 seed")
    ap.add_argument("--device", default="cuda",
                    help="模式 B: cuda / cpu (default cuda; eval 跑 inference 需 GPU)")
    # 共用
    ap.add_argument("--binarized-root", required=True,
                    help="包含 m4singer/ vocalverse/ 的 binarized 目錄")
    ap.add_argument("--pro-dataset", default="m4singer", choices=["m4singer", "vocalverse"],
                    help="pro 端用哪個 dataset 算 mean envelope 參考(預設 m4singer)")
    ap.add_argument("--pro-mean-n", type=int, default=100,
                    help="採樣多少 pro samples 算 pro mean envelope。預設 100(spot check 夠);"
                         "full test set eval 建議 500(SEM 從 ~10%% 降到 ~4.5%%)。"
                         "再大效益遞減。")
    ap.add_argument("--pro-split-file", default=None,
                    help="只從這個 split file 內的 sample 抽 pro mean,避免 test 自己被含進"
                         " reference(self-contamination)。預設:自動偵測 "
                         "`{binarized_root}/splits/train.txt` 若存在則用。")
    ap.add_argument("--max-viz", type=int, default=20,
                    help="最多 render 幾個 sample 的 mel_grid.png(預設 20;全 test set"
                         " 跑時不會 render 4000 張)。設 0 全部不畫;設 -1 不限制。"
                         "metrics.json 跟 aggregate 仍對所有 sample 計算。")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.listening_dir is None and (args.stage1_ckpt is None or args.stage2_ckpt_dir is None):
        ap.error("必須給 --listening-dir(模式 A)或 --stage1-ckpt + --stage2-ckpt-dir(模式 B)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. pro mean envelope ──
    # split_file 邏輯:用戶指定 > 自動偵測 train.txt > None(用全 dataset)
    pro_split = Path(args.pro_split_file) if args.pro_split_file else None
    if pro_split is None:
        auto_train = Path(args.binarized_root) / "splits" / "train.txt"
        if auto_train.exists():
            pro_split = auto_train
            print(f"[pro_mean] auto-using {pro_split} to avoid test contamination")
    cache_tag = f"n{args.pro_mean_n}_{'train' if pro_split else 'all'}"
    pro_mean_env = compute_pro_mean_env(
        Path(args.binarized_root), args.pro_dataset, args.pro_mean_n,
        cache_path=args.out_dir / f"pro_mean_env_{cache_tag}.npy",
        seed=args.seed, split_file=pro_split,
    )

    # ── 2. 取得每 sample 的 mels + f0 ──
    samples: dict = {}
    if args.listening_dir is not None:
        # 模式 A:直接讀 .mel.npy
        sample_dirs = sorted([p for p in args.listening_dir.iterdir() if p.is_dir()])
        for sd in sample_dirs:
            try:
                mels, f0 = load_mels_from_listening_dir(sd)
            except FileNotFoundError as e:
                print(f"✗ {sd.name}: {e}")
                continue
            samples[sd.name] = (mels, f0, sd.name, sd.name.split("_")[0])
        print(f"[load] {len(samples)} samples from {args.listening_dir}")
    else:
        # 模式 B:fresh inference
        samples = run_fresh_inference(args)

    if not samples:
        raise SystemExit("沒收到任何 sample,退出")

    # ── 3. compute metrics per (sample, label) + 畫圖 + 寫 json ──
    # 為了在大規模 eval(全 test set ~4K)時避免 render 數千張 PNG,挑 mix-of-M4-and-VV
    # 的前 max_viz 個畫圖,其餘只算 metrics(metrics.json 仍寫,給 spot debug 用)
    all_keys = sorted(samples.keys())
    if args.max_viz == 0:
        viz_keys = set()
    elif args.max_viz < 0:
        viz_keys = set(all_keys)
    else:
        m4_first = [k for k in all_keys if k.startswith("m4singer_")]
        vv_first = [k for k in all_keys if k.startswith("vocalverse_")]
        # 平均分配 viz quota:M4 半 / VV 半
        n_m4 = min(len(m4_first), args.max_viz // 2)
        n_vv = min(len(vv_first), args.max_viz - n_m4)
        viz_keys = set(m4_first[:n_m4] + vv_first[:n_vv])
    print(f"[viz] will render mel_grid.png for {len(viz_keys)}/{len(samples)} samples")

    per_sample: dict = {}
    for s_key, (mels, f0, _item_id, _ds) in samples.items():
        sample_out = args.out_dir / s_key
        sample_out.mkdir(parents=True, exist_ok=True)

        mel_orig = mels.get("_0_orig")
        mel_baseline = mels.get("_1_stage1_recon")
        if mel_orig is None or mel_baseline is None:
            print(f"✗ {s_key}: 缺 _0_orig 或 _1_stage1_recon")
            continue

        sample_metrics: dict = {}
        # 先算 baseline (_1_stage1_recon) 的 metrics,後續 step 都拿這個算 _extra
        baseline_metrics = compute_metrics(
            mel_baseline, mel_orig, mel_baseline, f0, pro_mean_env,
            baseline_metrics=None,  # baseline 自己 _extra 全 0
        )
        for label, mel_out in mels.items():
            if label == "_0_orig":
                # orig 沒比較對象(它就是 orig 本身)
                sample_metrics[label] = {}
                continue
            if label == "_1_stage1_recon":
                sample_metrics[label] = baseline_metrics
                continue
            sample_metrics[label] = compute_metrics(
                mel_out, mel_orig, mel_baseline, f0, pro_mean_env,
                baseline_metrics=baseline_metrics,
            )

        (sample_out / "metrics.json").write_text(
            json.dumps(sample_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_sample[s_key] = sample_metrics

        # 視覺化(只對 viz_keys 內的 sample 畫)
        if s_key in viz_keys:
            render_mel_grid(
                mels=mels, f0=f0, metrics=sample_metrics,
                out_path=sample_out / "mel_grid.png",
                sample_title=s_key,
            )
            print(f"[plot] {s_key}/mel_grid.png")

    # ── 4. aggregate report ──
    aggregate_and_report(per_sample, args.out_dir)


if __name__ == "__main__":
    main()