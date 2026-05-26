"""Stage 1 VAE 重建品質審計。

【為什麼存在】
v2 訓完發現 stage1 重建有 ~+0.12 SSIM gap(vocoder 端比 GT mel 餵入差),且 baseline
`tdr_mel = 1.18`(VAE 重建本身的時間 jitter)。Stage 1 當初 accept 只看 val_total 收
斂,沒做以下審計:

  1. 每 dataset(M4 vs VV)的 mel SSIM / L1 / KL 是否健康
  2. per-bin mel error 看哪個頻帶被 reconstruct 最差
  3. z 空間(m_q time-mean)t-SNE,看 z_p vs z_a 是否高度重疊
  4. val loss 拆 M4 / VV 是否對稱(若 VV >> M4 表示 Stage 1 對 amateur 表現偏差)

本 script 對 val + test 各抽 N 樣本(預設 100/dataset)跑上述 4 條,出 report.md +
2 張 PNG(z-space t-SNE / per-bin error)+ metrics.json。

【用法(本機 Win)】
    python scripts/stage1_audit.py ^
        --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt ^
        --binarized-root data\binarized_v2 ^
        --n-per-dataset 100 ^
        --out-dir outputs\stage1_audit

【健康判定】
| metric | ✅ 健康 | ⚠ 邊界 | ❌ 警訊 |
|---|---|---|---|
| mel SSIM | ≥ 0.90 | 0.85-0.90 | < 0.85 |
| mel L1 | < 0.20 | 0.20-0.30 | > 0.30 |
| M4 vs VV SSIM gap | < 0.05 | 0.05-0.10 | > 0.10 |
| z-space t-SNE 重疊 | 兩群顯著重疊 | 部分重疊 | 明顯分群(condition 解耦失敗)|

【若 ❌ 怎辦】
- mel SSIM < 0.85 → 走 Plan C(重訓 Stage 1,且建議調 KL beta annealing)
- M4 vs VV gap > 0.10 → 加 oversample VV 或 Plan C 重新平衡
- z-space 明顯分群 → Stage 1 decoder condition 解耦失敗,需要重訓 Stage 1
- 全 ✅ → 信任 Stage 1,進 Plan A-E
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.model.svb_vae_zh import SVBVAEZh
from nsvb.utils.audio_config import NUM_MELS, LATENT_DOWN_FACTOR


def pad_to_multiple(arr: np.ndarray, multiple: int, pad_value: float = 0.0) -> np.ndarray:
    T = arr.shape[0]
    pad = (-T) % multiple
    if pad == 0:
        return arr
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[0] = (0, pad)
    return np.pad(arr, pad_widths, constant_values=pad_value)


def load_svbvae(stage1_ckpt: str, device: str):
    """從 stage1 ckpt 載入 SVBVAEZh + 回 config dict 供推理 reference。"""
    state = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    # Stage1Config field 名跟 Stage2Config 略有差異,用 .get 安全處理 fallback
    model = SVBVAEZh(
        num_mels=cfg.get("num_mels", 80),
        ppg_dim=cfg.get("ppg_dim", 1280),
        spk_emb_dim=cfg.get("spk_emb_dim", 256),
        latent_size=cfg.get("latent_size", 128),
        hidden_size=cfg.get("hidden_size", 192),
        enc_n_layers=cfg.get("enc_n_layers", 8),
        dec_n_layers=cfg.get("dec_n_layers", 4),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


@torch.no_grad()
def encode_decode(cvae, npz_path: Path, device: str):
    """跑 encoder + decoder,回 (mel_orig, mel_recon, m_q_time_mean, kl_per_sample)。

    KL 計算:每 frame 0.5 * (m_q^2 + exp(2*logs_q) - 2*logs_q - 1) 平均。對齊 stage1
    FVAE 訓中算 KL 的方式(sequence-level mean)。
    """
    with np.load(npz_path, allow_pickle=True) as d:
        mel_np = d["mel"].astype(np.float32)
        ppg_np = d["ppg"].astype(np.float32)
        f0_np = d["f0"].astype(np.float32)
        spk_np = d["spk_emb"].astype(np.float32)
    T_orig = mel_np.shape[0]

    mel_pad = pad_to_multiple(mel_np, LATENT_DOWN_FACTOR, pad_value=-10.0)
    ppg_pad = pad_to_multiple(ppg_np, LATENT_DOWN_FACTOR, pad_value=0.0)
    f0_pad = pad_to_multiple(f0_np, LATENT_DOWN_FACTOR, pad_value=0.0)

    mel = torch.from_numpy(mel_pad).unsqueeze(0).to(device)
    ppg = torch.from_numpy(ppg_pad).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0_pad).unsqueeze(0).to(device)
    spk = torch.from_numpy(spk_np).unsqueeze(0).to(device)
    mel_mask = torch.zeros_like(f0_t)
    mel_mask[:, :T_orig] = 1.0

    g = cvae.condition(ppg, f0_t, spk)
    mel_chfirst = mel.transpose(1, 2)
    mask_exp = mel_mask.unsqueeze(1)
    g_sqz = cvae.fvae.g_pre_net(g)
    z_q, m_q, logs_q, _ = cvae.fvae.encoder(mel_chfirst, mask_exp, g_sqz)

    # Decoder 走 m_q(deterministic posterior mean)
    mel_recon_chfirst = cvae.fvae.decoder(m_q, mask_exp, g)
    mel_recon = mel_recon_chfirst[:, :, :T_orig].squeeze(0).transpose(0, 1).cpu().numpy()

    # KL: per-frame 0.5 * (m_q^2 + exp(2 logs_q) - 2 logs_q - 1)
    kl_per_frame = 0.5 * (m_q.pow(2) + (2.0 * logs_q).exp() - 2.0 * logs_q - 1.0)
    # valid frames only (mask_z)
    T_z = m_q.shape[-1]
    mask_z = mel_mask[:, :T_z * LATENT_DOWN_FACTOR][:, ::LATENT_DOWN_FACTOR]   # 約略下採
    kl_sample = float(
        (kl_per_frame.sum(dim=1) * mask_z).sum() / mask_z.sum().clamp(min=1)
    )

    # z time-mean: [latent_size]
    mq_time_mean = m_q.squeeze(0).mean(dim=-1).cpu().numpy()
    return mel_np, mel_recon, mq_time_mean, kl_sample


def compute_metrics(mel_orig: np.ndarray, mel_recon: np.ndarray) -> dict:
    """SSIM + L1 + per-bin MSE."""
    from skimage.metrics import structural_similarity as ssim

    T = min(mel_orig.shape[0], mel_recon.shape[0])
    a = mel_orig[:T]
    b = mel_recon[:T]
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range < 1e-6:
        ssim_val = float("nan")
    else:
        ssim_val = float(ssim(a, b, data_range=data_range))
    l1 = float(np.mean(np.abs(a - b)))
    per_bin_mse = np.mean((a - b) ** 2, axis=0).astype(np.float32)  # [NUM_MELS]
    return {"ssim": ssim_val, "l1": l1, "per_bin_mse": per_bin_mse}


def pick_samples(split_file: Path, binarized_root: Path, n_per_dataset: int,
                 seed: int = 42) -> dict:
    """從 split file 抽 N samples each per dataset。回 {'m4singer': [Path], 'vocalverse': [Path]}"""
    items = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    # 跟 stage2_ckpts_listening 同 filter:M4 含 '#',VV 不含 '#'(VV 都是純數字)
    m4_items = [it for it in items if "#" in it]
    vv_items = [it for it in items if "#" not in it]
    rng = random.Random(seed)
    m4_picks = rng.sample(m4_items, min(n_per_dataset, len(m4_items)))
    vv_picks = rng.sample(vv_items, min(n_per_dataset, len(vv_items)))

    def to_paths(item_ids, ds_name):
        out = []
        for it in item_ids:
            p = binarized_root / ds_name / f"{it}.npz"
            if p.exists():
                out.append(p)
        return out

    return {
        "m4singer": to_paths(m4_picks, "m4singer"),
        "vocalverse": to_paths(vv_picks, "vocalverse"),
    }


def verdict_ssim(v: float) -> str:
    if v >= 0.90:
        return "✅"
    if v >= 0.85:
        return "⚠️"
    return "❌"


def verdict_l1(v: float) -> str:
    if v < 0.20:
        return "✅"
    if v < 0.30:
        return "⚠️"
    return "❌"


def verdict_gap(gap: float) -> str:
    if gap < 0.05:
        return "✅"
    if gap < 0.10:
        return "⚠️"
    return "❌"


def render_zspace_tsne(
    z_vectors: dict, out_path: Path, n_iter: int = 1000,
) -> Optional[str]:
    """z_vectors = {dataset_name: np.ndarray [N, latent_size]}; 畫 2D t-SNE 散點。"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return "sklearn not installed; skip t-SNE plot"

    all_z = np.concatenate(list(z_vectors.values()), axis=0)
    labels = np.concatenate([
        np.full(v.shape[0], k) for k, v in z_vectors.items()
    ])
    if all_z.shape[0] < 5:
        return f"too few samples ({all_z.shape[0]}); skip t-SNE"

    print(f"[audit] t-SNE on {all_z.shape[0]} z vectors (dim={all_z.shape[1]})...", flush=True)
    tsne = TSNE(n_components=2, perplexity=min(30, all_z.shape[0] - 1),
                random_state=42, n_iter=n_iter)
    z2d = tsne.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"m4singer": "tab:blue", "vocalverse": "tab:orange"}
    for ds in z_vectors.keys():
        mask = labels == ds
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=colors.get(ds, "gray"),
                   label=f"{ds} (n={mask.sum()})", alpha=0.6, s=20)
    ax.set_title("z-space (m_q time-mean) t-SNE — z_p vs z_a 應高度重疊")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return None


def render_per_bin_error(per_bin: dict, out_path: Path):
    """per_bin = {dataset: np.ndarray [NUM_MELS]}; bar chart 每 mel bin 的 MSE。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(NUM_MELS)
    width = 0.4
    colors = {"m4singer": "tab:blue", "vocalverse": "tab:orange"}
    for i, (ds, mse) in enumerate(per_bin.items()):
        ax.bar(x + (i - 0.5) * width, mse, width=width, label=ds,
               color=colors.get(ds, "gray"))
    ax.set_xlabel("mel bin (low ← → high freq)")
    ax.set_ylabel("MSE per bin")
    ax.set_title("Per-mel-bin reconstruction MSE — 高 mel bin 是否被特別差")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Stage 1 reconstruction audit")
    ap.add_argument("--stage1-ckpt", required=True, type=Path,
                    help="Stage 1 ckpt path,例如 checkpoints_v2\\stage1\\stage1_best.pt")
    ap.add_argument("--binarized-root", required=True, type=Path,
                    help="含 m4singer/ vocalverse/ splits/ 的 binarized root")
    ap.add_argument("--n-per-dataset", type=int, default=100,
                    help="每 dataset 從 val + test 各抽多少 samples")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.binarized_root / "splits"
    val_split = splits_dir / "val.txt"
    test_split = splits_dir / "test.txt"
    if not val_split.exists() or not test_split.exists():
        raise SystemExit(f"splits 不存在: {val_split} / {test_split}")

    print(f"[audit] device={args.device}")
    print(f"[audit] loading SVBVAEZh from {args.stage1_ckpt}", flush=True)
    cvae, cfg = load_svbvae(str(args.stage1_ckpt), args.device)
    print(f"[audit] CVAE loaded; latent_size={cfg.get('latent_size', '?')}, "
          f"hidden_size={cfg.get('hidden_size', '?')}", flush=True)

    # 抽 samples:val + test 各抽 n_per_dataset,total 4 個 bucket
    sample_paths = {}
    for split_name, split_file in (("val", val_split), ("test", test_split)):
        picks = pick_samples(split_file, args.binarized_root, args.n_per_dataset, args.seed)
        for ds, paths in picks.items():
            key = f"{split_name}_{ds}"
            sample_paths[key] = paths
            print(f"[audit] {key}: {len(paths)} samples picked", flush=True)

    # 跑 inference + metrics
    results = {}
    z_vectors = {"m4singer": [], "vocalverse": []}
    for key, paths in sample_paths.items():
        ds_name = key.split("_", 1)[1]    # "m4singer" or "vocalverse"
        bucket = {"ssim": [], "l1": [], "kl": [], "per_bin_mse": []}
        for i, p in enumerate(paths):
            mel_orig, mel_recon, mq_mean, kl = encode_decode(cvae, p, args.device)
            m = compute_metrics(mel_orig, mel_recon)
            bucket["ssim"].append(m["ssim"])
            bucket["l1"].append(m["l1"])
            bucket["per_bin_mse"].append(m["per_bin_mse"])
            bucket["kl"].append(kl)
            z_vectors[ds_name].append(mq_mean)
            if (i + 1) % 20 == 0:
                print(f"  {key}: {i+1}/{len(paths)}", flush=True)
        results[key] = {
            "n": len(paths),
            "ssim_mean": float(np.nanmean(bucket["ssim"])),
            "ssim_std": float(np.nanstd(bucket["ssim"])),
            "l1_mean": float(np.mean(bucket["l1"])),
            "l1_std": float(np.std(bucket["l1"])),
            "kl_mean": float(np.mean(bucket["kl"])),
            "per_bin_mse_mean": np.mean(bucket["per_bin_mse"], axis=0).tolist(),
        }
        print(f"[audit] {key}: SSIM={results[key]['ssim_mean']:.4f}±{results[key]['ssim_std']:.4f}, "
              f"L1={results[key]['l1_mean']:.4f}, KL={results[key]['kl_mean']:.3f}",
              flush=True)

    # 視覺化
    print(f"[audit] rendering z-space t-SNE...", flush=True)
    z_arrays = {ds: np.stack(vs) if vs else np.zeros((0, cfg.get("latent_size", 128)))
                for ds, vs in z_vectors.items()}
    tsne_warn = render_zspace_tsne(z_arrays, args.out_dir / "zspace_tsne.png")
    if tsne_warn:
        print(f"[audit] t-SNE warning: {tsne_warn}", flush=True)

    print(f"[audit] rendering per-bin error bar...", flush=True)
    # 取 val_m4 跟 val_vv 的 per_bin_mse 作圖(test 邏輯同)
    per_bin_for_plot = {
        "m4singer": np.array(results["val_m4singer"]["per_bin_mse_mean"]),
        "vocalverse": np.array(results["val_vocalverse"]["per_bin_mse_mean"]),
    }
    render_per_bin_error(per_bin_for_plot, args.out_dir / "per_bin_error.png")

    # 寫 metrics.json
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 寫 report.md
    write_report(results, args.stage1_ckpt, args.n_per_dataset, args.out_dir,
                 tsne_warn=tsne_warn)
    print(f"\n[OK] audit complete. See {args.out_dir / 'report.md'}", flush=True)


def write_report(results: dict, stage1_ckpt: Path, n_per_dataset: int, out_dir: Path,
                 tsne_warn: Optional[str] = None):
    lines = ["# Stage 1 audit report", ""]
    lines.append(f"- Stage 1 ckpt: `{stage1_ckpt}`")
    lines.append(f"- N per dataset per split: {n_per_dataset}")
    lines.append(f"- Output dir: `{out_dir}`")
    lines.append("")

    # 表格:val_m4 / val_vv / test_m4 / test_vv 四列
    lines.append("## 1. 重建品質彙整")
    lines.append("")
    headers = ["bucket", "n", "SSIM (mean±std)", "L1", "KL"]
    rows = []
    for key in ("val_m4singer", "val_vocalverse", "test_m4singer", "test_vocalverse"):
        if key not in results:
            continue
        r = results[key]
        rows.append([
            key,
            str(r["n"]),
            f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f} {verdict_ssim(r['ssim_mean'])}",
            f"{r['l1_mean']:.4f} {verdict_l1(r['l1_mean'])}",
            f"{r['kl_mean']:.3f}",
        ])
    widths = [max(len(str(c)) for c in [h] + [row[i] for row in rows])
              for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    lines.append("")

    # M4 vs VV gap
    if "val_m4singer" in results and "val_vocalverse" in results:
        gap_val = results["val_m4singer"]["ssim_mean"] - results["val_vocalverse"]["ssim_mean"]
        gap_test = (results["test_m4singer"]["ssim_mean"]
                    - results["test_vocalverse"]["ssim_mean"]) \
                    if "test_m4singer" in results else None
        lines.append("## 2. M4 vs VV SSIM gap")
        lines.append("")
        lines.append(f"- val: M4 - VV = **{gap_val:+.4f}** {verdict_gap(abs(gap_val))}")
        if gap_test is not None:
            lines.append(f"- test: M4 - VV = **{gap_test:+.4f}** {verdict_gap(abs(gap_test))}")
        lines.append("")
        lines.append("**讀法**:正值表 M4 重建較好 = Stage 1 對 amateur 表現偏差;"
                     "差 > 0.10 是 ❌(需要 oversample VV / Plan C 重訓 Stage 1)。")
        lines.append("")

    # 視覺化檔案
    lines.append("## 3. 視覺化")
    lines.append("")
    lines.append(f"- [z-space t-SNE](./zspace_tsne.png)({'⚠ skipped' if tsne_warn else '已產出'})")
    lines.append(f"- [per-bin reconstruction MSE](./per_bin_error.png)")
    lines.append("")
    lines.append("**讀 t-SNE 圖**:z_p (M4,藍) 跟 z_a (VV,橘) 應該大幅重疊 — 表示 condition")
    lines.append("(ppg / f0 / spk_emb)解耦成功,z 不含 dataset 簽名。明顯分群代表解耦失敗,需要 Stage 1 重訓 + 調 KL beta。")
    lines.append("")
    lines.append("**讀 per-bin 圖**:bin 0 是最低頻、bin 79 最高頻。若高 bin (50+) MSE 顯著大於低 bin → ")
    lines.append("Stage 1 高頻細節 reconstruct 差(可能是電音來源之一),調 KL annealing 給更多 high-freq capacity。")
    lines.append("")

    # 結論 / 建議
    lines.append("## 4. 結論 / 建議行動")
    lines.append("")
    val_m4_ssim = results.get("val_m4singer", {}).get("ssim_mean", 0)
    val_vv_ssim = results.get("val_vocalverse", {}).get("ssim_mean", 0)
    gap = abs(val_m4_ssim - val_vv_ssim)
    if val_vv_ssim < 0.85 or val_m4_ssim < 0.85:
        lines.append("- ❌ **重建 SSIM 過低**:Stage 1 本身就是瓶頸。**強烈建議走 Plan C**(重訓 Stage 1,KL annealing 調整)")
    elif val_vv_ssim < 0.90 or val_m4_ssim < 0.90:
        lines.append("- ⚠ **重建 SSIM 邊界**:Stage 1 不完美但可用。先跑 Plan A-E 看是否能搾出表現,失敗再考慮 Plan C。")
    else:
        lines.append("- ✅ **重建 SSIM 健康**:Stage 1 沒問題,Plan A-E 失敗時不用先怪 Stage 1。")
    lines.append("")
    if gap > 0.10:
        lines.append(f"- ❌ **M4/VV gap = {gap:.4f}**:Stage 1 對 amateur 顯著差,建議重訓 + oversample VV(或減小 M4 多樣性)。")
    elif gap > 0.05:
        lines.append(f"- ⚠ **M4/VV gap = {gap:.4f}**:Stage 1 略偏 pro 端,可接受。")
    else:
        lines.append(f"- ✅ **M4/VV gap = {gap:.4f}**:Stage 1 對兩 dataset 表現對稱,健康。")
    lines.append("")
    lines.append("(t-SNE 是否健康請肉眼判斷 zspace_tsne.png。)")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()