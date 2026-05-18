"""Diagnose: is the amateur-side electronic artifact from Stage 1 mel reconstruction
or from the vocoder itself?

【問題背景】
聽測時 amateur 端 (_1_stage1_recon.wav) 有電音, M4 (pro) 端沒有。電音可能來自:
  (a) Stage 1 mel reconstruction 對 amateur 加了時間域 noise → vocoder 把這個 noise
      放大成電音
  (b) vocoder 本身對 amateur mel 分布不熟 (NSVB pretrained 主要 pro singing voice)
      → 即使餵 GT amateur mel 也會出電音
  (c) F0 path 問題 (SineGen 對 amateur F0 邊界敏感)

本 script 隔離 (a) vs (b):
  1. 對每 sample, 跑 vocoder(GT amateur mel, GT F0_interp) → _T_vocoder_on_gt.wav
  2. 既有 _1_stage1_recon.wav 就是 vocoder(stage1_recon mel, GT F0_interp)
  3. 同 F0 路徑、只差 mel:GT vs stage1 重建。聽差就知道 stage 1 mel 加了多少 noise。
     量化:對兩條 wav 各跑 mel SSIM + F0 RMSE vs GT。

對 (c) 的測試留給後續另一支 script。

【執行】
本 script 假設 stage2_ckpts_listening.py 跑時加了 --dump-mel,所以 sample_dir 內已有:
  _0_orig.wav            原音
  _0_orig.mel.npy        GT mel (從 .npz)
  _1_stage1_recon.wav    vocoder(stage1_recon_mel, F0_interp)
  _1_stage1_recon.mel.npy stage1 重建後的 mel
  f0.npy                  F0 array

跑完會多出:
  _T_vocoder_on_gt.wav    vocoder(GT mel, F0_interp) ← NEW

外加:
  diagnostic.json (per-sample)
  vocoder_path_report.md  (aggregate + verdict)

Usage:
    python scripts/diagnose_stage1_vocoder_path.py \\
        --listening-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v2_listening \\
        --vocoder-ckpt /content/drive/MyDrive/nsvb_ckpts/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.utils.audio_config import SAMPLE_RATE
from nsvb.utils.audio_io import load_wav, compute_mel
from scripts.stage2_ckpts_listening import load_vocoder, interp_f0_unvoiced
from scripts.vocoder_identity_test import mel_ssim, f0_rmse


@torch.no_grad()
def run_vocoder(vocoder, mel_np: np.ndarray, f0_np: np.ndarray, device: str) -> np.ndarray:
    """vocoder(mel [T, NUM_MELS], f0 [T]) → wav np.ndarray。F0 走 unvoiced log 內插。"""
    mel_t = torch.from_numpy(mel_np.astype(np.float32)).unsqueeze(0).transpose(1, 2).to(device)
    f0_interp = interp_f0_unvoiced(f0_np.astype(np.float32))
    f0_t = torch.from_numpy(f0_interp).unsqueeze(0).to(device)
    wav = vocoder(mel_t, f0_t).squeeze(1).squeeze(0).cpu().numpy()
    return wav


def diagnose_sample(sample_dir: Path, vocoder, device: str) -> dict:
    """單一 sample 的隔離測試。回傳 metric dict。"""
    gt_mel = np.load(sample_dir / "_0_orig.mel.npy")
    f0     = np.load(sample_dir / "f0.npy")

    # vocoder(GT mel) — NEW
    wav_on_gt = run_vocoder(vocoder, gt_mel, f0, device)
    sf.write(str(sample_dir / "_T_vocoder_on_gt.wav"), wav_on_gt, SAMPLE_RATE)
    print(f"  + _T_vocoder_on_gt.wav saved")

    # 既有檔案
    wav_orig  = load_wav(str(sample_dir / "_0_orig.wav"),           sr=SAMPLE_RATE)
    wav_recon = load_wav(str(sample_dir / "_1_stage1_recon.wav"),   sr=SAMPLE_RATE)

    # mel SSIM:對兩條 vocoder 輸出各重抽 mel,跟 GT mel 比
    # SSIM 高 = 重建跟 GT 像;低 = 結構性偏差(電音 / formant 跑掉等)
    mel_re_on_gt    = compute_mel(wav_on_gt)
    mel_re_on_recon = compute_mel(wav_recon)
    ssim_on_gt    = mel_ssim(gt_mel, mel_re_on_gt)
    ssim_on_recon = mel_ssim(gt_mel, mel_re_on_recon)

    # F0 RMSE:對兩條 vocoder 輸出抽 F0,跟原 wav 的 F0 比
    rmse_on_gt,    iou_on_gt    = f0_rmse(wav_orig, wav_on_gt,    device)
    rmse_on_recon, iou_on_recon = f0_rmse(wav_orig, wav_recon,    device)

    return {
        # mel SSIM (越高越像 GT,目標 ≥ 0.90)
        "ssim_vocoder_on_gt_mel":     ssim_on_gt,
        "ssim_vocoder_on_recon_mel":  ssim_on_recon,
        "ssim_delta":                 ssim_on_gt - ssim_on_recon,  # 正號=stage1 recon 有掉
        # F0 RMSE Hz (越低越好,目標 ≤ 10 Hz)
        "f0_rmse_vocoder_on_gt":      rmse_on_gt,
        "f0_rmse_vocoder_on_recon":   rmse_on_recon,
        # voicing IOU (越接近 1 = voiced 段判定一致)
        "voicing_iou_vocoder_on_gt":     iou_on_gt,
        "voicing_iou_vocoder_on_recon":  iou_on_recon,
    }


# ── verdict ──────────────────────────────────────────────────

def verdict_for_sample(m: dict) -> str:
    """單一 sample 的判定:電音來自 (a) stage1 mel 還是 (b) vocoder?"""
    ssim_gt    = m["ssim_vocoder_on_gt_mel"]
    ssim_recon = m["ssim_vocoder_on_recon_mel"]
    delta = m["ssim_delta"]

    # Case 1: 兩條都低 → vocoder bottleneck
    if ssim_gt < 0.85 and ssim_recon < 0.85:
        return f"❌ vocoder bottleneck (both SSIM < 0.85, gap={delta:+.2f})"
    # Case 2: vocoder_on_gt 高 + vocoder_on_recon 低 → stage1 mel 是噪音來源
    if ssim_gt >= 0.85 and ssim_recon < 0.85:
        return f"❌ stage1 recon mel is the cause (vocoder OK on GT, fails on recon, gap={delta:+.2f})"
    # Case 3: 兩條都高 + 仍有可聽到電音 → F0 path 或主觀感受差異
    if ssim_gt >= 0.85 and ssim_recon >= 0.85:
        return f"✅ both clean (SSIM ≥ 0.85, gap={delta:+.2f}) — 電音來自 F0 path 或非 SSIM 可捕捉"
    # Case 4: 反常 — recon 比 GT 還高(理論不該發生)
    return f"⚠️ unexpected (gt={ssim_gt:.2f}, recon={ssim_recon:.2f}, gap={delta:+.2f})"


def write_report(results: dict, out_dir: Path):
    """寫 aggregate report.md。"""
    if not results:
        out_dir.joinpath("vocoder_path_report.md").write_text(
            "# 無有效樣本\n", encoding="utf-8",
        )
        return

    m4_keys = [k for k in results if k.startswith("m4singer_")]
    vv_keys = [k for k in results if k.startswith("vocalverse_")]

    lines = ["# Stage 1 mel vs vocoder path diagnosis\n"]
    lines.append("**問題**：amateur 端 `_1_stage1_recon.wav` 有電音,M4 端沒有。是 Stage 1 mel reconstruction 加了噪音,還是 vocoder 對 amateur 分布不熟?")
    lines.append("")
    lines.append("**測試方法**:同 F0 路徑(`interp_f0_unvoiced`),只比 mel 來源 — GT vs stage1_recon。")
    lines.append("")

    # ── Per-sample 表 ──
    lines.append("## Per-sample\n")
    header = ["sample", "SSIM(voc on GT)", "SSIM(voc on recon)", "Δ", "F0_RMSE(GT)", "F0_RMSE(recon)", "verdict"]
    rows = []
    for k in sorted(results.keys()):
        m = results[k]
        rows.append([
            k,
            f"{m['ssim_vocoder_on_gt_mel']:.3f}",
            f"{m['ssim_vocoder_on_recon_mel']:.3f}",
            f"{m['ssim_delta']:+.3f}",
            f"{m['f0_rmse_vocoder_on_gt']:.1f}",
            f"{m['f0_rmse_vocoder_on_recon']:.1f}",
            verdict_for_sample(m),
        ])
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows]) for i, h in enumerate(header)]
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    lines.append("")

    # ── M4 vs VV 群組均值 ──
    def mean_group(keys: list) -> dict:
        if not keys:
            return {}
        agg = {}
        for mk in ["ssim_vocoder_on_gt_mel", "ssim_vocoder_on_recon_mel",
                   "ssim_delta", "f0_rmse_vocoder_on_gt", "f0_rmse_vocoder_on_recon"]:
            agg[mk] = float(np.mean([results[k][mk] for k in keys]))
        return agg

    m_m4 = mean_group(m4_keys)
    m_vv = mean_group(vv_keys)

    lines.append("## M4 (pro control) vs VV (amateur) 群組平均\n")
    g_rows = []
    if m_m4:
        g_rows.append(["M4 (pro)",
                       f"{m_m4['ssim_vocoder_on_gt_mel']:.3f}",
                       f"{m_m4['ssim_vocoder_on_recon_mel']:.3f}",
                       f"{m_m4['ssim_delta']:+.3f}",
                       f"{m_m4['f0_rmse_vocoder_on_gt']:.1f}",
                       f"{m_m4['f0_rmse_vocoder_on_recon']:.1f}"])
    if m_vv:
        g_rows.append(["VV (amateur)",
                       f"{m_vv['ssim_vocoder_on_gt_mel']:.3f}",
                       f"{m_vv['ssim_vocoder_on_recon_mel']:.3f}",
                       f"{m_vv['ssim_delta']:+.3f}",
                       f"{m_vv['f0_rmse_vocoder_on_gt']:.1f}",
                       f"{m_vv['f0_rmse_vocoder_on_recon']:.1f}"])
    g_header = ["dataset", "SSIM(voc on GT)", "SSIM(voc on recon)", "Δ", "F0_RMSE(GT)", "F0_RMSE(recon)"]
    g_widths = [max(len(str(c)) for c in [h] + [r[i] for r in g_rows]) for i, h in enumerate(g_header)]
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(g_header, g_widths)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in g_rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, g_widths)) + " |")
    lines.append("")

    # ── 整體 verdict ──
    lines.append("## 整體判定\n")
    if m_vv and m_m4:
        # VV 兩個 SSIM 差距是否顯著
        if m_vv["ssim_delta"] > 0.05:
            lines.append(f"→ VV 端 stage1_recon 比 vocoder(GT) SSIM 低 **{m_vv['ssim_delta']:+.3f}**:")
            lines.append("  **Stage 1 mel reconstruction 顯著退化 amateur mel 品質。**")
            lines.append("  建議:Stage 1 fine-tune 或調整 KL annealing;或 vocoder fine-tune on dereverb'd amateur mel。")
        elif m_vv["ssim_vocoder_on_gt_mel"] < 0.85:
            lines.append(f"→ VV 端 vocoder(GT amateur mel) 已 < 0.85 (SSIM={m_vv['ssim_vocoder_on_gt_mel']:.3f}):")
            lines.append("  **Vocoder 本身對 amateur 分布不熟。** Stage 1 mel 沒明顯加噪。")
            lines.append("  建議:vocoder fine-tune on amateur mel(NSVB pretrained 主要 pro singing voice 訓練)。")
        else:
            lines.append(f"→ VV 端兩條 SSIM 都 ≥ 0.85 且 gap < 0.05:")
            lines.append("  **mel/vocoder path 沒抓到電音來源。** 嫌疑回到 F0 path(SineGen / F0 抽取 / interp 邏輯)。")

        # 跟 M4 對比
        m4_gt_ssim = m_m4["ssim_vocoder_on_gt_mel"]
        vv_gt_ssim = m_vv["ssim_vocoder_on_gt_mel"]
        if m4_gt_ssim - vv_gt_ssim > 0.05:
            lines.append(f"\n→ M4 (pro) 端 vocoder(GT mel) SSIM = {m4_gt_ssim:.3f}, VV (amateur) 端 = {vv_gt_ssim:.3f}:")
            lines.append(f"  差距 {m4_gt_ssim - vv_gt_ssim:+.3f} — **vocoder 對 amateur 分布有偏向**。")
        else:
            lines.append(f"\n→ M4 跟 VV 在 vocoder(GT mel) SSIM 接近({m4_gt_ssim:.3f} vs {vv_gt_ssim:.3f}):")
            lines.append("  vocoder 對兩邊都 OK,問題在 stage1 recon 或 F0 path。")

    lines.append("")
    lines.append("## 建議聽法\n")
    lines.append("把每個 sample 資料夾的三個 wav 連著聽,順序:")
    lines.append("1. `_0_orig.wav`         — GT 錄音(完全 clean)")
    lines.append("2. `_T_vocoder_on_gt.wav` — vocoder 對 GT mel 的重建(隔離 vocoder 行為)")
    lines.append("3. `_1_stage1_recon.wav` — vocoder 對 stage1_recon mel 的重建(目前 pipeline)")
    lines.append("")
    lines.append("如果 2 → 3 才出現電音 → Stage 1 mel 是噪音來源(這份 report 應該指出 Δ > 0.05)")
    lines.append("如果 1 → 2 就出現電音 → vocoder 是噪音來源(SSIM(voc on GT) 該 < 0.85)")
    lines.append("如果 2 跟 3 都 clean(但你 _1_stage1_recon 聽得到電音)→ 主觀感受 vs SSIM 不同調,F0 path 仍是嫌疑")

    out_dir.joinpath("vocoder_path_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] report written: {out_dir / 'vocoder_path_report.md'}")


def main():
    # Windows cp950 emoji 安全
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Diagnose stage1 mel vs vocoder as cause of artifact")
    ap.add_argument("--listening-dir", required=True, type=Path,
                    help="Listening output dir (must have --dump-mel done)")
    ap.add_argument("--vocoder-ckpt", required=True,
                    help="Path to NSVB 1012 HifiGAN ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[load] vocoder from {args.vocoder_ckpt}")
    vocoder = load_vocoder(args.vocoder_ckpt, args.device)

    sample_dirs = sorted([p for p in args.listening_dir.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise SystemExit(f"no sample dirs in {args.listening_dir}")

    results: dict = {}
    for sd in sample_dirs:
        required = [sd / "_0_orig.mel.npy", sd / "f0.npy",
                    sd / "_0_orig.wav", sd / "_1_stage1_recon.wav"]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            print(f"✗ {sd.name}: missing {missing} — listening 跑時要加 --dump-mel")
            continue
        print(f"\n[diag] {sd.name}")
        m = diagnose_sample(sd, vocoder, args.device)
        results[sd.name] = m
        # per-sample json
        (sd / "diagnostic.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  SSIM(voc on GT)    = {m['ssim_vocoder_on_gt_mel']:.3f}")
        print(f"  SSIM(voc on recon) = {m['ssim_vocoder_on_recon_mel']:.3f}")
        print(f"  Δ                  = {m['ssim_delta']:+.3f}")
        print(f"  → {verdict_for_sample(m)}")

    write_report(results, args.listening_dir)


if __name__ == "__main__":
    main()