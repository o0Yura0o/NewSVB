"""估算本機 (RTX 3070) 跑 PopBuTFy end-to-end (binarize + 訓練) 所需時間 + VRAM 可行性。

【目的】
本研究計畫在 PopBuTFy(英文 paired SVB dataset)上重訓 Stage 2(沿用 v2 之 Stage 1
ckpt)以驗證 v2 之 pro_direction_alignment +0.84 解讀。本腳本量測 end-to-end
兩階段所需資源:

Phase A: binarize
  - 以實際 PopBuTFy mp3 跑 nsvb.data.binarizer.binarize_one
  - 量測 per-sample 時間 + Feature extractor 載入 VRAM + 處理 peak VRAM
  - 投射到 28965 chunks 總時間

Phase B: Stage 2 訓練
  - 以與 v2 完全相同之超參數 (lr_m=1e-4 / lr_dz=2e-4 / lambda_adv_mel=0.05 /
    freeze_d_mel / batch_size=16 / max_frames=1500) 實例化
  - 跑 warmup + 計時,量測 time-per-step + peak VRAM
  - OOM 自動退讓 batch_size
  - 投射到 120k steps 總時間

【為何兩階段分別量測 VRAM】
- binarize 之 GPU 用戶:Whisper-large-v3 PPG (~3 GB) + Resemblyzer (~50 MB) +
  torchcrepe F0 (~80 MB);base ~3 GB,加上長音訊處理 peak 可能再加 1-2 GB
- 訓練之 GPU 用戶:M (kernel=1 pointwise) + D_z + 凍 D_mel + Stage 1 CVAE (frozen)
  + batch=16 × T=1500 × 80mel × float32 ≈ 8 MB / batch,加上 activation peak
- 兩階段順序執行(非並行),需各自獨立 release GPU 後再起下一階段量測

【使用】
    python -m scripts.estimate_popbutfy_training_time

可選參數:
    --popbutfy-root      PopBuTFy mp3 根目錄(預設指向本機 NSVB 路徑)
    --binarize-warmup    binarize warmup 樣本數(預設 2)
    --binarize-measure   binarize 計時樣本數(預設 6)
    --train-warmup       訓練 warmup steps(預設 3)
    --train-measure      訓練計時 steps(預設 10)
    --skip-binarize      跳過 Phase A(若已知無需重估)
    --skip-train         跳過 Phase B
"""
import argparse
import gc
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.task.stage2 import Stage2Config, Stage2Trainer
from nsvb.data.binarizer import binarize_one, FeatureExtractors
from nsvb.data.popbutfy_adapter import list_popbutfy


DEFAULT_POPBUTFY_ROOT = r"C:\Users\neo29\workspace\SVC\NSVB\data\processed\PopBuTFy_new\data"


def fmt_gb(x_bytes: float) -> str:
    return f"{x_bytes / 1e9:.2f} GB"


def fmt_hr(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f} 秒"
    if sec < 3600:
        return f"{sec/60:.1f} 分"
    return f"{sec/3600:.2f} 小時"


# ────────────────────────────────────────────────────────────────
# Phase A: binarize 時間 + VRAM 量測
# ────────────────────────────────────────────────────────────────
def measure_binarize(popbutfy_root: Path, n_warmup: int, n_measure: int):
    """跑 binarize_one 於實際 PopBuTFy mp3,回傳 (per-sample 秒, peak VRAM, base VRAM, total)。"""
    print("\n" + "═" * 60)
    print("【Phase A】binarize 時間 + VRAM 量測")
    print("═" * 60)

    if not popbutfy_root.exists():
        print(f"[fatal] PopBuTFy root 不存在: {popbutfy_root}")
        sys.exit(1)

    # 掃 amateur + pro 兩 side
    print(f"[binarize] 掃描 {popbutfy_root}")
    amateur = list_popbutfy(popbutfy_root, "amateur")
    pro = list_popbutfy(popbutfy_root, "pro")
    all_specs = amateur + pro
    n_total = len(all_specs)
    print(f"[binarize] 找到 {len(amateur)} amateur + {len(pro)} pro = {n_total} chunks")

    # 隨機抽 warmup + measure 樣本(seed 固定可重現)
    rng = random.Random(42)
    n_pick = n_warmup + n_measure
    if n_pick > n_total:
        print(f"[fatal] 樣本不足:需 {n_pick},只有 {n_total}")
        sys.exit(1)
    picks = rng.sample(all_specs, n_pick)

    # ── 載入 extractors,量測 base VRAM ─────────────────────
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print("\n[binarize] 載入 FeatureExtractors (Whisper-large-v3 + Resemblyzer + torchcrepe)")
    t_load = time.time()
    try:
        extractors = FeatureExtractors(device="cuda")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[fatal] 載入 extractors 即 OOM,本機 8 GB VRAM 不足以跑 Whisper-large-v3")
        print(f"        錯誤: {e}")
        print(f"        建議:改用 --whisper-model openai/whisper-tiny(降低精度)或於 A100 跑")
        return None
    load_sec = time.time() - t_load
    vram_base = torch.cuda.memory_allocated()
    print(f"  載入耗時 {load_sec:.1f}s, 基準 VRAM = {fmt_gb(vram_base)}")

    # ── warmup ─────────────────────────────────────────
    print(f"\n[binarize] warmup {n_warmup} 個樣本(讓 cudnn / Whisper KV cache 穩定)")
    for i, spec in enumerate(picks[:n_warmup]):
        t0 = time.time()
        try:
            _ = binarize_one(spec, extractors, dereverb=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ⚠ OOM at warmup {i+1}: {spec.item_id}: {e}")
            del extractors
            torch.cuda.empty_cache()
            gc.collect()
            return None
        print(f"  warmup {i+1}/{n_warmup}: {Path(spec.wav_path).name}  {time.time()-t0:.2f}s")

    # ── 計時 + peak VRAM ───────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    print(f"\n[binarize] 計時 {n_measure} 個樣本")
    durs = []
    for i, spec in enumerate(picks[n_warmup:]):
        t0 = time.time()
        try:
            _ = binarize_one(spec, extractors, dereverb=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ⚠ OOM at measure {i+1}: {spec.item_id}: {e}")
            del extractors
            torch.cuda.empty_cache()
            gc.collect()
            return None
        dur = time.time() - t0
        durs.append(dur)
        print(f"  {i+1}/{n_measure}: {Path(spec.wav_path).name}  {dur:.2f}s")

    vram_peak = torch.cuda.max_memory_allocated()
    per_sample = mean(durs)
    std_sample = stdev(durs) if len(durs) > 1 else 0.0
    total_sec = per_sample * n_total

    # 釋放 GPU 給 Phase B
    del extractors
    torch.cuda.empty_cache()
    gc.collect()

    return dict(
        n_total=n_total,
        per_sample=per_sample,
        std_sample=std_sample,
        total_sec=total_sec,
        vram_base=vram_base,
        vram_peak=vram_peak,
        load_sec=load_sec,
    )


# ────────────────────────────────────────────────────────────────
# Phase B: Stage 2 訓練時間 + VRAM 量測 (v2 超參數)
# ────────────────────────────────────────────────────────────────
def v2_config(stage1_ckpt: str, batch_size: int, binarized_root: str) -> Stage2Config:
    """v2 訓練實際使用之超參數,1:1 對齊。"""
    return Stage2Config(
        latent_size=128,
        max_steps=120_000,
        batch_size=batch_size,
        max_frames=1500,
        num_workers=2,
        lambda_adv_z=1.0,
        lambda_patchnce=1.0,
        lambda_adv_mel=0.05,
        lambda_identity_pro=0.1,
        identity_pro_prob=0.2,
        lr_m=1e-4,
        lr_dz=2e-4,
        lr_dmel=1e-5,
        freeze_d_mel=True,
        d_z_warmup_steps=5000,
        binarized_root=binarized_root,
        amateur_dataset="vocalverse",   # proxy:同 model+batch shape,訓練耗時相同
        pro_dataset="m4singer",
        stage1_ckpt=stage1_ckpt,
        ckpt_dir="/tmp/popbutfy_estimate_dryrun",
        save_interval=999_999,
        log_interval=999_999,
        audio_quality_monitor_interval=999_999,
        f0_support_method="none",
    )


def try_one_batch_size(stage1_ckpt: str, binarized_root: str, batch_size: int,
                       warmup: int, measure: int):
    """嘗試以 batch_size 跑訓練,回傳 (success, per_step, peak_vram) 或 None。"""
    print(f"\n[try batch_size={batch_size}]")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    cfg = v2_config(stage1_ckpt, batch_size, binarized_root)
    try:
        trainer = Stage2Trainer(cfg)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM during init: {e}")
        return None
    except Exception as e:
        print(f"  ERROR during init: {type(e).__name__}: {e}")
        return None

    print(f"  warmup {warmup} steps")
    try:
        for i in range(warmup):
            trainer.train_step()
            trainer.step += 1
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM during warmup: {e}")
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        return None

    torch.cuda.reset_peak_memory_stats()
    print(f"  計時 {measure} steps")
    torch.cuda.synchronize()
    t_start = time.time()
    try:
        for i in range(measure):
            trainer.train_step()
            trainer.step += 1
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM during measure: {e}")
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        return None
    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    per_step = elapsed / measure
    peak_vram = torch.cuda.max_memory_allocated()

    print(f"  ✓ ok: {per_step*1000:.0f} ms/step, peak VRAM {fmt_gb(peak_vram)}")
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    return per_step, peak_vram


def measure_training(stage1_ckpt: str, binarized_root: str,
                     target_batch: int, warmup: int, measure: int):
    """v2 對照之訓練量測,batch_size OOM 自動退讓。"""
    print("\n" + "═" * 60)
    print("【Phase B】Stage 2 訓練時間 + VRAM 量測 (v2 超參數對照)")
    print("═" * 60)

    if not Path(stage1_ckpt).exists():
        print(f"[fatal] Stage 1 ckpt 不存在: {stage1_ckpt}")
        return None

    # 嘗試由大到小之 batch_size
    bs_candidates = [target_batch]
    bs = target_batch
    while bs > 1:
        bs = max(bs // 2, 1)
        bs_candidates.append(bs)
        if bs == 1:
            break
    bs_candidates = sorted(set(bs_candidates), reverse=True)

    for bs in bs_candidates:
        result = try_one_batch_size(stage1_ckpt, binarized_root, bs, warmup, measure)
        if result is not None:
            per_step, peak_vram = result
            return dict(
                batch_size=bs,
                per_step=per_step,
                peak_vram=peak_vram,
                total_sec=per_step * 120_000,
                target_batch=target_batch,
            )
    return None


# ────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────
def report(gpu_name: str, gpu_total: float, bin_res, train_res):
    print("\n" + "═" * 60)
    print("【總結】end-to-end 估算")
    print("═" * 60)
    print(f"  GPU                 : {gpu_name} ({fmt_gb(gpu_total)} VRAM)")

    total_sec = 0.0
    if bin_res is not None:
        print("\n── Phase A: binarize ──")
        print(f"  PopBuTFy 樣本數     : {bin_res['n_total']:,} chunks (amateur+pro)")
        print(f"  Per-sample 時間     : {bin_res['per_sample']:.2f} ± {bin_res['std_sample']:.2f} 秒")
        print(f"  Extractors 載入時間 : {bin_res['load_sec']:.1f} 秒 (一次性)")
        print(f"  Base VRAM (模型常駐): {fmt_gb(bin_res['vram_base'])} "
              f"({bin_res['vram_base']/gpu_total*100:.0f}%)")
        print(f"  Peak VRAM (含處理)  : {fmt_gb(bin_res['vram_peak'])} "
              f"({bin_res['vram_peak']/gpu_total*100:.0f}%)")
        print(f"  Binarize 總時間預估 : {fmt_hr(bin_res['total_sec'])}")
        total_sec += bin_res['total_sec']

    if train_res is not None:
        print("\n── Phase B: Stage 2 訓練 ──")
        bs = train_res['batch_size']
        tag = "  ← 跟 v2 一致 (16)" if bs == train_res['target_batch'] else \
              f"  ⚠ < v2 target ({train_res['target_batch']})"
        print(f"  最大可用 batch_size : {bs}{tag}")
        print(f"  Peak VRAM           : {fmt_gb(train_res['peak_vram'])} "
              f"({train_res['peak_vram']/gpu_total*100:.0f}%)")
        print(f"  Time per step       : {train_res['per_step']*1000:.0f} ms "
              f"({1/train_res['per_step']:.2f} it/s)")
        print(f"  120k steps 預估時間 : {fmt_hr(train_res['total_sec'])}")
        print(f"  對照 v2 (A100): 5.33 it/s / ~6.5 hr → 本機慢約 "
              f"{5.33/(1/train_res['per_step']):.1f} 倍")
        total_sec += train_res['total_sec']

    if bin_res is not None and train_res is not None:
        print("\n── End-to-end ──")
        print(f"  總計時間            : {fmt_hr(total_sec)}")
        print(f"  最大同時 VRAM 需求  : {fmt_gb(max(bin_res['vram_peak'], train_res['peak_vram']))} "
              f"(兩階段順序執行,取 max 為門檻)")
        if max(bin_res['vram_peak'], train_res['peak_vram']) > gpu_total * 0.9:
            print(f"  ⚠ Peak VRAM 接近上限 — 留意記憶體碎片化於長跑時可能 OOM")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--popbutfy-root", default=DEFAULT_POPBUTFY_ROOT,
                    help="PopBuTFy mp3 根目錄(內含 *_Amateur / *_Professional folders)")
    ap.add_argument("--stage1-ckpt", default="checkpoints/stage1/stage1_best.pt",
                    help="v2 之 Stage 1 ckpt")
    ap.add_argument("--binarized-root", default="data/binarized",
                    help="訓練 warmup 用之 binarized 根目錄(用 M4+VV 即可,model+batch shape "
                         "跟 PopBuTFy 相同,binarize 後訓練耗時不依賴資料內容)")
    ap.add_argument("--target-batch", type=int, default=16,
                    help="v2 之 batch_size(本機若 OOM 會自動退讓)")
    ap.add_argument("--binarize-warmup", type=int, default=2)
    ap.add_argument("--binarize-measure", type=int, default=6)
    ap.add_argument("--train-warmup", type=int, default=3)
    ap.add_argument("--train-measure", type=int, default=10)
    ap.add_argument("--skip-binarize", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[fatal] CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    print(f"[setup] GPU = {gpu_name}, total VRAM = {fmt_gb(gpu_total)}")
    print(f"[setup] PopBuTFy root = {args.popbutfy_root}")
    print(f"[setup] stage1 ckpt = {args.stage1_ckpt}")

    bin_res = None
    if not args.skip_binarize:
        bin_res = measure_binarize(
            Path(args.popbutfy_root),
            args.binarize_warmup,
            args.binarize_measure,
        )

    train_res = None
    if not args.skip_train:
        train_res = measure_training(
            args.stage1_ckpt,
            args.binarized_root,
            args.target_batch,
            args.train_warmup,
            args.train_measure,
        )

    report(gpu_name, gpu_total, bin_res, train_res)


if __name__ == "__main__":
    main()