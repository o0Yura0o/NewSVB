"""
scripts/vocoder_identity_test.py
==================================

【這支檔案做什麼】
Phase 0 硬性 gate：驗證 pretrained HifiGAN（NSVB 自帶，於英文歌聲 PopBuTFy 訓練）
餵中文歌聲（M4Singer / VocalVerse）的 GT mel 是否仍能高品質重建。

【為什麼必要】
NSVB pretrained HifiGAN 是英文資料訓練；若餵中文歌聲 mel 已經斷裂，
後續 M 在 z 層做的所有改進都會被 vocoder 重建誤差淹沒——
聽測上完全感受不到 enhancement。這是**先驗的 dealbreaker**，必須在
任何訓練之前驗證。

【測試流程】
對 K 首歌（每資料集 K=20 預設）：
    1. 載入 wav，跑 audio_io.compute_mel → mel_gt
    2. 餵 mel_gt 給 vocoder → wav_recon
    3. 對 wav_recon 重新 compute_mel → mel_recon
    4. 算兩個指標：
        - mel SSIM(mel_gt, mel_recon)：頻譜結構相似度
        - F0 RMSE：對 wav_gt / wav_recon 各抽 F0，比較

【通過條件】
    mel SSIM   ≥  0.90   AND   F0 RMSE ≤  10 Hz   →  PASS
    0.85 ≤ SSIM < 0.90  OR   10 < RMSE ≤ 20      →  MARGINAL（聽測判定）
    其他                                          →  FAIL（fine-tune vocoder）

【為什麼 SSIM 而非 L1/L2 mel error】
- mel L1 對「均勻 dB 偏移」（vocoder 把所有頻段都偏 1 dB）會給高分但人耳聽不出差，
  反而對「peak 位置稍偏」這種人耳敏感的 artifact 給低分
- SSIM 看「結構」（峰、谷、對比、紋理），更貼近人耳對頻譜的感知
- skimage.metrics.structural_similarity 是廣泛驗證的實作

【為什麼也比 F0】
- mel SSIM 高不代表 F0 對：vocoder 可能把 mel 重建得很像但相位錯導致
  weak harmonic 跑掉，F0 抽取會偏；F0 對歌聲尤其關鍵
- 用 binarizer 共用的 torchcrepe（與訓練端一致），保證測量基準相同

【vocoder loader 是 stub】
- 此 script 寫好了所有指標 / I/O / CLI 邏輯，但實際 HifiGAN 模型載入是 stub
  （`load_vocoder()` 會 raise NotImplementedError），等 Phase 1 把 NSVB 的
  `modules/hifigan/` 移植進 `nsvb/vocoder/` 後再填上 ~10 行程式
- 這樣設計的好處：metric 基礎建設先備好，未來只需加 vocoder hookup 就能跑

【跨平台】
- 全部走 pathlib，Windows / Linux 皆可
- CLI 參數而非 hardcoded 路徑
- 報告寫 utf-8 JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from nsvb.utils.audio_config import SAMPLE_RATE
from nsvb.utils.audio_io import (
    load_wav,
    compute_mel,
    pad_wav_to_mel_length,
)
from nsvb.data.feature_extract.f0_torchcrepe import extract_f0


def extract_f0_parselmouth(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    用 parselmouth 抽 F0，**逐字 port NSVB `data_gen/tts/data_gen_utils.py:get_pitch`**
    （f0_min=80, f0_max=750, voicing_threshold=0.6, hop=128 對應 pad_size=4）。
    """
    import parselmouth
    from nsvb.utils.audio_config import HOP_SIZE

    time_step = HOP_SIZE / sample_rate * 1000  # ms
    f0_min = 80
    f0_max = 750
    pad_size = 4 if HOP_SIZE == 128 else 2

    f0 = parselmouth.Sound(wav, sample_rate).to_pitch_ac(
        time_step=time_step / 1000,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array["frequency"]

    lpad = pad_size * 2
    f0 = np.pad(f0, [[lpad, 0]], mode="constant")
    return f0.astype(np.float32)


def interp_f0_unvoiced(f0: np.ndarray) -> np.ndarray:
    """
    對 F0 的 unvoiced (== 0) 段做 log-space 線性內插，回傳「無 0 gap」的連續 F0。

    為什麼餵 vocoder 前要這樣做：
      NSVB 訓練 pipeline 中 F0 在進 vocoder 前會經過 norm_interp_f0 → denorm_f0：
        - norm_interp_f0: log2(f0+eps) 後對 unvoiced 段在 log space 做 np.interp 內插
        - denorm_f0(use_uv=False): 不再把 unvoiced 設回 0，保持內插值
      結果：vocoder SineGen 看到的是連續 F0 trajectory（無 0→Hz 硬跳）。

      若我們直接餵 voiced=Hz, unvoiced=0 的 raw F0：
        - SineGen 在 unvoiced 段 sine_wave=0
        - voiced 段邊界 sine_wave 突然從 0 跳到正弦相位 → 高頻 transient
        - 累積整段 → 電音 / 音高跑掉 artifact

    為什麼在 log space 內插：
      F0 是對數知覺的（半音差是固定 ratio），log-space 線性等於 Hz-space 幾何平均，
      過 unvoiced 段中央時 pitch 緩升 / 緩降，符合人類發聲漸變
    """
    f0 = f0.copy().astype(np.float32)
    uv = f0 == 0
    if uv.all():
        return f0
    if uv.any():
        # log space 線性內插
        log_f0 = np.log2(np.maximum(f0, 1e-8))
        log_f0[uv] = np.interp(
            np.where(uv)[0], np.where(~uv)[0], log_f0[~uv],
        )
        f0 = (2.0 ** log_f0).astype(np.float32)
    return f0


def extract_f0_pyworld(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    用 pyworld DIO + StoneMask 抽 F0（ver1 vocoder_identity_test.py 的方法）。

    為什麼提供這個替代抽法：
      pyworld 是 vocoder 領域的傳統 F0 工具，DIO+StoneMask 對 voiced/unvoiced 邊界
      與 phase 一致性的處理可能比 parselmouth/torchcrepe 更接近 vocoder 訓練時看到的
      F0 統計（即便 NSVB 訓練 binarizer 用的是 parselmouth，pyworld 在 phase 連續性上
      可能更平滑，不產生 unvoiced 段的硬切變）。

    為什麼 frame_period 寫成 ms：
      pyworld 規定，frame_period = 1000 * hop / sr (ms per frame)
    """
    import pyworld as pw
    from nsvb.utils.audio_config import HOP_SIZE
    f0, t = pw.dio(
        wav.astype(np.float64),
        sample_rate,
        frame_period=1000.0 * HOP_SIZE / sample_rate,
    )
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sample_rate)
    return f0.astype(np.float32)


# ── 通過閾值 ─────────────────────────────────────────
# 為什麼 0.90：經驗值；NSVB 原 paper 對 PopBuTFy 自身重建 SSIM ~0.95，
#             跨語言預期下降，0.90 是「人耳近不可分」的下限
SSIM_PASS = 0.90
SSIM_MARGINAL = 0.85
# 為什麼 10 Hz：人耳對 F0 的 just-noticeable difference 約半音（~6%）；
#               對 200 Hz 半音 = 12 Hz；10 Hz 是嚴格但合理的閾值
F0_RMSE_PASS = 10.0
F0_RMSE_MARGINAL = 20.0


# ── Vocoder 載入（stub）─────────────────────────────
def load_vocoder(ckpt_path: Path, device: str = "cuda") -> Callable:
    """
    載入 NSVB 作者提供的 vocoder ckpt（`1012_hifigan_all_songs_nsf`，
    實際是 HifiGAN-NSF，訓練自 100+ 小時中英文混合歌聲），回傳一個 callable：
        vocoder(mel: Tensor [1, NUM_MELS, T], f0: Tensor [1, T]) -> wav: Tensor [1, N_samples]

    【關於架構：是 HifiGAN-NSF 不是 PWG】
    雖然 ckpt 旁的 config.yaml 在 `generator_params:` 區塊看起來像 PWG，
    但 ckpt 的 state_dict 實際 keys（m_source, noise_convs, conv_pre, ups,
    resblocks, conv_post）證實是真 HifiGAN-NSF 架構。
    我們的 nsvb.backbone.vocoder.HifiGanNSFGenerator 與 ckpt state_dict
    完全一一對應（244 keys 全 match，strict=True 載入通過）。

    【為什麼餵 F0 (Hz) 而非 pitch index】
    HifiGAN-NSF 透過 SourceModuleHnNSF（內部 SineGen + Linear）直接吃連續 F0 (Hz)，
    不需要量化 pitch index（與 PWG 的 nn.Embedding(300) 設計不同）。
    我們直接餵 torchcrepe 抽出來的連續 F0。
    """
    from nsvb.backbone.vocoder import HifiGanNSFGenerator

    print(f"[vocoder] loading HifiGAN-NSF from {ckpt_path}", flush=True)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    if "model_gen" in sd:
        sd = sd["model_gen"]

    g = HifiGanNSFGenerator(c_out=1, num_mels=80, audio_sample_rate=22050)
    g.load_state_dict(sd, strict=True)
    g.to(device).eval()
    # 推理移除 weight_norm 加速 ~10%
    g.remove_weight_norm()
    print(f"[vocoder] loaded; params={sum(p.numel() for p in g.parameters())/1e6:.2f}M",
          flush=True)

    def _vocode(mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [1, NUM_MELS, T_mel]
            f0:  [1, T_mel]               Hz (連續值, 0=unvoiced)
        Returns:
            wav: [1, T_audio]              T_audio = T_mel * 128
        """
        with torch.no_grad():
            wav = g(mel, f0)  # [1, 1, T_audio]
        return wav.squeeze(1)

    return _vocode


# ── 指標計算 ─────────────────────────────────────────
def mel_ssim(mel_a: np.ndarray, mel_b: np.ndarray) -> float:
    """
    對兩個 mel spectrogram [T, NUM_MELS] 算 structural similarity。

    為什麼用 skimage 而非自寫：
      SSIM 公式有 stabilization constants（C1, C2, C3）需要根據 data range 調，
      skimage 的實作對 unit 自動處理；自寫容易差參數。

    為什麼把兩 mel 對齊到較短長度：
      vocoder 重建的 wav 長度可能比原 wav ±1 frame，重新 compute_mel 後 frame 數
      也會差 ±1；clip 到較短再比是安全的對齊方式。
    """
    from skimage.metrics import structural_similarity as ssim

    T = min(mel_a.shape[0], mel_b.shape[0])
    a = mel_a[:T]
    b = mel_b[:T]

    # mel 是 ln scale，range 約 [-11.5, +0.5]；data_range 取實際全資料範圍
    # 為什麼用 float()：numpy scalar 對 builtin max/min 的型別推斷在某些
    # IDE/mypy 嚴格模式會報警，明寫 float 排除歧義
    data_range = float(max(a.max(), b.max())) - float(min(a.min(), b.min()))
    if data_range < 1e-6:
        return float("nan")

    # multichannel=False 因為 mel 是 2D（time × bins，不是 RGB）
    score = ssim(a, b, data_range=data_range)
    return float(score)


def f0_rmse(wav_a: np.ndarray, wav_b: np.ndarray, device: str) -> Tuple[float, float]:
    """
    對兩段 wav 各抽 F0，計算 voiced frame 上的 RMSE 與 voiced ratio 一致性。

    Returns:
        rmse_hz:        在 (a 與 b 同時 voiced) 的 frame 上的 RMSE
        voiced_match:   兩段 voicing mask 的 IOU（越接近 1 表 voicing 判斷一致）

    為什麼只在 「同時 voiced」 frame 上算 RMSE：
      若 a voiced / b unvoiced，F0 差會無限大或 0，污染統計；
      用 IOU 另外監控 voicing 不一致的程度
    """
    f0_a, _ = extract_f0(wav_a, sample_rate=SAMPLE_RATE, device=device)
    f0_b, _ = extract_f0(wav_b, sample_rate=SAMPLE_RATE, device=device)

    T = min(f0_a.shape[0], f0_b.shape[0])
    f0_a = f0_a[:T]
    f0_b = f0_b[:T]

    voiced_a = f0_a > 0
    voiced_b = f0_b > 0
    both = voiced_a & voiced_b
    union = voiced_a | voiced_b

    if both.sum() == 0:
        rmse = float("nan")
    else:
        diff = f0_a[both] - f0_b[both]
        rmse = float(np.sqrt(np.mean(diff ** 2)))

    iou = float(both.sum() / max(union.sum(), 1))
    return rmse, iou


# ── 單檔測試 ────────────────────────────────────────
def test_one(
    wav_path: Path,
    vocoder: Callable,
    device: str,
    save_dir: Optional[Path] = None,
    f0_method: str = "torchcrepe",
    f0_interp: bool = False,
) -> dict:
    """
    對單一 wav 跑：
        load → mel_gt + F0_gt → vocoder(mel, F0) → wav_recon → mel_recon
    回傳 metric dict；若 save_dir 給了，把 GT 與 recon wav 都存下供人耳聽測。

    為什麼 vocoder 還要 F0：
      NSVB 1012 ckpt config 標 `use_pitch_embed: true`，generator 把 F0 做為
      pitch embedding 條件；少餵 F0 會 missing arg 或輸出畸變。
    """
    # 1. Load + (optional) loudness norm + GT mel
    # 為什麼預設不做 loudness：NSVB 訓練 vocoder 時沒有 loudness norm，
    # 加了會讓 mel 分布偏離 vocoder 期望（影響 vocoder identity test 結果）
    wav_gt = load_wav(str(wav_path))
    # NOTE: loudness_normalize() 移到 binarize 階段，vocoder identity test 直接用 raw wav
    mel_gt = compute_mel(wav_gt)
    wav_gt = pad_wav_to_mel_length(wav_gt, mel_gt)

    # 2. 抽 F0 給 vocoder 用（pitch embedding 條件必要）
    if f0_method == "parselmouth":
        f0_gt = extract_f0_parselmouth(wav_gt, sample_rate=SAMPLE_RATE)
    elif f0_method == "pyworld":
        f0_gt = extract_f0_pyworld(wav_gt, sample_rate=SAMPLE_RATE)
    else:
        f0_gt, _ = extract_f0(wav_gt, sample_rate=SAMPLE_RATE, device=device)
    # F0 對齊到 mel 長度（trim/pad ±1 frame）
    T_mel = mel_gt.shape[0]
    if f0_gt.shape[0] < T_mel:
        f0_gt = np.concatenate([f0_gt, np.zeros(T_mel - f0_gt.shape[0], dtype=f0_gt.dtype)])
    else:
        f0_gt = f0_gt[:T_mel]

    # 對 unvoiced 段做 log-space 線性內插，餵 vocoder 一個無 0 gap 的連續 F0
    # 對齊 NSVB 訓練 pipeline 的 norm_interp_f0 + denorm_f0(use_uv=False) 行為
    if f0_interp:
        f0_gt = interp_f0_unvoiced(f0_gt)

    # 3. Vocoder 重建
    mel_t = torch.from_numpy(mel_gt).float().T.unsqueeze(0).to(device)  # [1, NUM_MELS, T]
    f0_t = torch.from_numpy(f0_gt).float().unsqueeze(0).to(device)       # [1, T]
    with torch.no_grad():
        wav_recon_t = vocoder(mel_t, f0_t)  # [1, N] or [N]
    wav_recon = wav_recon_t.squeeze().cpu().numpy().astype(np.float32)

    # 4. 重新算 mel
    mel_recon = compute_mel(wav_recon)

    # 5. SSIM + F0 RMSE
    ssim_score = mel_ssim(mel_gt, mel_recon)
    rmse_hz, voicing_iou = f0_rmse(wav_gt, wav_recon, device=device)

    # 6. （可選）存 GT/recon wav 給人耳聽測
    # 為什麼成對存：聽 recon 自己無法判斷，必須與 GT 對比聽（音色/動態/雜訊）
    saved_gt = None
    saved_recon = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = wav_path.stem
        # 用 scipy.io.wavfile 寫 16-bit int wav（多數播放器最相容）
        from scipy.io import wavfile
        gt_path = save_dir / f"{stem}__gt.wav"
        recon_path = save_dir / f"{stem}__recon.wav"
        # clip 到 [-1, 1] 避免 int16 overflow
        gt_i16 = np.clip(wav_gt, -1.0, 1.0) * 32767
        recon_i16 = np.clip(wav_recon, -1.0, 1.0) * 32767
        wavfile.write(gt_path, SAMPLE_RATE, gt_i16.astype(np.int16))
        wavfile.write(recon_path, SAMPLE_RATE, recon_i16.astype(np.int16))
        saved_gt = str(gt_path)
        saved_recon = str(recon_path)

    return {
        "wav_path": str(wav_path),
        "duration_sec": float(len(wav_gt) / SAMPLE_RATE),
        "mel_ssim": ssim_score,
        "f0_rmse_hz": rmse_hz,
        "voicing_iou": voicing_iou,
        "saved_gt": saved_gt,
        "saved_recon": saved_recon,
    }


# ── Dataset 抽樣 ─────────────────────────────────────
def sample_wavs(root: Path, n: int, seed: int = 42) -> List[Path]:
    """
    從一個資料集根目錄遞迴抽 n 個 wav。

    為什麼用 seed：
      可重現；同一 seed 永遠抽同樣的測試集，方便比對不同 vocoder ckpt
    """
    candidates = sorted(root.rglob("*.wav"))
    if not candidates:
        raise RuntimeError(f"No .wav under {root}")
    rng = np.random.default_rng(seed)
    n = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(idx)]


# ── 整體 verdict ─────────────────────────────────────
def verdict(ssim_mean: float, rmse_mean: float) -> str:
    """根據聚合指標給 PASS / MARGINAL / FAIL。"""
    if ssim_mean >= SSIM_PASS and rmse_mean <= F0_RMSE_PASS:
        return "PASS"
    if ssim_mean < SSIM_MARGINAL or rmse_mean > F0_RMSE_MARGINAL:
        return "FAIL"
    return "MARGINAL"


# ── CLI ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vocoder identity test (Phase 0 gate)")
    parser.add_argument(
        "--vocoder-ckpt", required=True,
        help="HifiGAN vocoder checkpoint 路徑",
    )
    parser.add_argument(
        "--wav-dirs", nargs="+", required=True,
        help='格式："name=path"；例：m4=data/m4singer vocalverse=data/VocalVerse',
    )
    parser.add_argument(
        "--n-per-dir", type=int, default=20,
        help="每個資料集抽多少首做測試（預設 20）",
    )
    parser.add_argument(
        "--out-dir", default="outputs/phase0_vocoder",
        help="report.json 與抽樣 metric 輸出位置",
    )
    parser.add_argument(
        "--device", default=None, choices=[None, "cpu", "cuda"],
        help="預設 auto",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="抽樣可重現的 seed",
    )
    parser.add_argument(
        "--save-wavs", action="store_true",
        help="把 GT 與重建 wav 都存到 {out_dir}/wavs/{dataset}/ 供人耳聽測對比",
    )
    parser.add_argument(
        "--f0-method", default="torchcrepe", choices=["torchcrepe", "parselmouth", "pyworld"],
        help="F0 抽取方法。pyworld=ver1 用法 (DIO+StoneMask)；parselmouth=NSVB binarizer "
             "用法 (f0_min=80, f0_max=750)；torchcrepe=最精確但與 vocoder 訓練分布有差",
    )
    parser.add_argument(
        "--f0-interp", action="store_true",
        help="在送 vocoder 前對 unvoiced 段做 log-space 線性內插（連續 F0 trajectory）。"
             "對齊 NSVB norm_interp_f0 + denorm_f0(use_uv=False) 行為。"
             "若 voiced↔0 硬跳是電音元兇，這條會解決",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 "name=path" pairs
    dirs: dict = {}
    for spec in args.wav_dirs:
        if "=" not in spec:
            print(f"ERROR: --wav-dirs 應為 'name=path' 格式，得到 {spec!r}")
            sys.exit(1)
        name, path = spec.split("=", 1)
        dirs[name] = Path(path)

    # 載入 vocoder（目前是 stub；尚未實作會在這裡 raise）
    vocoder = load_vocoder(Path(args.vocoder_ckpt), device=device)

    overall_results = {}
    for name, root in dirs.items():
        print(f"\n[vocoder-test] === {name} @ {root} ===", flush=True)
        if not root.exists():
            print(f"[vocoder-test] WARNING: {root} not found, skipping")
            continue
        wavs = sample_wavs(root, n=args.n_per_dir, seed=args.seed)
        print(f"[vocoder-test] sampled {len(wavs)} wavs", flush=True)

        # 為每個 dataset 開獨立子目錄存 wav
        save_dir = (out_dir / "wavs" / name) if args.save_wavs else None

        per_file = []
        for i, wav_path in enumerate(wavs):
            result = test_one(wav_path, vocoder=vocoder, device=device,
                              save_dir=save_dir, f0_method=args.f0_method,
                              f0_interp=args.f0_interp)
            per_file.append(result)
            print(f"  [{i+1}/{len(wavs)}] {wav_path.name}: "
                  f"SSIM={result['mel_ssim']:.3f}  "
                  f"F0_RMSE={result['f0_rmse_hz']:.2f} Hz  "
                  f"voicing_IOU={result['voicing_iou']:.3f}",
                  flush=True)

        ssim_arr = np.array([r["mel_ssim"] for r in per_file])
        rmse_arr = np.array([r["f0_rmse_hz"] for r in per_file])
        v = verdict(ssim_arr.mean(), rmse_arr.mean())

        overall_results[name] = {
            "n_files": len(per_file),
            "mel_ssim_mean": float(ssim_arr.mean()),
            "mel_ssim_std": float(ssim_arr.std()),
            "f0_rmse_mean": float(rmse_arr.mean()),
            "f0_rmse_std": float(rmse_arr.std()),
            "verdict": v,
            "per_file": per_file,
        }
        print(f"[vocoder-test] {name} verdict={v}  "
              f"SSIM={ssim_arr.mean():.3f}±{ssim_arr.std():.3f}  "
              f"F0_RMSE={rmse_arr.mean():.2f}±{rmse_arr.std():.2f} Hz")

    # 寫 report
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "vocoder_ckpt": args.vocoder_ckpt,
            "thresholds": {
                "ssim_pass": SSIM_PASS,
                "ssim_marginal": SSIM_MARGINAL,
                "f0_rmse_pass": F0_RMSE_PASS,
                "f0_rmse_marginal": F0_RMSE_MARGINAL,
            },
            "results": overall_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[vocoder-test] report saved to {report_path}")

    # 整體 gate：所有資料集都要 PASS
    all_pass = all(r["verdict"] == "PASS" for r in overall_results.values())
    if all_pass:
        print("[vocoder-test] ✅ ALL DATASETS PASS — vocoder OK，可進 Stage 1")
        sys.exit(0)
    else:
        print("[vocoder-test] ❌ NOT ALL PASS — fine-tune vocoder 於中文歌聲後再進 Stage 1")
        sys.exit(2)


if __name__ == "__main__":
    main()