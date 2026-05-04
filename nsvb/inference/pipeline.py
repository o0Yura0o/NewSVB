"""
nsvb/inference/pipeline.py
============================

【這支檔案做什麼】
NSVB-ZH 的推理主管線，提供兩種 entrypoint：

    run_mode_a(models, features_a, **opts) → wav  (T_a-aligned)
        x_a → φ → z_a → M → z_a' → θ(z_a', f0_a, spk_emb_a) → mel → vocoder

    run_mode_b(models, features_a, features_p_ref, **opts) → wav  (T_p-aligned)
        x_a → φ → M → z_a'  (T_z_a)
        DTW(mel_a, mel_p_ref) → path
        z_a' → gather warp → z_a'_warped (T_z_p)
        decoder(z_a'_warped, f0_p_ref, spk_emb_a) → mel(T_p) → vocoder

對應 rebuild_checklist §H 的 Mode A / Mode B 設計。

【模式 A：純自動推理】
- 輸入：x_a 一個音檔
- 輸出長度：T_a（直接配原伴奏）
- F0：用 x_a 自身測得的 F0
- 適合：所有使用者的預設體驗

【模式 B：完全參考】
- 輸入：x_a + x_p_ref（同首歌專業參考）
- 輸出長度：T_p（跟 pro 參考的時長）
- F0：用 x_p_ref 的 F0
- 音色：用 x_a 的 spk_emb（Risk 4 防護，鎖音色）
- PPG：用 x_p_ref 的 PPG
        為什麼用 pro 端 PPG：x_p_ref 與 x_a 唱同一首歌（同詞同旋律），
        pro 端 PPG 在 T_p 軸上自然對齊；若 warp amateur PPG 到 T_p 反而引入 amateur
        歌詞咬字錯誤對下游的干擾
- 適合：使用者有同首歌專業 reference，想要 pro 的節奏 + 音高模板

【共用：encoder→M→decoder 三步】
封裝在 _encode_map_decode 函式（取自 stage2._encode_and_downsample +
_decode_with_mapped_z 的推理子集，去除訓練專屬的 mask 下採等）；
M(z) 用 deterministic 路徑（z = encoder.m_q，不 sample）與訓練端一致。

【為什麼推理用 m_q（posterior mean）而非 random sampled z】
訓練 stage2._encode_and_downsample 用 m_q 而非 sampled z（見其註解：
"deterministic 較穩；採樣 z 引入 noise，gradient signal 變糟"）。
推理保持一致——同一輸入應該每次得到同一輸出，sample 會引入隨機性破壞 reproducibility。

【F0 連續化（餵 vocoder 前）】
與 vocoder_identity_test.interp_f0_unvoiced 一致：把 unvoiced=0 的 frame
在 log-Hz space 線性內插，避免 SineGen 看到 0→Hz 硬跳產生電音 transient。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from nsvb.utils.audio_config import LATENT_DOWN_FACTOR, SAMPLE_RATE
from nsvb.inference.feature_pipeline import (
    InferenceFeatures,
    features_to_batch,
)
from nsvb.inference.model_loader import InferenceModels
from nsvb.inference.dtw_warp import dtw_path_mel, warp_latent


@dataclass
class InferenceResult:
    """推理結果容器。

    Attributes:
        wav:           [N_samples] float32 in [-1, 1]
        mel:           [T_mel, NUM_MELS] float32   產生的 mel（vocoder 輸入）
        f0_used:       [T_mel] float32   實際餵 decoder/vocoder 的 F0（連續內插後）
        sample_rate:   22050
        dtw_cost:      Mode B 才有值，否則 None
    """
    wav: np.ndarray
    mel: np.ndarray
    f0_used: np.ndarray
    sample_rate: int = SAMPLE_RATE
    dtw_cost: Optional[float] = None


# ── Padding 工具：mel 必須對齊到 latent stride ──────────
def _pad_batch_to_multiple(
    batch: dict,
    factor: int,
) -> Tuple[dict, int]:
    """
    把 batch 的時間軸（mel rate）pad 到 factor 的倍數，回傳 (新 batch, 原 T_mel)。

    為什麼必要：
      FVAE encoder 用 stride conv 下採（kernel=2*s, padding=s/2），輸出 T_z = T_mel // s；
      decoder 用 ConvTranspose(kernel=s, stride=s) 上採，輸出 T_z * s。
      若 T_mel 不是 s 的倍數，T_z * s < T_mel → decoder 輸出與 mel_mask 形狀不相符 → 錯誤。
      訓練時 binarizer + max_frames=600（恰好 4 的倍數）剛好繞過此問題；
      推理時 user 輸入長度任意，必須在這層補齊。

    為什麼 pad 0：
      mel: log10 silence ~ -10，pad 0 不是真 silence 但反正會被 mask 為 invalid，
           decoder 不會被影響
      f0/ppg: 0 = unvoiced / 無內容向量，與 dataset.collate_fn 一致
      mel_mask: 0 = padding（重要：vocoder 不該看 padding 段）

    Args:
        batch:  features_to_batch 的輸出
        factor: latent_down_factor（NSVB 4）

    Returns:
        new_batch, original_T_mel
    """
    T_mel = batch["mel"].shape[1]
    if T_mel % factor == 0:
        return batch, T_mel

    pad_t = factor - (T_mel % factor)
    T_new = T_mel + pad_t

    new_batch = {}
    # mel: [B, T, NUM_MELS]
    mel = batch["mel"]
    new_batch["mel"] = torch.cat([mel,
        torch.zeros(mel.shape[0], pad_t, mel.shape[2], device=mel.device, dtype=mel.dtype)
    ], dim=1)
    # ppg: [B, T, ppg_dim]
    ppg = batch["ppg"]
    new_batch["ppg"] = torch.cat([ppg,
        torch.zeros(ppg.shape[0], pad_t, ppg.shape[2], device=ppg.device, dtype=ppg.dtype)
    ], dim=1)
    # f0: [B, T]
    f0 = batch["f0"]
    new_batch["f0"] = torch.cat([f0,
        torch.zeros(f0.shape[0], pad_t, device=f0.device, dtype=f0.dtype)
    ], dim=1)
    # mel_mask: [B, T] — pad 為 0（標記 padding 段）
    mm = batch["mel_mask"]
    new_batch["mel_mask"] = torch.cat([mm,
        torch.zeros(mm.shape[0], pad_t, device=mm.device, dtype=mm.dtype)
    ], dim=1)
    # spk_emb: [B, D] 不變
    new_batch["spk_emb"] = batch["spk_emb"]
    return new_batch, T_mel


# ── F0 內插（移到此檔案以避免跨 script 引用） ────────────
def _interp_f0_unvoiced_torch(f0: torch.Tensor) -> torch.Tensor:
    """
    對 unvoiced (== 0) 段做 log-space 線性內插，回傳「無 0 gap」的連續 F0。
    Tensor 版本（vocoder 餵 GPU tensor 時用）。

    為什麼要 GPU 版本：
      整個 inference forward 都在 GPU，最後再拉到 CPU 做 numpy interp 然後又上 GPU
      不必要；torch.where + 索引可在 GPU 完成（雖然 np.interp 沒有原生 GPU 版，
      我們自己用 cumsum trick 也行，但 numpy 寫法清楚，這裡先 cpu→numpy→cpu→gpu，
      F0 是 [1, T] 小張量，拷貝成本可忽略）。
    """
    f0_np = f0.detach().cpu().numpy()
    out = np.empty_like(f0_np)
    for b in range(f0_np.shape[0]):
        x = f0_np[b].copy()
        uv = x == 0
        if uv.all():
            out[b] = x
            continue
        if uv.any():
            log_x = np.log2(np.maximum(x, 1e-8))
            log_x[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], log_x[~uv])
            x = (2.0 ** log_x).astype(np.float32)
        out[b] = x
    return torch.from_numpy(out).to(f0.device).to(f0.dtype)


# ── 共用：encoder → M（不含 decoder）────────────────────
@torch.no_grad()
def _encode_and_map(
    models: InferenceModels,
    batch: dict,
) -> torch.Tensor:
    """
    跑 frozen φ 拿 z（用 posterior mean m_q）然後過 M 得 z_mapped。

    Args:
        batch: features_to_batch 的輸出（mel/ppg/f0/spk_emb/mel_mask）

    Returns:
        z_mapped: [1, latent, T_z]
    """
    cvae = models.cvae
    M = models.M

    # 構建 condition g
    g = cvae.condition(batch["ppg"], batch["f0"], batch["spk_emb"])  # [1, gin, T_mel]

    # FVAE 期望 channel-first
    x = batch["mel"].transpose(1, 2)               # [1, NUM_MELS, T_mel]
    x_mask = batch["mel_mask"].unsqueeze(1)        # [1, 1, T_mel]
    g_sqz = cvae.fvae.g_pre_net(g)                  # [1, gin, T_z]

    # encoder 直接呼叫，拿 m_q（與訓練端 _encode_and_downsample 一致）
    _z_q, m_q, _logs_q, _x_mask_sqz = cvae.fvae.encoder(x, x_mask, g_sqz)

    # 過 M
    z_mapped = M(m_q)                               # [1, latent, T_z]
    return z_mapped


@torch.no_grad()
def _decode_to_mel(
    models: InferenceModels,
    z: torch.Tensor,
    ppg: torch.Tensor,
    f0: torch.Tensor,
    spk_emb: torch.Tensor,
    mel_mask: torch.Tensor,
) -> torch.Tensor:
    """
    把 z 餵回 frozen θ 解出 mel。z 與 (ppg, f0, mel_mask) 必須在同一時間軸：
        z 是 latent rate (T_z)，但 (ppg, f0, mel_mask) 是 mel rate (T_mel = T_z * down)。

    Returns:
        mel_recon: [1, T_mel, NUM_MELS]   time-major
    """
    cvae = models.cvae
    g = cvae.condition(ppg, f0, spk_emb)             # [1, gin, T_mel]
    mel_mask_expand = mel_mask.unsqueeze(1)          # [1, 1, T_mel]
    mel_recon = cvae.fvae.decoder(z, mel_mask_expand, g)  # [1, NUM_MELS, T_mel]
    return mel_recon.transpose(1, 2)                  # [1, T_mel, NUM_MELS]


@torch.no_grad()
def _vocode(
    models: InferenceModels,
    mel: torch.Tensor,
    f0: torch.Tensor,
    apply_f0_interp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mel + F0 → wav。

    Args:
        mel: [1, T_mel, NUM_MELS]
        f0:  [1, T_mel]   Hz, unvoiced=0

    Returns:
        wav:     [1, T_audio]
        f0_used: [1, T_mel]   實際餵 vocoder 的 F0（log-interp 後）

    為什麼預設 apply_f0_interp=True：
      vocoder SineGen 對 unvoiced=0 frame 邊界會產生電音 transient（vocoder_identity
      _test.interp_f0_unvoiced docstring 詳述）；推理一定要做。
      暴露為 flag 是為了 debug — 想看「無 interp 會多糟」可關掉對比。
    """
    f0_used = _interp_f0_unvoiced_torch(f0) if apply_f0_interp else f0
    # vocoder 期望 [1, NUM_MELS, T_mel]
    mel_cf = mel.transpose(1, 2)
    wav = models.vocoder(mel_cf, f0_used)             # [1, T_audio]
    return wav, f0_used


# ── Mode A ──────────────────────────────────────────────
@torch.no_grad()
def run_mode_a(
    models: InferenceModels,
    features_a: InferenceFeatures,
    apply_f0_interp: bool = True,
) -> InferenceResult:
    """
    Mode A：純自動推理（rebuild_checklist §H Mode A）。

    流程：
        x_a → φ → z_a → M → z_a' → θ(z_a', f0_a, spk_emb_a) → mel → vocoder

    Args:
        models:           load_inference_models 的回傳
        features_a:       業餘 input 的 InferenceFeatures
        apply_f0_interp:  vocoder 前 F0 log-interp（預設 True，避免電音）

    Returns:
        InferenceResult  (wav 長度 = T_mel * HOP_SIZE，可直接配原伴奏)

    為什麼 spk_emb 用 features_a.spk_emb：
      Mode A 沒有 pro 參考，spk_emb 必然來自 amateur；這個欄位本來就是
      Risk 4 的主防線（鎖音色），用業餘自己的 embedding 代表「保留業餘音色」。
    """
    raw_batch = features_to_batch(features_a, models.device)
    # 為什麼這裡 pad：FVAE encoder/decoder 的 stride conv 要求 T_mel % LATENT_DOWN_FACTOR == 0
    batch, T_mel_orig = _pad_batch_to_multiple(raw_batch, LATENT_DOWN_FACTOR)

    # 1. encoder → M
    z_mapped = _encode_and_map(models, batch)        # [1, latent, T_z]

    # 2. decoder（用 amateur 的 ppg/f0/spk_emb/mask）
    mel_out = _decode_to_mel(
        models, z_mapped,
        ppg=batch["ppg"], f0=batch["f0"],
        spk_emb=batch["spk_emb"], mel_mask=batch["mel_mask"],
    )                                                 # [1, T_mel_padded, NUM_MELS]

    # 3. vocoder
    wav, f0_used = _vocode(models, mel_out, batch["f0"],
                           apply_f0_interp=apply_f0_interp)

    # 4. trim 回原 T_mel 長度（在 wav / mel / f0 同步切掉 padding 段）
    # 為什麼 trim：vocoder 在 padding 段輸出是垃圾（mel=0 + f0=0 → SineGen 噪音），
    #              user 看到的 wav 必須對應到 input 長度
    from nsvb.utils.audio_config import HOP_SIZE
    n_samples_orig = T_mel_orig * HOP_SIZE
    wav_out = wav.squeeze(0).cpu().numpy()[:n_samples_orig].astype(np.float32)
    mel_np = mel_out.squeeze(0).cpu().numpy()[:T_mel_orig].astype(np.float32)
    f0_np = f0_used.squeeze(0).cpu().numpy()[:T_mel_orig].astype(np.float32)

    return InferenceResult(wav=wav_out, mel=mel_np, f0_used=f0_np)


# ── Mode B ──────────────────────────────────────────────
@torch.no_grad()
def run_mode_b(
    models: InferenceModels,
    features_a: InferenceFeatures,
    features_p_ref: InferenceFeatures,
    apply_f0_interp: bool = True,
    dtw_metric: str = "euclidean",
) -> InferenceResult:
    """
    Mode B：完全參考（rebuild_checklist §H Mode B）。

    流程：
        z_a (T_z_a) ← φ(x_a)
        z_a' (T_z_a) ← M(z_a)
        DTW(mel_a, mel_p_ref) → path_to_a (T_mel_p)
        z_a'_warped (T_z_p) ← gather(z_a', latent index)
        mel(T_mel_p) ← θ(z_a'_warped, ppg=ppg_p_ref, f0=f0_p_ref, spk_emb=spk_emb_a)
        wav ← vocoder(mel, f0_p_ref)

    Args:
        features_a:        業餘 input
        features_p_ref:    同首歌的 pro reference
        apply_f0_interp:   vocoder F0 連續化
        dtw_metric:        'euclidean' / 'cosine'，DTW 用的距離

    Returns:
        InferenceResult，wav 長度 = T_mel_p * HOP_SIZE（跟 pro 參考時長一致），
        dtw_cost 紀錄 normalized DTW cost（debug 用）

    為什麼 PPG 用 ppg_p_ref：
      pro reference 與 amateur 唱同詞同旋律；ppg_p_ref 在 T_p 時間軸上自然對齊，
      不需要 warp。warp amateur PPG 到 T_p 雖也可，但會引入 amateur 咬字模糊度
      對 decoder 的干擾。

    為什麼 spk_emb 用 features_a.spk_emb（不用 pro）：
      Risk 4：保留業餘歌手音色，否則整個 Mode B 等於把人「換成 pro」，
      偏離 NSVB-ZH 是「修飾技術而不換音色」的核心定位。

    為什麼 DTW 跑在 mel-rate 而非 latent-rate：
      mel 1 frame ~5.8 ms，足以分辨子音 / 短促 vibrato；
      latent 1 frame ~23 ms（down=4），對 phonetic 邊界已經太粗。
      mel-rate DTW 比較準。warp_latent 內部會把 mel-index 摺回 latent-index。
    """
    raw_a = features_to_batch(features_a, models.device)
    raw_p = features_to_batch(features_p_ref, models.device)
    # 兩邊都 pad 到 LATENT_DOWN_FACTOR 倍數（同 Mode A 的理由）；
    # padding 段對 DTW path 沒影響，因為我們會用「原始 T_mel_p」做 path 和最終 trim
    batch_a, T_mel_a_orig = _pad_batch_to_multiple(raw_a, LATENT_DOWN_FACTOR)
    batch_p, T_mel_p_orig = _pad_batch_to_multiple(raw_p, LATENT_DOWN_FACTOR)

    # 1. encoder + M（amateur 端，跑在 padded 長度上）
    z_mapped_a = _encode_and_map(models, batch_a)      # [1, latent, T_z_a_padded]

    # 2. DTW on mel-rate features（用「原始」mel，不含 padding）→ path
    # 為什麼用原始 mel：padding 段是 0，會嚴重污染 DTW cost matrix；
    # path 也要對應 user 真正聽得見的 frame
    path_to_a, dtw_cost = dtw_path_mel(
        mel_a=features_a.mel, mel_p=features_p_ref.mel, metric=dtw_metric,
    )                                                  # [T_mel_p_orig] int64

    # 3. warp z_mapped_a 到 T_z_p（pad 到 4 的倍數，方便 decoder 配對）
    # 為什麼這裡也要 pad path：z_warped 出來的 T_z 必須能被 decoder 上採到
    # batch_p 的 T_mel_p_padded；path 末段補上「最後一個 amateur frame」當 padding
    pad_p = T_mel_p_orig % LATENT_DOWN_FACTOR
    if pad_p != 0:
        pad_p = LATENT_DOWN_FACTOR - pad_p
        path_padded = np.concatenate([path_to_a,
                                       np.full(pad_p, path_to_a[-1], dtype=path_to_a.dtype)])
    else:
        path_padded = path_to_a
    z_warped = warp_latent(z_mapped_a, path_padded, latent_down_factor=LATENT_DOWN_FACTOR)

    # 4. decoder：用 pro reference 的 ppg/f0/mel_mask + amateur spk_emb
    mel_out = _decode_to_mel(
        models, z_warped,
        ppg=batch_p["ppg"], f0=batch_p["f0"],
        spk_emb=batch_a["spk_emb"],     # ← 用 amateur 音色（Risk 4 防護）
        mel_mask=batch_p["mel_mask"],
    )

    # 5. vocoder（餵 pro F0，含 padding 段）
    wav, f0_used = _vocode(models, mel_out, batch_p["f0"],
                           apply_f0_interp=apply_f0_interp)

    # 6. trim 回原始 T_mel_p 長度
    from nsvb.utils.audio_config import HOP_SIZE
    n_samples_orig = T_mel_p_orig * HOP_SIZE
    wav_out = wav.squeeze(0).cpu().numpy()[:n_samples_orig].astype(np.float32)
    mel_np = mel_out.squeeze(0).cpu().numpy()[:T_mel_p_orig].astype(np.float32)
    f0_np = f0_used.squeeze(0).cpu().numpy()[:T_mel_p_orig].astype(np.float32)

    return InferenceResult(wav=wav_out, mel=mel_np, f0_used=f0_np, dtw_cost=dtw_cost)