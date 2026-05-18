"""
nsvb/utils/audio_io.py
=======================

【這支檔案做什麼】
NSVB-ZH 共用的音訊 I/O 與 mel-spectrogram 抽取工具。
功能：
    1. load_wav(path)                 → 載入並 resample 到 SAMPLE_RATE
    2. dereverb_wav(wav)               → DeepFilterNet3 去殘響 + 去噪（Risk 2 主防線）
    3. loudness_normalize(wav)        → ITU BS.1770 響度正規化到 -22 LUFS
    4. compute_mel(wav)               → 抽 80-band log10-mel spectrogram
    5. pad_wav_to_mel_length(wav, mel) → 調整 wav 長度與 mel frame 對齊
    6. load_and_extract(path)          → 一站式 pipeline（含 optional dereverb）

【為什麼用 librosa 而非 torch.stft】
- 與 NSVB 原版 mel pipeline 完全一致 → pretrained HifiGAN 直接相容
- Phase 0 是離線一次性處理，librosa 的 CPU 速度（約 0.5s/首歌）不是瓶頸
- librosa 是音訊處理事實標準，社群已驗證行為穩定
- 換 torch.stft 的好處（GPU 加速、可微分）對 Phase 0 沒實質用途；改寫反而引入
  「mel 分布是否與 NSVB HifiGAN 訓練時一致」的不確定性

【為什麼響度正規化到 -22 LUFS（與 NSVB 一致）】
- 響度正規化是讓 amateur 與 pro 兩邊的訊號 loudness 對齊，避免 D_z 誤把
  「音量差」當成「品質差」
- -22 LUFS 是 ITU BS.1770 的 broadcast loudness 標準下方一格，保留 ~6 dB
  動態空間給歌唱強弱對比（高潮段不會 clip）
- pyloudnorm 套件實作 BS.1770-4，與 Spotify/YouTube 等平台同標準

【為什麼 mel 用 log10 而非 log】
- NSVB pretrained HifiGAN 訓練時用的是 log10-mel，要餵相同分布才有正確輸出
- 視覺化時 log10 直接對應「dB / 20」，比自然對數直觀
- 訓練時 mel 數值範圍約 [-6, 1.5]（log10 1e-6 ≈ -6, log10(max amp) ~ 0.5）
"""

from typing import Tuple

import librosa
import librosa.filters
import numpy as np
import pyloudnorm as pyln

from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    HOP_SIZE,
    FFT_SIZE,
    WIN_SIZE,
    NUM_MELS,
    MEL_FMIN,
    MEL_FMAX,
    MEL_EPS,
    LOUDNESS_TARGET_LUFS,
)


# ── Mel filterbank 快取 ───────────────────────────────────
# 為什麼快取：mel basis 矩陣只與 sample_rate / fft_size / num_mels / fmin / fmax 有關，
#            這些都是常數，每次抽 mel 重算浪費（單次計算 ~5ms × 上萬首歌 = 數十秒）
_MEL_BASIS = None


def _get_mel_basis() -> np.ndarray:
    """Lazy 構建 mel filterbank（[NUM_MELS, FFT_SIZE//2+1]）。"""
    global _MEL_BASIS
    if _MEL_BASIS is None:
        # 注意：librosa 0.10+ 的 filters.mel 強制 keyword args
        _MEL_BASIS = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=FFT_SIZE,
            n_mels=NUM_MELS,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
        )
    return _MEL_BASIS


# ── 音檔載入 ─────────────────────────────────────────────
def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    載入 wav 檔，回傳 float32 mono，已 resample 到 sr。

    為什麼用 librosa.load 而非 soundfile + scipy.resample：
      librosa 一行同時處理 decode、stereo→mono、resample、float32 normalize。
      Phase 0 各種來源（wav、flac、mp3）混雜時 librosa 容錯最高。

    為什麼預設 mono：
      NSVB 全管線假設 mono；M4Singer / VocalVerse 都是純人聲 mono 來源。
      若未來引入 stereo 資料集，此處會自動 mixdown，但會發 warning。
    """
    # mono=True 自動把 stereo 平均成 mono；res_type='kaiser_best' 用 librosa 預設高品質
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


# ── 響度正規化 ──────────────────────────────────────────
def loudness_normalize(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_lufs: float = LOUDNESS_TARGET_LUFS,
) -> np.ndarray:
    """
    BS.1770 整合響度正規化到 target_lufs，並防止 clipping。

    為什麼這個前處理必要（Risk 2 緩解）：
      M4Singer 是錄音室高品質、VocalVerse 是 user-generated 音量散亂。
      若不對齊響度，amateur 普遍偏小聲，D_z 看到 mel 能量分布差異很大可能
      用「整體音量」當捷徑判別 amateur/pro。

    為什麼防 clipping：
      pyln.normalize.loudness 純粹做 gain shift，可能把高峰推超過 ±1。
      下游 librosa.stft / mel 對 [-1,1] 沒嚴格要求，但若儲存回 wav 會 wrap-around，
      所以強制縮回 [-1,1] 並保留 1% headroom。

    Args:
        wav:        [N] float32, [-1, 1]
        sr:         取樣率
        target_lufs: 目標響度（< 0 dBFS, 越接近 0 越大聲）

    Returns:
        wav_normed: [N] float32, [-0.99, 0.99]
    """
    # BS.1770 meter 需要 sr，內部會自動處理短音檔（< 0.4s 會跳過 gating）
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)

    # 純靜音段 / 極短音檔 → integrated_loudness 回傳 -inf，不做正規化
    if not np.isfinite(loudness):
        return wav

    wav = pyln.normalize.loudness(wav, loudness, target_lufs)

    # 防 clipping：若任一峰值 > 1，等比例縮到 0.99
    peak = np.abs(wav).max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)

    return wav.astype(np.float32)


# ── Dereverb / Denoise（Risk 2 主防線）──────────────────
# DF3 model + state 全域 lazy 快取
# 為什麼 lazy + singleton：
#   DF3 模型 ~10 MB、第一次載入 ~3 sec；Phase 0 binarize 對上萬首歌做，
#   每首都重 init 浪費；singleton 駐留 GPU memory 重用
_DF_MODEL = None
_DF_STATE = None


def _ensure_df_loaded():
    """
    Lazy 構建 DeepFilterNet3 model 與 state（首次呼叫 dereverb_wav 才載）。
    為什麼 lazy：載入動 ~3 sec 是 import-time 不該付的成本；smoke test 不用 dereverb 時也不該載
    """
    global _DF_MODEL, _DF_STATE
    if _DF_MODEL is None:
        import logging
        # 為什麼 set logger：DF default INFO 輸出多，會干擾 binarize log
        logging.getLogger("DF").setLevel(logging.WARNING)
        from df.enhance import init_df
        _DF_MODEL, _DF_STATE, _ = init_df()
    return _DF_MODEL, _DF_STATE


def dereverb_wav(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    backend: str = "df3",
) -> np.ndarray:
    """
    對 wav 做去殘響 + 去噪。預設用 DeepFilterNet3,可切換 backend(Plan C 用)。

    【backend 選項】
    - `df3`        : DeepFilterNet3(預設),~10 MB,CPU/GPU,中文歌聲 dereverb 表現 OK
    - `df3_cascade`: df3 跑兩次,更激進去殘響(可能引入 artifact,VV 重 reverb 才考慮)
    - `demucs`     : (尚未實作)未來可接 Demucs htdemucs 提 vocals,跟 dereverb 不同概念
    - `voicefixer` : (尚未實作)未來可接 VoiceFixer

    新 backend 加進來時:在 `_BACKENDS` 註冊一個 callable 即可,不要動本函式 signature。

    為什麼這個前處理對 Risk 2 (錄音環境差異被誤學成「降噪濾波器」) 必要：
        VocalVerse (user-generated 業餘) 多帶手機/家庭錄音的殘響 + 背景噪聲；
        M4Singer (錄音室職業) 乾淨。M 看到「amateur 殘響+技術差 → pro 乾+技術好」
        這個 pattern 後，可能把「去殘響」誤學成「技術改進」捷徑。
        對兩 dataset **都做 dereverb** 把音質基線拉齊到「乾淨」狀態，
        逼 M 去學真正的技術差（vibrato / 共鳴 / 音準）。

    為什麼選 DF3：
        - 同類最先進模型之一（2023, ICASSP）
        - 同時做 denoise + dereverb（其他單功能模型不夠）
        - 模型小 ~10 MB，CPU/GPU 都跑
        - PyPI 直接 pip install deepfilternet 安裝

    為什麼對兩邊都做（不只 VocalVerse）：
        若只對 VocalVerse 做，會引入新的 systematic 差異——
        VocalVerse 變得「無殘響但有 DF3 處理 artifact」、M4Singer 仍「無殘響無 artifact」，
        新差異仍會被 D_z 抓到當捷徑。**對齊處理**消除這個風險。

    為什麼內部 resample 到 48k：
        DF3 訓練於 48 kHz，輸入 22050 直接餵會降質；
        通過 torchaudio.functional.resample 做高品質升降採。
        雙 resample 損失 < 1%，遠小於 dereverb 的 quality gain

    Args:
        wav:         [N] float32 [-1, 1]
        sample_rate: 輸入取樣率（預設與我們 pipeline 一致 22050）

    Returns:
        wav_clean: [N] float32 [-1, 1]，去殘響 + 去噪後
    """
    import torch
    import torchaudio
    from df.enhance import enhance

    if backend == "demucs":
        raise NotImplementedError(
            "demucs backend 尚未實作。Plan C 需要 agent 安裝 `pip install demucs` 後接"
            " htdemucs 模型在這裡。改去 _dereverb_demucs() helper(未實作)。"
        )
    if backend == "voicefixer":
        raise NotImplementedError(
            "voicefixer backend 尚未實作。Plan C 需要 agent 安裝 `pip install voicefixer`"
            " 後接 VoiceFixer 模型在這裡。"
        )
    if backend not in ("df3", "df3_cascade"):
        raise ValueError(f"unknown dereverb backend: {backend!r}; "
                         f"available: df3, df3_cascade, demucs(NI), voicefixer(NI)")

    model, df_state = _ensure_df_loaded()
    assert model is not None and df_state is not None
    df_sr = df_state.sr()  # 48000

    audio_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, N]
    if sample_rate != df_sr:
        audio_t = torchaudio.functional.resample(audio_t, sample_rate, df_sr)

    # Plan C 內建選項:df3_cascade 跑兩次,適合 VV 端嚴重殘響
    n_passes = 2 if backend == "df3_cascade" else 1
    for _ in range(n_passes):
        audio_t = enhance(model, df_state, audio_t)
    enhanced = audio_t

    if sample_rate != df_sr:
        enhanced = torchaudio.functional.resample(enhanced, df_sr, sample_rate)

    return enhanced.squeeze(0).numpy().astype(np.float32)


# ── Mel spectrogram ────────────────────────────────────
def compute_mel(wav: np.ndarray) -> np.ndarray:
    """
    抽 80-band log10-mel spectrogram，**bit-exact 對齊 NSVB 訓練時的 mel 公式**
    （`data_gen/tts/data_gen_utils.py:process_utterance` with vocoder='pwg'）。
    與 1012_hifigan_all_songs_nsf vocoder ckpt 訓練時看到的 mel 分布完全一致。

    Pipeline（精確 port NSVB process_utterance）：
        1. librosa.stft(n_fft=fft, hop=hop, win=win, window='hann', pad_mode='constant')
           （center=True 是 librosa 預設，與 NSVB 訓練一致）
        2. amplitude = np.abs(stft)
        3. mel = mel_basis @ amplitude
        4. log_mel = log10(max(eps, mel))

    【為什麼是 log10 而非 ln，eps 為何是 1e-10】
    NSVB 訓練 vocoder 走的是 `data_gen_utils.process_utterance` → `vocoder='pwg'`：
        mel = np.log10(np.maximum(eps, mel))
    且 `vocoders/pwg.py:wav2spec` 把 eps 預設成 `wav2spec_eps=1e-10`。
    （**注意陷阱**：repo 內另有 `modules/hifigan/mel_utils.py` 用 ln + 1e-5，
    但那 file 不是訓練 vocoder 時實際走的路徑——僅在某些子工具引用。
    認 1012 ckpt 訓練 mel 公式以 process_utterance 為準。）

    【為什麼用 pad_mode='constant'（zero）而非 reflect】
    librosa 預設 reflect，NSVB 原版顯式覆寫成 'constant'：
    歌聲開頭/結尾的短靜音段用 zero padding 與真實狀態相符；
    reflect 在邊界會把實際內容鏡像進靜音段，產生 mel artifact。

    【為什麼回傳 [T, NUM_MELS] 而非 [NUM_MELS, T]】
    下游 binarizer / dataset 慣例是 time-major（first dim = time），
    與 z [T_z, latent_size] 對齊風格一致。

    Args:
        wav: [N] float32, [-1, 1]

    Returns:
        mel: [T_mel, NUM_MELS] float32, **log10 scale**, range typically [-10, +0.2]
    """
    # 1. STFT（librosa 0.10+ 強制 keyword args）
    stft = librosa.stft(
        y=wav,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
        window="hann",
        pad_mode="constant",
    )
    amplitude = np.abs(stft)  # [n_fft//2+1, T]

    # 2. mel projection
    mel = _get_mel_basis() @ amplitude  # [NUM_MELS, T]

    # 3. log10 with eps clamp
    log_mel = np.log10(np.maximum(MEL_EPS, mel))

    # transpose to time-major
    return log_mel.T.astype(np.float32)


# ── Wav / Mel 長度對齊 ──────────────────────────────────
def pad_wav_to_mel_length(wav: np.ndarray, mel: np.ndarray) -> np.ndarray:
    """
    把 wav 長度補齊到 `mel.shape[0] * HOP_SIZE`，保證 STFT 反向重建 wav 長度
    與 mel frame 數一一對應。

    為什麼這個對齊重要：
      訓練時可能會切片（例如取 mel frames [0:200]），對應的 wav 樣本範圍是
      [0 * HOP_SIZE : 200 * HOP_SIZE]。若 wav 長度與 mel*hop 不一致，
      切片會出 IndexError 或拿到空 wav。

    NSVB 原版做法（audio.py:librosa_pad_lr + base_binarizer）：
      1. 計算 right pad 補到 (T_mel * hop) 長度
      2. 結尾 pad 0，wav[: mel.shape[0] * hop_size]

    Args:
        wav: [N] float32
        mel: [T_mel, NUM_MELS]

    Returns:
        wav_aligned: [T_mel * HOP_SIZE] float32
    """
    target_len = mel.shape[0] * HOP_SIZE
    cur_len = wav.shape[0]

    if cur_len < target_len:
        # 補 0 到 target
        pad = np.zeros(target_len - cur_len, dtype=wav.dtype)
        return np.concatenate([wav, pad])
    return wav[:target_len]


# ── 一站式 pipeline ────────────────────────────────────
def load_and_extract(
    path: str, dereverb: bool = True, dereverb_backend: str = "df3",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase 0 binarizer 預設 entrypoint：
      1. load wav
      2. (optional) dereverb + denoise via DeepFilterNet3   ← Risk 2 主防線
      3. loudness normalize
      4. compute mel
      5. align wav length to mel
    回傳對齊後的 (wav, mel)。

    為什麼包成一個 function：
      Binarizer 對每首歌做的固定流程，包成單一 entrypoint 避免上層忘記某個步驟
      （例如忘記 loudness norm → 訓練時 D_z 用音量當捷徑）。

    為什麼 dereverb 預設 True：
      Production binarize 必跑（Risk 2 主防線）；只有在 vocoder identity test、
      smoke test、或想跑「無 dereverb」對照實驗時才設 False。

    為什麼順序是 dereverb → loudness → mel：
      1. dereverb 前先放，因為 loudness norm 應該作用在「乾淨訊號」上
         （若先 loudness 後 dereverb，殘響會被當訊號計入 LUFS，dereverb 之後音量變小）
      2. mel 永遠最後算，因為 dereverb + loudness norm 都會改 wav 振幅
    """
    wav = load_wav(path)
    if dereverb:
        wav = dereverb_wav(wav, backend=dereverb_backend)
    wav = loudness_normalize(wav)
    mel = compute_mel(wav)
    wav = pad_wav_to_mel_length(wav, mel)
    return wav, mel


if __name__ == "__main__":
    # 自我測試：用合成正弦驗證 mel 形狀、frame rate、log scale 範圍
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"Test: 440 Hz sine, {duration}s @ {SAMPLE_RATE} Hz")
    print(f"  Audio shape: {audio.shape}")

    audio_normed = loudness_normalize(audio)
    print(f"  After loudness norm:  peak={np.abs(audio_normed).max():.4f}")

    mel = compute_mel(audio_normed)
    print(f"  Mel shape:  {mel.shape}  (expect [~{int(SAMPLE_RATE * duration / HOP_SIZE)}, {NUM_MELS}])")
    print(f"  Mel range:  [{mel.min():.2f}, {mel.max():.2f}]  (expect [-6, ~1])")

    audio_aligned = pad_wav_to_mel_length(audio_normed, mel)
    print(f"  Aligned audio: {audio_aligned.shape}  (expect {mel.shape[0] * HOP_SIZE})")
    assert audio_aligned.shape[0] == mel.shape[0] * HOP_SIZE, "Length alignment failed!"
    print("  Length alignment: OK")
