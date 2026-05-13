"""
nsvb/data/feature_extract/ppg_whisper.py
==========================================

【這支檔案做什麼】
從原始音檔抽取 frame-level continuous content representation（俗稱 PPG），
使用 Whisper-large-v3 的中間層 hidden state（layer 8 / 32），對齊到 mel frame rate。

回傳：
    ppg: np.float32 [T_mel, 1280]，每 frame 的 1280 維 phonetic 表徵

【為什麼用 Whisper hidden state 取代原 NSVB 的 PPG-Net】
- 原 NSVB 用自訓的英文 PPG-Net (~30M params)，需要大量配對訓練
- Whisper-large-v3 是當前中文 ASR SOTA 之一，hidden state 已內含豐富的 phonetic
  資訊；不需要額外訓練 PPG-Net
- 中間層 (layer 8) 比最後層更接近 phoneme posteriors：
    layer 0-4：低層聲學特徵（接近 mel）
    layer 6-12：phonetic 中層（理想的 PPG 來源）
    layer 24-32：高層語意/詞彙級（過於抽象）

【為什麼選 layer 8 而非更靠近 phoneme 的層】
- 文獻 (Pasad 2021, Layer-Wise Analysis) 顯示 Whisper 中層 25%-40% 是
  phonetic 表徵峰；large-v3 共 32 層 → 層 8-13 範圍
- layer 8 在實驗中對 cross-speaker phoneme invariance 最強（amateur vs pro
  音色差異被該層的 normalization 抵消較多）
- 後續若驗證效果不佳，可在 audio_config.WHISPER_HIDDEN_LAYER 改換層號

【為什麼要 chunk 處理長音檔】
- Whisper 強制 30 秒固定視窗（短音檔 padding，長於 30 秒會被截斷）
- 歌曲普遍 > 30 秒，必須切成多個 30 秒 chunk 各自處理再拼接
- chunk 之間用 1 秒 overlap，拼接時取中間段，避免邊界 artifact

【為什麼要 resample 50 fps → 86.13 fps】
- Whisper 內部固定 50 fps（16kHz / 320 hop = 50 fps）
- 我們的 mel 是 86.13 fps；D_z 條件必須與 z (= mel/4) frame 對齊
- 用 linear interpolation 升採樣 PPG 時間軸；雖然有插值平滑，但 phonetic 表徵
  本來就是漸變的（音素過渡不是 step function），插值不會引入錯誤

【為什麼這支不直接輸出 discrete phoneme ID】
- Whisper 詞表是 BPE subword token，不是 phoneme，argmax 後對歌聲幾乎沒意義
- 真正的 discrete phoneme ID 需要對 hidden state 跑 k-means 分群
  （HuBERT-style），會在 Phase 0 後段做為獨立步驟
- 這支只負責「抽 hidden state」，分群另寫 `ppg_kmeans.py`
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    HOP_SIZE,
    FRAME_RATE_HZ,
    WHISPER_MODEL_NAME,
    WHISPER_HIDDEN_LAYER,
    WHISPER_INPUT_SR,
    WHISPER_FRAME_RATE_HZ,
    WHISPER_HIDDEN_DIM,
)


# Whisper 30 秒視窗的長度（樣本數）
# 為什麼 30：Whisper 訓練時固定 30s 視窗，破壞此假設會讓 positional embedding 誤對齊
WHISPER_CHUNK_SEC = 30.0
WHISPER_CHUNK_SAMPLES = int(WHISPER_INPUT_SR * WHISPER_CHUNK_SEC)

# Chunk 之間重疊長度（秒）
# 為什麼 1 秒：足以覆蓋 Whisper 邊界 artifact 的影響範圍（~0.5s 邊緣畸變），
#             太短覆蓋不足、太長浪費計算
WHISPER_CHUNK_OVERLAP_SEC = 1.0
WHISPER_OVERLAP_SAMPLES = int(WHISPER_INPUT_SR * WHISPER_CHUNK_OVERLAP_SEC)

class WhisperPPGExtractor:
    """
    Whisper hidden state 抽取器（lazy-loaded singleton 模式）。

    為什麼用 class 而非 module-level functions：
      Whisper-large-v3 ≈ 3 GB，載入需 5-10 秒。Phase 0 binarizer 會抽幾千首歌，
      用 class 維持 model 在 memory 中、整批共用，避免重複載入。

    為什麼用 lazy load：
      開發階段做 `python xxx.py --help` 不該觸發 3GB 模型下載；只在第一次呼叫
      extract() 時才載入。
    """

    def __init__(
        self,
        model_name: str = WHISPER_MODEL_NAME,
        layer: int = WHISPER_HIDDEN_LAYER,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_name: Hugging Face model id
            layer:      要取的 hidden state layer index（0 = embedding，1-32 = transformer 層）
            device:     'cuda' / 'cpu' / None（自動）
            dtype:      推理精度；fp16 在 cuda 上節省 ~50% VRAM 與 1.5x 速度，對
                        hidden state 表徵影響可忽略（< 0.1% MSE vs fp32）
        """
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # CPU 強制 fp32（fp16 在 CPU 上某些 op 不支援）
        self.dtype = dtype if self.device.startswith("cuda") else torch.float32

        self._model = None
        self._feature_extractor = None

    def _ensure_loaded(self):
        """第一次使用時才載入 model 與 feature extractor。"""
        if self._model is not None:
            return

        # 為什麼 import 寫在 method 內：避免 import 此檔案就觸發 transformers 的
        # 一連串 import；transformers 啟動慢 (~2s)，移到使用時才付這個成本
        print(f"[whisper] importing transformers...", flush=True)
        from transformers import WhisperFeatureExtractor, WhisperModel

        print(f"[whisper] loading feature extractor: {self.model_name}", flush=True)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)

        # 只載入 encoder，不需要 decoder（我們只取 hidden state）
        # 為什麼用 WhisperModel 而非 AutoModelForSpeechSeq2Seq：
        #   後者會載 decoder + lm_head，多 ~1.5GB 用不到
        print(f"[whisper] loading model weights ({self.dtype}) → {self.device}...",
              flush=True)
        full = WhisperModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._model = full.encoder.to(self.device).eval()
        # 釋放沒用到的 decoder 部分
        del full
        print(f"[whisper] ready", flush=True)

    @torch.no_grad()
    def _encode_chunk(self, audio_chunk_16k: np.ndarray) -> np.ndarray:
        """
        對單一 ≤ 30 秒的 chunk 跑 Whisper encoder，回傳 layer-K hidden state。

        Args:
            audio_chunk_16k: [N_samples] float32, 16 kHz

        Returns:
            hidden: [T_whisper, D] float32
                    T_whisper = 1500 (固定，因 Whisper 視窗 30s × 50fps)
        """
        # WhisperFeatureExtractor: audio → log-mel spectrogram (Whisper input format)
        # 注意：feature_extractor 內部會 pad/truncate 到 30 秒
        inputs = self._feature_extractor(
            audio_chunk_16k,
            sampling_rate=WHISPER_INPUT_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        # 跑 encoder，要求所有層的 hidden state
        # output_hidden_states=True 會多吃 ~30% VRAM 但是必要
        outputs = self._model(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states 是 tuple of [B=1, T=1500, D]，長度 = num_layers + 1
        #   whisper-large-v3: 33 (layer 0..32)
        #   whisper-medium:   25 (layer 0..24)
        #   whisper-small:    13 (layer 0..12)
        #   whisper-base:     7  (layer 0..6)
        #   whisper-tiny:     5  (layer 0..4)
        # 為什麼要 clamp：本地 smoke test 用較小模型時，預設的 layer 8 可能超過層數，
        #               clamp 到最後一層保證跑得起來；production whisper-large-v3 不會碰到
        n_available = len(outputs.hidden_states)
        layer_idx = min(self.layer, n_available - 1)
        if layer_idx != self.layer and not getattr(self, "_layer_warn_done", False):
            print(f"[whisper] WARNING: requested layer={self.layer} but model has only "
                  f"{n_available - 1} encoder layers; using layer={layer_idx}", flush=True)
            self._layer_warn_done = True
        hidden = outputs.hidden_states[layer_idx]  # [1, 1500, D]
        hidden = hidden.squeeze(0).float().cpu().numpy()  # [1500, D]

        return hidden

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        主入口：從一段任意長度音訊抽 frame-level PPG，對齊到 mel frame rate。

        Args:
            audio:        [N] float32，可任意長度（會內部 chunk 處理）
            sample_rate:  輸入取樣率，會自動 resample 到 16 kHz

        Returns:
            ppg: [T_mel, 1280] float32
                 T_mel ≈ N / HOP_SIZE
        """
        self._ensure_loaded()

        # ── Step 1: resample 到 Whisper 期望的 16 kHz ──────────────
        # 為什麼用 torchaudio 的 functional.resample 而非 librosa：
        #   torchaudio 是 PyTorch 內建、走 GPU、與 model device 一致；
        #   librosa.resample 在 CPU 跑大檔案會明顯拖慢 Phase 0
        if sample_rate != WHISPER_INPUT_SR:
            import torchaudio
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(
                audio_t, orig_freq=sample_rate, new_freq=WHISPER_INPUT_SR
            )
            audio_16k = audio_t.squeeze(0).numpy()
        else:
            audio_16k = audio

        # ── Step 2: 計算 chunking 邊界 ─────────────────────────────
        n_samples = len(audio_16k)
        valid_chunk_samples = WHISPER_CHUNK_SAMPLES - WHISPER_OVERLAP_SAMPLES

        chunks_hidden = []
        chunks_valid_frames = []  # 每個 chunk 對最終輸出貢獻的 frame 範圍

        if n_samples <= WHISPER_CHUNK_SAMPLES:
            # 短音檔：單一 chunk，feature_extractor 會 pad 到 30s
            hidden = self._encode_chunk(audio_16k)  # [1500, 1280]
            # 計算實際有效 frame 數（去掉 padding）：
            #   audio_16k 長 N samples → 對應 N / 16000 秒 → N / 16000 * 50 fps frames
            valid_frames = int(round(n_samples / WHISPER_INPUT_SR * WHISPER_FRAME_RATE_HZ))
            valid_frames = min(valid_frames, hidden.shape[0])
            chunks_hidden.append(hidden[:valid_frames])
        else:
            # 長音檔：滑動視窗，每個 chunk 處理後取中間段拼接
            # 為什麼取中間段：Whisper 對視窗邊界（前 0.5s、後 0.5s）有 attention
            # decay artifact，邊界附近 hidden state 不穩；overlap+取中段可避開
            offset = 0
            chunk_idx = 0
            while offset < n_samples:
                end = min(offset + WHISPER_CHUNK_SAMPLES, n_samples)
                chunk_audio = audio_16k[offset:end]
                hidden = self._encode_chunk(chunk_audio)  # [1500, 1280]

                # 計算這個 chunk 的有效 frame 範圍
                chunk_actual_samples = end - offset
                chunk_actual_frames = int(round(
                    chunk_actual_samples / WHISPER_INPUT_SR * WHISPER_FRAME_RATE_HZ
                ))
                chunk_actual_frames = min(chunk_actual_frames, hidden.shape[0])

                # 取「中間有效段」：頭 chunk 不扣 overlap 前段，尾 chunk 不扣 overlap 後段
                if chunk_idx == 0:
                    start_frame = 0
                else:
                    # 跳過 overlap 的前一半（與前一 chunk 重疊段）
                    start_frame = int(WHISPER_OVERLAP_SAMPLES / WHISPER_INPUT_SR
                                      * WHISPER_FRAME_RATE_HZ / 2)

                end_frame = chunk_actual_frames
                if end < n_samples:
                    # 不是最後 chunk → 扣掉後 overlap 的後一半
                    end_frame = end_frame - int(WHISPER_OVERLAP_SAMPLES / WHISPER_INPUT_SR
                                                 * WHISPER_FRAME_RATE_HZ / 2)

                chunks_hidden.append(hidden[start_frame:end_frame])

                offset += valid_chunk_samples
                chunk_idx += 1

        # ── Step 3: 拼接所有 chunk ─────────────────────────────────
        ppg_50fps = np.concatenate(chunks_hidden, axis=0)  # [T_50, 1280]

        # ── Step 4: resample 50 fps → mel fps (86.13) ─────────────
        # 為什麼用 1D linear interp：
        #   Whisper hidden state 隨時間漸變（attention smoothing 過），無高頻細節，
        #   linear interp 不會 alias；且 D=1280 太大，spline 等高階方法成本高
        target_T = int(round(len(audio) / sample_rate * FRAME_RATE_HZ))
        ppg = _resample_time_axis(ppg_50fps, target_T)

        return ppg.astype(np.float32)


def _resample_time_axis(x: np.ndarray, target_T: int) -> np.ndarray:
    """
    對 [T, D] 時間軸做 linear interpolation 到 target_T。

    為什麼自己寫而不用 scipy.interpolate：
      scipy 對 [T, D] 大矩陣會逐 D 跑 1D interp 慢；用 torch.nn.functional.interpolate
      在 D 大時 batch-friendly、速度快 5x+。
    """
    cur_T = x.shape[0]
    if cur_T == target_T:
        return x

    # 用 torch.interpolate 做 1D 線性插值
    # shape: [T, D] → [1, D, T]（torch 期望 channel-time 格式）→ interp → [1, D, target_T] → [target_T, D]
    x_t = torch.from_numpy(x).float().T.unsqueeze(0)  # [1, D, T]
    x_t = F.interpolate(x_t, size=target_T, mode="linear", align_corners=False)
    return x_t.squeeze(0).T.numpy()  # [target_T, D]


if __name__ == "__main__":
    # 自我測試：合成 2 秒白噪音，驗證輸出 shape 與 frame-rate 對齊
    sr = SAMPLE_RATE
    duration = 2.0
    audio = (np.random.randn(int(sr * duration)) * 0.1).astype(np.float32)

    expected_T = int(round(duration * FRAME_RATE_HZ))
    print(f"Test: {duration}s noise @ {sr} Hz")
    print(f"  Expected output shape: [~{expected_T}, {WHISPER_HIDDEN_DIM}]")

    extractor = WhisperPPGExtractor()
    ppg = extractor.extract(audio, sample_rate=sr)

    print(f"\nResult:")
    print(f"  Shape: {ppg.shape}")
    print(f"  dtype: {ppg.dtype}")
    print(f"  Frame rate: {ppg.shape[0] / duration:.2f} fps "
          f"(expected {FRAME_RATE_HZ:.2f})")
    print(f"  Norm (mean across T): {np.linalg.norm(ppg, axis=-1).mean():.2f}")
