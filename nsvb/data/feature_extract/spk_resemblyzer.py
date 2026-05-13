"""
nsvb/data/feature_extract/spk_resemblyzer.py
==============================================

【這支檔案做什麼】
從原始音檔抽取一個 256 維的 speaker embedding，每首歌一個固定向量。
回傳：
    spk_emb: np.float32 [256]，已 L2-normalized

【為什麼需要 speaker embedding】
NSVB-ZH 的 decoder θ 接收 spk_emb 作為「音色錨」（structural anchor for timbre）。
這在訓練/推理兩個層面都關鍵：

  1. 訓練端：Stage 1 CVAE 用 spk_emb 作為 decoder 條件之一，讓 z 可以盡量「不含
     音色」（speaker information 由 spk_emb 條件化出去），達到 z 在 amateur/pro
     之間分布重疊的目標（Monitor 2）。

  2. 推理端：Mode A/B/C 都餵 `spk_emb_a`（業餘歌手 embedding）→ decoder 輸出
     保留業餘歌手音色，即便 M 把 z 推向 pro-like。這是 Risk 4（音色外洩）的主要
     防線：即便 z 解耦不完美，spk_emb 條件會把 decoder 拉回正確音色。

【為什麼用 Resemblyzer 而非更新的選項】
Resemblyzer 是 GE2E (Wan et al. 2018) 的 PyTorch 移植，雖然 2019 年發布，但：
  - **輕量**：模型 ~17 MB，CPU 推理單首歌 ~0.5 秒
  - **API 極簡**：preprocess_wav + embed_utterance 兩行
  - **跨語言穩定**：在中文歌聲場景下測試表現足夠（speaker discrimination 不需要
    SOTA，只需要「同一歌手在不同歌中產生相近 embedding」這個 invariance）
  - 原 NSVB 也使用，重用降低相容性風險

未來若要升級的選項：
  - WavLM-base + x-vector head：跨語言 SOTA，但模型大 ~360 MB
  - ECAPA-TDNN (SpeechBrain)：說話人辨識 SOTA，但要多裝 SpeechBrain 套件
  - 由於 spk_emb 對訓練品質的邊際貢獻有限（主要靠 D_z + PatchNCE），
    現階段不值得為此換套件

【為什麼是「每首歌一個向量」而非 frame-level】
- Speaker identity 在一首歌內不變，沒必要 frame-level
- Resemblyzer 內部對音檔做 VAD 切片後對多個 partial 取平均，已自動處理
  「短靜音段不影響 embedding」的問題
- 單向量也方便儲存（每首歌只多 1 KB），訓練時直接 broadcast 到所有 frame

【為什麼要 L2-normalized】
- GE2E 訓練目標就是 cosine similarity，輸出向量本身就在單位球上
- 確認 normalize 的好處：下游若做 cosine 距離分析、t-SNE 視覺化（Monitor 2）
  不需要再做一次 normalize
- Resemblyzer.embed_utterance 預設已 L2-normalize，這支只是再驗證一次保險
"""

from typing import Optional

import numpy as np
import torch

from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    RESEMBLYZER_INPUT_SR,
    RESEMBLYZER_EMBED_DIM,
)


class SpeakerEmbeddingExtractor:
    """
    Resemblyzer VoiceEncoder 包裝器（lazy-loaded）。

    為什麼包成 class：
      與 WhisperPPGExtractor 一致設計；Resemblyzer 雖小但仍需要 ~1s 載入時間，
      Phase 0 對幾千首歌跑 batch 時，model 持續駐留 memory 比每次重建快很多。
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda' / 'cpu' / None（自動）

        為什麼預設可走 GPU：
          Resemblyzer 預設 CPU，但其內部 LSTM 在 GPU 上 ~3x 快；對幾千首歌的批次
          抽取有實質意義
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = None

    def _ensure_loaded(self):
        """第一次使用時載入 model。"""
        if self._encoder is not None:
            return
        from resemblyzer import VoiceEncoder
        # device 參數接受 torch device string
        self._encoder = VoiceEncoder(device=self.device, verbose=False)

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        主入口：從一段音訊抽 speaker embedding。

        Args:
            audio:        [N] float32，已正規化到 [-1, 1]
            sample_rate:  輸入取樣率；Resemblyzer 內部要 16 kHz，會自動 resample

        Returns:
            spk_emb: [256] float32，L2-normalized

        為什麼用 preprocess_wav：
          Resemblyzer 的 preprocess_wav 做了：
            1. resample 到 16 kHz
            2. trim silence (使用內建 VAD)
            3. RMS-based volume normalization
          這些前處理對 embedding 品質有實質影響，即便我們已經做了 loudness norm，
          仍建議走 Resemblyzer 的 pipeline 以匹配它訓練時的分布。

        為什麼 source_sr 用 SAMPLE_RATE 而非 audio.size 推回：
          preprocess_wav 需要明確 source_sr 才能正確 resample；推回會出錯。
        """
        self._ensure_loaded()

        from resemblyzer import preprocess_wav

        # preprocess_wav 接受 numpy + source_sr，回傳 16 kHz 的 wav
        wav_16k = preprocess_wav(audio, source_sr=sample_rate)

        # embed_utterance 內部把 wav 切成多個 partial（每 1.6s），
        # 對每個 partial 跑 LSTM encoder，最後對所有 partial 的 embedding 取平均
        # 並 L2-normalize → 回傳 [256] float32
        embed = self._encoder.embed_utterance(wav_16k)

        # 保險再做一次 normalize（防 Resemblyzer 未來版本改變預設行為）
        embed = embed / (np.linalg.norm(embed) + 1e-8)

        return embed.astype(np.float32)

    def extract_batch(
        self,
        audios: list,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        批次抽取（不真的並行，只是封裝循環）。

        為什麼不真正 batch GPU：
          Resemblyzer 內部對每首歌做 partial slicing 後是動態 sequence length，
          batch 起來要 padding 反而慢。Phase 0 用 multiprocessing 在 CPU 端
          並行更實際（每個 process 載一份 encoder）。

        Args:
            audios:      list of [N_i] np.ndarray
            sample_rate: 共用取樣率

        Returns:
            embeds: [B, 256] float32
        """
        return np.stack([self.extract(a, sample_rate) for a in audios])


if __name__ == "__main__":
    # 自我測試：兩段不同的合成訊號 → embedding 應該不同
    #          同一段切兩半 → embedding 應該很接近
    import time

    sr = SAMPLE_RATE
    duration = 3.0  # Resemblyzer 內部 partial 是 1.6s，至少要 ~2s 才有多 partial

    # 合成兩段不同特徵的訊號（不同基頻、不同諧波結構）
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio_a = (0.3 * np.sin(2 * np.pi * 200 * t)
               + 0.15 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    audio_b = (0.3 * np.sin(2 * np.pi * 350 * t)
               + 0.15 * np.sin(2 * np.pi * 700 * t)
               + 0.1 * np.sin(2 * np.pi * 1050 * t)).astype(np.float32)

    extractor = SpeakerEmbeddingExtractor()

    start = time.time()
    e_a = extractor.extract(audio_a, sample_rate=sr)
    e_a2 = extractor.extract(audio_a, sample_rate=sr)
    e_b = extractor.extract(audio_b, sample_rate=sr)
    elapsed = time.time() - start

    print(f"Embedding shape:  {e_a.shape}, dtype: {e_a.dtype}")
    print(f"Norm of e_a:      {np.linalg.norm(e_a):.4f}  (expect ~1.0)")
    print(f"cos(e_a, e_a2):   {np.dot(e_a, e_a2):.4f}  (expect 1.0, same input)")
    print(f"cos(e_a, e_b):    {np.dot(e_a, e_b):.4f}  (expect < 1.0, different)")
    print(f"3 extractions in {elapsed:.2f}s")
