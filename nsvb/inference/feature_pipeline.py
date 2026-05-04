"""
nsvb/inference/feature_pipeline.py
====================================

【這支檔案做什麼】
推理時對單一音檔做完整特徵抽取，回傳 model 期望的 tensor 字典：
    mel:        [T_mel, NUM_MELS]   float32
    f0:         [T_mel]              float32   Hz, unvoiced=0
    ppg:        [T_mel, ppg_dim]     float32
    spk_emb:    [spk_emb_dim]        float32   L2-normed
    phoneme_id: [T_mel]              int64     (若提供 centroids)
    register_soft: [T_mel, 5]        float32   soft Gaussian register

與 binarizer.py 的差異：
    - binarizer 對「整個資料集」批次 binarize，輸出 .npz 給訓練用
    - 此檔對「單一音檔」做即時抽取，回傳 in-memory tensor 給推理用
    - 兩者共用底層的 audio_io / f0_torchcrepe / ppg_whisper / spk_resemblyzer

【為什麼不直接呼叫 binarizer.py 的 main】
- binarizer 是 batch CLI 工具，包了 manifest 解析、CSV / .npz 寫檔等多餘邏輯
- 推理只需要 in-memory tensor，沒必要走檔案系統往返
- feature_pipeline 是純函式介面，方便 web demo / batch inference 複用

【為什麼默認跑 dereverb + loudness norm】
推理時 user 提供的音檔錄音環境千差萬別，跟訓練資料前處理對齊（Risk 2 主防線）：
    train (binarize): load → dereverb → loudness → mel
    infer  (此處)   : load → dereverb → loudness → mel
完全對齊；若使用者明確要關 dereverb（例如已是 studio 乾淨音檔）才在 CLI 設 --no-dereverb。

【為什麼 phoneme_id 是 optional】
推理路徑只用到 PPG（continuous, decoder 條件）；phoneme_id 是 D_z 的條件，
推理時不需要餵 D_z，所以可選不抽。保留入口是為了 debug / 視覺化用。

【F0 對齊到 mel frame】
torchcrepe 與 mel STFT 邊界處理會差 ±1 frame；統一用 mel.shape[0] 為基準長度，
F0 用 trim_or_pad_to_length 對齊（PPG 在 ppg_whisper 內部已做了 target_T align）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    HOP_SIZE,
    NUM_MELS,
)
from nsvb.utils.audio_io import (
    load_wav,
    dereverb_wav,
    loudness_normalize,
    compute_mel,
    pad_wav_to_mel_length,
)
from nsvb.utils.soft_bucket import soft_bucketize_f0
from nsvb.data.feature_extract.f0_torchcrepe import (
    extract_f0,
    trim_or_pad_to_length,
)
from nsvb.data.feature_extract.ppg_whisper import WhisperPPGExtractor
from nsvb.data.feature_extract.spk_resemblyzer import SpeakerEmbeddingExtractor


@dataclass
class InferenceFeatures:
    """單一音檔抽出的全部特徵（CPU tensor / numpy）。

    為什麼用 dataclass 而非 dict：
      type checker 可知道每個欄位形狀，未來新增欄位（例如 voicing）有 IDE 提示；
      與 dict access 比較也不容易拼錯欄位名。
    """
    wav: np.ndarray                    # [N_samples] float32 (已對齊到 T_mel * HOP_SIZE)
    mel: np.ndarray                    # [T_mel, NUM_MELS]
    f0: np.ndarray                     # [T_mel] Hz
    voicing: np.ndarray                # [T_mel] [0,1]
    ppg: np.ndarray                    # [T_mel, ppg_dim]
    spk_emb: np.ndarray                # [spk_emb_dim]
    register_soft: Optional[np.ndarray] = None   # [T_mel, 5]
    phoneme_id: Optional[np.ndarray] = None       # [T_mel] int64

    @property
    def t_mel(self) -> int:
        return self.mel.shape[0]


def _assign_phoneme_id(ppg: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    複製 cluster_ppg.assign_phoneme_id 的純 numpy distance 分群（避免拖整個 sklearn）。

    為什麼 inline 而非 import cluster_ppg.assign_phoneme_id：
      後者是 batch I/O 函式，吃 list of npz 路徑、會寫檔；單檔 inference 不需要那層
    """
    ppg32 = ppg.astype(np.float32)
    cen32 = centroids.astype(np.float32)
    ppg_norm = np.sum(ppg32 ** 2, axis=1, keepdims=True)
    cen_norm = np.sum(cen32 ** 2, axis=1)
    dot = ppg32 @ cen32.T
    dist = ppg_norm + cen_norm[None, :] - 2.0 * dot
    return np.argmin(dist, axis=1).astype(np.int64)


class InferenceFeatureExtractor:
    """
    推理用單檔特徵抽取器。內部維持 Whisper / Resemblyzer / (optional) k-means
    centroids 等重型資源在 memory 中，方便對多個音檔重用。

    為什麼包成 class：
      Whisper-large-v3 載入 ~10 秒、~3GB，每次抽特徵都重載入會浪費；
      把 model 持有在 instance 上，user 可實例化一次跑多次 extract()。
    """

    def __init__(
        self,
        device: Optional[str] = None,
        kmeans_centroids: Optional[np.ndarray] = None,
        whisper_model_name: Optional[str] = None,
    ):
        """
        Args:
            device:           'cuda' / 'cpu' / None (auto)
            kmeans_centroids: [K, 1280] PPG k-means centroids；提供則回傳 phoneme_id
            whisper_model_name: 覆寫 audio_config 的預設 whisper model
                                （smoke test 用 whisper-tiny 加速）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kmeans_centroids = kmeans_centroids

        # PPG / spk_emb 抽取器（lazy-loaded inside themselves）
        if whisper_model_name:
            self.ppg_extractor = WhisperPPGExtractor(
                model_name=whisper_model_name, device=self.device,
            )
        else:
            self.ppg_extractor = WhisperPPGExtractor(device=self.device)
        self.spk_extractor = SpeakerEmbeddingExtractor(device=self.device)

    def extract(
        self,
        wav_path: Path,
        dereverb: bool = True,
        loudness: bool = True,
    ) -> InferenceFeatures:
        """
        對單一 wav 檔做完整特徵抽取。

        Args:
            wav_path: 路徑
            dereverb: 是否跑 DeepFilterNet3 dereverb（與訓練端對齊，預設 True）
            loudness: 是否跑 ITU BS.1770 loudness norm（與訓練端對齊，預設 True）

        Returns:
            InferenceFeatures（全部 numpy / cpu）

        為什麼 dereverb / loudness 都 default True：
          訓練 binarize 走 load_and_extract（含兩者），推理保持一致避免分布偏移
          （Risk 2 主防線）。

        為什麼回傳 numpy 而非 torch tensor：
          特徵抽取 pipeline 大量用 numpy；最後在 model 入口統一 to_tensor 並 batch dim
          可以集中處理 device / dtype，更乾淨。
        """
        # 1. Load + dereverb + loudness + mel + 對齊
        wav = load_wav(str(wav_path))
        if dereverb:
            wav = dereverb_wav(wav)
        if loudness:
            wav = loudness_normalize(wav)
        mel = compute_mel(wav)
        wav = pad_wav_to_mel_length(wav, mel)
        T_mel = mel.shape[0]

        # 2. F0 + voicing
        f0, voicing = extract_f0(wav, sample_rate=SAMPLE_RATE, device=self.device)
        f0 = trim_or_pad_to_length(f0, T_mel)
        voicing = trim_or_pad_to_length(voicing, T_mel)

        # 3. PPG（內部已 align 到 mel fps；保險再 trim/pad 一次）
        ppg = self.ppg_extractor.extract(wav, sample_rate=SAMPLE_RATE)
        ppg = trim_or_pad_to_length(ppg, T_mel)

        # 4. Speaker embedding
        spk_emb = self.spk_extractor.extract(wav, sample_rate=SAMPLE_RATE)

        # 5. Soft register（CPU torch → numpy；register 對 D_z 才用，但保留供 debug）
        register_soft = soft_bucketize_f0(
            torch.from_numpy(f0).unsqueeze(0)
        ).squeeze(0).numpy().astype(np.float32)

        # 6. Phoneme ID（optional, 若提供 centroids）
        phoneme_id = None
        if self.kmeans_centroids is not None:
            phoneme_id = _assign_phoneme_id(ppg, self.kmeans_centroids)

        return InferenceFeatures(
            wav=wav, mel=mel, f0=f0, voicing=voicing,
            ppg=ppg, spk_emb=spk_emb,
            register_soft=register_soft, phoneme_id=phoneme_id,
        )


def features_to_batch(
    features: InferenceFeatures,
    device: torch.device,
) -> dict:
    """
    把 InferenceFeatures（numpy, 無 batch dim）轉成 model 期望的 batch=1 dict。

    為什麼集中在這函式：
      推理永遠 batch=1（serving 場景），to_device + unsqueeze 邏輯重複多處易出錯；
      集中讓 user 與測試只需呼叫一次。

    Returns:
        dict with keys (all on device, batched):
            mel:      [1, T_mel, NUM_MELS]
            f0:       [1, T_mel]
            ppg:      [1, T_mel, ppg_dim]
            spk_emb:  [1, spk_emb_dim]
            mel_mask: [1, T_mel]   全 1（推理單檔無 padding）
    """
    mel = torch.from_numpy(features.mel).float().unsqueeze(0).to(device)
    f0 = torch.from_numpy(features.f0).float().unsqueeze(0).to(device)
    ppg = torch.from_numpy(features.ppg).float().unsqueeze(0).to(device)
    spk_emb = torch.from_numpy(features.spk_emb).float().unsqueeze(0).to(device)
    # 推理單檔無 padding；全 1 mask
    mel_mask = torch.ones(1, features.t_mel, device=device)
    return {
        "mel": mel, "f0": f0, "ppg": ppg,
        "spk_emb": spk_emb, "mel_mask": mel_mask,
    }