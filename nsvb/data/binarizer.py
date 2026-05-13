"""
nsvb/data/binarizer.py
========================

【這支檔案做什麼】
Phase 0 的核心：把原始音檔（VocalVerse 業餘 + M4Singer 職業）轉成訓練可用的
特徵 dict，逐首存盤。每首歌輸出一個 .npz 檔，包含：

    wav            [N_samples]            float32   響度正規化後的波形（HifiGAN 需要）
    mel            [T_mel, 80]            float32   log10-mel
    f0             [T_mel]                float32   Hz, unvoiced=0
    voicing        [T_mel]                float32   CREPE 信心度 [0,1]
    register_soft  [T_mel, 5]             float32   D_z 的軟條件
    register_id    [T_mel]                int8      硬 bucket id（JSD 統計用）, -1=unvoiced
    ppg            [T_mel, 1280]          float16   Whisper hidden state（fp16 省 50% 空間）
    spk_emb        [256]                  float32   speaker embedding
    metadata       dict                            dataset, speaker_id, item_id, sr, hop

【為什麼一首歌一個 .npz】
- **可檢查性**：訓練前可以直接用 `np.load(path).files` 看任意一首歌的特徵
- **可平行化**：Phase 0 可用 multiprocessing 對檔案 list 分派，每個 process 各自寫
- **可斷點續跑**：跑到一半中斷，下次 skip 已存在的 .npz 就好；不需要 transactional rebuild
- **NSVB 原版 IndexedDataset 太複雜**：原版用 .data + .idx 二進位 packed 格式，
  讀取方便但寫入要計算 offset、出錯難 debug；對 ~10k 首歌的規模沒必要

【為什麼 PPG 存 fp16】
- PPG 是這個 dataset 最大宗特徵：[T_mel, 1280] × 5 分鐘歌曲 ≈ 25800 × 1280 × 4 byte
  = 132 MB / song（fp32），10000 首歌 = 1.3 TB。fp16 砍半到 660 GB，仍大但可接受
- Whisper hidden state 數值範圍小（typically [-10, 10]），fp16 精度損失 < 0.1%
- 訓練時讀進來自動上轉 fp32（torch.from_numpy(x.astype(np.float32))）

【為什麼這個 binarizer 不處理 phoneme_id】
- phoneme_id 來自對 PPG 做 k-means 分群，需要先看過所有 utt 的 PPG 才能 fit centroids
- 這是 Phase 0 第二階段（cluster_ppg.py，後續實作）：先全部抽完 PPG，再跑分群，
  分群結果寫回每個 .npz（增加 phoneme_id key）
- 這支只負責「單首歌的全特徵抽取」，職責清晰

【為什麼這支用 sequential 而非 multiprocessing】
- GPU 模型（Whisper、Resemblyzer）在 multiprocessing 下要每個 process 各載一份，
  GPU memory 一份 5 GB×N，受不了
- 真正的並行化策略：
    Tier 1（CPU-bound）：mel、loudness norm、librosa 載入 → multiprocessing
    Tier 2（GPU-bound）：F0 (torchcrepe)、PPG (Whisper)、spk_emb (Resemblyzer) → sequential
- 這支先做 Tier 2 sequential 版，能跑通 Phase 0 即可；速度優化可在後期再做
- 估計 sequential 速度：每首歌約 3-5 秒（GPU），10000 首歌 = 8~14 小時，可接受
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Phase 0 輸入特徵抽取器
from nsvb.utils.audio_io import load_and_extract
from nsvb.utils.audio_config import (
    SAMPLE_RATE,
    HOP_SIZE,
    NUM_MELS,
    WHISPER_MODEL_NAME,
    RESEMBLYZER_EMBED_DIM,
)
from nsvb.utils.soft_bucket import (
    soft_bucketize_f0,
    hz_to_register_id,
    NUM_BUCKETS,
)
from nsvb.data.feature_extract.f0_torchcrepe import (
    extract_f0,
    trim_or_pad_to_length,
)
from nsvb.data.feature_extract.ppg_whisper import WhisperPPGExtractor
from nsvb.data.feature_extract.spk_resemblyzer import SpeakerEmbeddingExtractor
# Optional：vocalverse 多維過濾用；module-level import 讓 type hint 可用
from nsvb.data.vocalverse_mos import FilterCriteria

import torch


# ── Dataset 列舉 ─────────────────────────────────────────
# 每個 sample 用三元組 (wav_path, speaker_id, item_id) 唯一識別
# - wav_path: 絕對路徑
# - speaker_id: 用於分割 train/val（同 speaker 不應跨 split），
#               以及 spk_emb 的 group-by 統計
# - item_id: dataset-internal 唯一 ID，用於檔名與 metadata


@dataclass
class SampleSpec:
    wav_path: str
    speaker_id: str
    item_id: str
    dataset: str  # "m4singer" or "vocalverse"


def list_m4singer(root: Path) -> List[SampleSpec]:
    """
    M4Singer 結構：data/m4singer/{歌手}#{歌名}/{idx}.wav
    例：data/m4singer/Alto-1#newboy/0000.wav

    為什麼這樣處理：
      M4Singer 每首歌被切成多個 phrase（0000.wav, 0001.wav...），每個 phrase
      就是一個訓練樣本。speaker_id 從 `{歌手}#...` 取 `#` 之前的部分。
      song_id（用於分割 train/val）從 `#` 之後取，避免同一首歌的不同 phrase
      跨 train/val 造成資訊洩漏。
    """
    samples = []
    for song_dir in sorted(root.iterdir()):
        if not song_dir.is_dir() or "#" not in song_dir.name:
            continue
        speaker, song = song_dir.name.split("#", 1)
        for wav in sorted(song_dir.glob("*.wav")):
            item_id = f"{song_dir.name}__{wav.stem}"  # 「Alto-1#newboy__0000」
            samples.append(SampleSpec(
                wav_path=str(wav.resolve()),
                speaker_id=speaker,
                item_id=item_id,
                dataset="m4singer",
            ))
    return samples


def list_vocalverse(root: Path) -> List[SampleSpec]:
    """
    VocalVerse 結構：data/VocalVerse/{user_id}/{wav_id}.wav

    為什麼用 user_id 當 speaker_id：
      VocalVerse 每個 user 是一個業餘歌手；同個 user 多首歌共用音色 / 技術水準，
      該作為一個 speaker group。
    """
    samples = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        for wav in sorted(user_dir.glob("*.wav")):
            item_id = f"{user_id}__{wav.stem}"
            samples.append(SampleSpec(
                wav_path=str(wav.resolve()),
                speaker_id=user_id,
                item_id=item_id,
                dataset="vocalverse",
            ))
    return samples


# ── 單首歌的特徵抽取 ─────────────────────────────────────
class FeatureExtractors:
    """
    把三個 GPU-bound 抽取器（torchcrepe 用的隱含模型 / Whisper / Resemblyzer）
    包成一個容器，避免 binarize_one 每次重新建立。

    為什麼包成 class 而非全域單例：
      多 process 並行時，每 process 一個 instance 比較乾淨；單例會被誤共享。
      雖然目前是 sequential，但保留 multi-proc 擴充性。
    """

    def __init__(
        self,
        device: Optional[str] = None,
        whisper_model_name: str = WHISPER_MODEL_NAME,
    ):
        """
        Args:
            device:             'cuda' / 'cpu' / None（None = 自動偵測）
            whisper_model_name: HF model id；本地 smoke test 可換成
                                'openai/whisper-tiny' 之類的小模型快速驗證 pipeline
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[binarize] device={self.device}, whisper_model={whisper_model_name}",
              flush=True)
        self.whisper = WhisperPPGExtractor(
            model_name=whisper_model_name,
            device=self.device,
        )
        self.spk = SpeakerEmbeddingExtractor(device=self.device)
        # F0 用 module-level function（torchcrepe 自己 cache model）

    def extract_all(self, wav: np.ndarray, mel: np.ndarray) -> dict:
        """
        對單一已載入的 (wav, mel) 抽全部特徵。

        為什麼接收已抽好的 mel：
          mel 在 audio_io.load_and_extract 已經算過，重算浪費 ~50ms × 10k 首
          = 8 分鐘。傳進來重用。

        Returns:
            dict 含所有特徵（numpy arrays）
        """
        T_mel = mel.shape[0]

        # 1. F0（torchcrepe）→ 對齊到 mel frame 數
        # device 沿用上層設定；torchcrepe 內部會在第一次呼叫時 cache model 到指定 device
        f0, voicing = extract_f0(wav, sample_rate=SAMPLE_RATE, device=self.device)
        f0 = trim_or_pad_to_length(f0, T_mel, pad_value=0.0)
        voicing = trim_or_pad_to_length(voicing, T_mel, pad_value=0.0)

        # 2. Soft register bucket（從 F0 推）
        # 為什麼這裡算而不延後到 dataset 載入時：
        #   register 計算便宜，但每 epoch 都重算的話累積成本可觀；
        #   存進 .npz 是「空間換時間」的合理選擇
        f0_t = torch.from_numpy(f0)
        register_soft = soft_bucketize_f0(f0_t).numpy().astype(np.float32)  # [T_mel, 5]
        register_id = hz_to_register_id(f0_t).numpy().astype(np.int8)  # [T_mel] int8 省空間

        # 3. PPG（Whisper hidden state）
        ppg = self.whisper.extract(wav, sample_rate=SAMPLE_RATE)  # [T_ppg, 1280]
        ppg = trim_or_pad_to_length(ppg, T_mel, pad_value=0.0)
        # fp16 儲存（節省 50% 空間）
        ppg_fp16 = ppg.astype(np.float16)

        # 4. Speaker embedding
        spk_emb = self.spk.extract(wav, sample_rate=SAMPLE_RATE)  # [256]

        return {
            "f0": f0,
            "voicing": voicing,
            "register_soft": register_soft,
            "register_id": register_id,
            "ppg": ppg_fp16,
            "spk_emb": spk_emb,
        }


def binarize_one(spec: SampleSpec, extractors: FeatureExtractors,
                 dereverb: bool = True) -> dict:
    """
    對一首歌做完整的特徵抽取，回傳可直接 np.savez 的 dict。

    Args:
        dereverb: 預設 True（production 必開, Risk 2 主防線）；smoke test 可關

    Returns:
        sample dict，schema 見檔頭文件
    """
    # 1. 載入 + (optional) dereverb + loudness norm + mel
    wav, mel = load_and_extract(spec.wav_path, dereverb=dereverb)

    # 2. 各種 feature
    feats = extractors.extract_all(wav, mel)

    # 3. 包成 dict（metadata 用 numpy 0-d array 也能被 np.savez 接受）
    sample = {
        "wav": wav.astype(np.float32),
        "mel": mel.astype(np.float32),
        **feats,
        # metadata：np.savez 需要 array-like 值，str 包成 0-d array
        "meta_dataset": np.array(spec.dataset),
        "meta_speaker_id": np.array(spec.speaker_id),
        "meta_item_id": np.array(spec.item_id),
        "meta_sample_rate": np.array(SAMPLE_RATE, dtype=np.int32),
        "meta_hop_size": np.array(HOP_SIZE, dtype=np.int32),
        "meta_dereverbed": np.array(int(dereverb), dtype=np.int32),
    }
    return sample


def save_sample(sample: dict, out_path: Path):
    """
    存成壓縮 .npz。

    為什麼用 savez_compressed 而非 savez：
      ppg 是大 array，但 Whisper hidden state 數值有低秩性（attention 後常含
      重複模式），壓縮率 ~30-50%；wav 也能壓 ~10%。
      代價是寫入慢 ~3x，但 Phase 0 一次性，不影響訓練。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **sample)


def chunk_sample(
    sample: dict,
    chunk_sec: float,
    min_remaining_sec: float = 3.0,
) -> List[dict]:
    """
    把整首歌的 feature dict 切成 N 個 chunk_sec 秒的 chunks。

    為什麼需要切：
      VocalVerse 每首約 200 秒，遠超 NSVB 訓練設計的 phrase 級單位（5-10s）。
      若直接以整首為 sample：
        - 訓練 dataset 每 epoch 看不完 200s 中的大部分內容（random crop 只取 ~5s）
        - 載入單一 .npz 約 120 MB 但訓練只用到 ~5s ≈ 95% IO 浪費
        - 整體 M4 21K samples vs VV 536 samples 1:39 比例失衡
      切成 5-sec chunks 後 VV ≈ 21K，與 M4 平衡，與 NSVB「每樣本=phrase」哲學一致。

    為什麼最後一個短 chunk 丟掉：
      < min_remaining_sec 的 chunk 訓練時可能：
        - 不足以涵蓋一個 phrase
        - max_frames padding 過多影響 batch 統計
        - 邊界 silence 比例太高
      丟掉只損失最多 chunk_sec - 1 秒 / 首歌 ≈ 0.5% 資料，影響可忽略。

    為什麼 spk_emb 重複放入每 chunk：
      spk_emb 是每首歌共用的「歌手音色錨」（Resemblyzer 對整首歌算的 256-dim 向量），
      切 chunk 後同首所有 chunks 仍是同一歌手；每 chunk 各自存一份 spk_emb 才能
      獨立載入。重複的成本只 1 KB × N chunks，可忽略。

    Args:
        sample:            binarize_one 的輸出
        chunk_sec:         每個 chunk 的目標長度（秒）
        min_remaining_sec: 短於此值的尾段直接丟棄

    Returns:
        list of chunk dicts；每個含與 sample 相同 schema，但 wav / mel / f0 /
        voicing / register_soft / register_id / ppg / phoneme_id 都被切片，
        metadata 中 item_id 加 `__c{idx:03d}` 後綴。
    """
    chunk_frames = int(chunk_sec * SAMPLE_RATE / HOP_SIZE)   # ~860 for 5s
    chunk_samples = int(chunk_sec * SAMPLE_RATE)              # ~110250 for 5s
    min_frames = int(min_remaining_sec * SAMPLE_RATE / HOP_SIZE)

    T_mel = sample["mel"].shape[0]
    N_samples = sample["wav"].shape[0]
    base_item_id = str(sample["meta_item_id"])

    # frame-rate 同步的 keys（沿時間軸切）
    framewise_keys = ["mel", "f0", "voicing", "register_soft", "register_id", "ppg"]
    # wav 是 sample-rate（時間軸更密）
    # spk_emb / metadata 不切（per-song scalar / vector）

    chunks: List[dict] = []
    for chunk_idx, start_f in enumerate(range(0, T_mel, chunk_frames)):
        end_f = min(start_f + chunk_frames, T_mel)
        if (end_f - start_f) < min_frames:
            # 尾段過短，丟棄
            break

        start_s = start_f * HOP_SIZE
        end_s = min(end_f * HOP_SIZE, N_samples)

        chunk = {k: sample[k][start_f:end_f] for k in framewise_keys}
        chunk["wav"] = sample["wav"][start_s:end_s]
        # spk_emb 整首共用，每 chunk 都存一份相同的
        chunk["spk_emb"] = sample["spk_emb"]
        # metadata：保留歌手 / dataset 不變，但 item_id 加 chunk 後綴
        chunk["meta_dataset"] = sample["meta_dataset"]
        chunk["meta_speaker_id"] = sample["meta_speaker_id"]
        chunk["meta_item_id"] = np.array(f"{base_item_id}__c{chunk_idx:03d}")
        chunk["meta_sample_rate"] = sample["meta_sample_rate"]
        chunk["meta_hop_size"] = sample["meta_hop_size"]
        chunk["meta_dereverbed"] = sample["meta_dereverbed"]
        chunks.append(chunk)

    return chunks


# ── 主流程 ──────────────────────────────────────────────
def run_binarize(
    dataset_root: Path,
    out_root: Path,
    dataset_name: str,
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    whisper_model_name: str = WHISPER_MODEL_NAME,
    dereverb: bool = True,
    vocalverse_filter: Optional[FilterCriteria] = None,
    vocalverse_label_dir: Optional[Path] = None,
    vocalverse_chunk_sec: Optional[float] = None,
):
    """
    對一個 dataset 跑完整 binarize。

    Args:
        dataset_root:  data/m4singer/  或  data/VocalVerse/
        out_root:      輸出根目錄（會在底下開 dataset_name/）
        dataset_name:  "m4singer" or "vocalverse"
        skip_existing: 已有 .npz 的 sample 跳過（斷點續跑）
        max_samples:   debug 用，只跑前 N 首
        dereverb:      是否做 DeepFilterNet3 去殘響+去噪（Risk 2 主防線，production 必開）
        vocalverse_filter:
            僅對 vocalverse 有效。多維過濾條件（amateur_score / pro 4-dim / amateur MOS），
            所有 set 的 max 用 AND 組合。None 或 inactive 則不過濾。
            推薦：FilterCriteria(amateur_score_max=3.0)（與 M4Singer 30h 對齊）。
        vocalverse_label_dir:
            VocalVerse_Datasets-human_labels/ 目錄路徑；
            None 時自動從 dataset_root/VocalVerse_Datasets-human_labels/ 找。
    """
    print(f"[binarize] Listing samples in {dataset_root}...", flush=True)
    if dataset_name == "m4singer":
        samples = list_m4singer(dataset_root)
    elif dataset_name == "vocalverse":
        samples = list_vocalverse(dataset_root)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    print(f"[binarize] Found {len(samples)} samples", flush=True)

    # 多維過濾（僅 vocalverse）
    if (dataset_name == "vocalverse"
            and vocalverse_filter is not None
            and vocalverse_filter.is_active()):
        from nsvb.data.vocalverse_mos import (
            find_label_dir, load_vocalverse_labels,
            filter_samples, format_filter_stats,
        )
        label_dir = vocalverse_label_dir or find_label_dir(dataset_root)
        if label_dir is None or not label_dir.is_dir():
            raise FileNotFoundError(
                f"VocalVerse label dir not found "
                f"(expected at {dataset_root}/VocalVerse_Datasets-human_labels/); "
                f"請手動傳 --vocalverse-label-dir 或關掉 vocalverse_filter"
            )
        labels_map = load_vocalverse_labels(label_dir, require_pro=False)
        print(f"[binarize] Loaded {len(labels_map)} label records from {label_dir}", flush=True)
        samples, stats = filter_samples(
            samples, labels_map, criteria=vocalverse_filter, on_missing_record="drop",
        )
        print(format_filter_stats(stats), flush=True)

    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"[binarize] Truncated to {len(samples)} (debug)", flush=True)

    # 載入抽取器（這步會把 Whisper / Resemblyzer 載入指定 device）
    print(f"[binarize] Loading feature extractors...", flush=True)
    extractors = FeatureExtractors(
        device=device,
        whisper_model_name=whisper_model_name,
    )

    out_dir = out_root / dataset_name
    n_done = n_skip = n_err = 0
    t_start = time.time()

    # 是否切 chunk（僅 VV 用）
    do_chunk = (dataset_name == "vocalverse" and vocalverse_chunk_sec is not None)
    if do_chunk:
        print(f"[binarize] VocalVerse chunking ON: each song → {vocalverse_chunk_sec}s chunks",
              flush=True)

    for i, spec in enumerate(samples):
        # 為什麼用 __c000.npz 當「整首已處理過」的代表：
        #   chunk 順序寫，c000 存在表示 binarize_one 跑成功且 chunking 啟動過；
        #   檢查所有 chunks 太貴（chunk 數依歌長變），c000 已是充分指標
        if do_chunk:
            probe_path = out_dir / f"{spec.item_id}__c000.npz"
        else:
            probe_path = out_dir / f"{spec.item_id}.npz"

        if skip_existing and probe_path.exists():
            n_skip += 1
            continue

        try:
            sample = binarize_one(spec, extractors, dereverb=dereverb)
            if do_chunk:
                chunks = chunk_sample(sample, vocalverse_chunk_sec)
                if not chunks:
                    print(f"[binarize] {spec.item_id}: song too short for any chunk, skipping")
                    n_err += 1
                    continue
                for ch_idx, ch in enumerate(chunks):
                    ch_path = out_dir / f"{spec.item_id}__c{ch_idx:03d}.npz"
                    save_sample(ch, ch_path)
                n_done += 1
            else:
                save_sample(sample, probe_path)
                n_done += 1
        except Exception as e:
            print(f"[binarize] ERROR on {spec.item_id}: {e}")
            n_err += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_min = (len(samples) - i - 1) / rate / 60
            print(f"[binarize] {i+1}/{len(samples)}  "
                  f"done={n_done} skip={n_skip} err={n_err}  "
                  f"rate={rate:.2f}/s  ETA={eta_min:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n[binarize] Done in {elapsed/60:.1f} min")
    print(f"[binarize] Total: {len(samples)}  done={n_done} skip={n_skip} err={n_err}")


def main():
    """
    CLI entrypoint。

    用法：
        python -m nsvb.data.binarizer --dataset m4singer
        python -m nsvb.data.binarizer --dataset vocalverse --max-samples 5  # debug
    """
    parser = argparse.ArgumentParser(description="NSVB-ZH Phase 0 binarizer")
    parser.add_argument(
        "--dataset", choices=["m4singer", "vocalverse"], required=True,
        help="Which dataset to process",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root containing m4singer/ and VocalVerse/",
    )
    parser.add_argument(
        "--out-root", default="data/binarized",
        help="Output dir; samples go to {out_root}/{dataset}/{item_id}.npz",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Debug: only process first N samples",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process even if output .npz exists",
    )
    parser.add_argument(
        "--device", default=None, choices=[None, "cpu", "cuda"],
        help="強制使用 cpu / cuda；預設 auto（cuda available 則 cuda）。"
             "本地 smoke test 顯卡記憶體吃緊時用 cpu。",
    )
    parser.add_argument(
        "--whisper-model", default=WHISPER_MODEL_NAME,
        help="Whisper HF model id；預設用 audio_config 的 whisper-large-v3（production），"
             "本地 smoke test 可換 'openai/whisper-tiny' 約 150MB、CPU 跑得動",
    )
    parser.add_argument(
        "--no-dereverb", action="store_true",
        help="關閉 DeepFilterNet3 去殘響+去噪預處理。"
             "預設啟用（Risk 2 主防線；production 必開）；smoke test / 對照實驗才關",
    )
    parser.add_argument(
        "--vocalverse-amateur-score-max", type=float, default=None,
        help="（僅 --dataset vocalverse）⭐ 主要過濾條件：依 (技巧+氣息控制)/2 平均分過濾，"
             "保留 amateur_score <= max 的「真業餘」樣本。"
             "推薦 3.0（與 M4Singer 30h 對齊；536 筆/29.8 h，per-singer 中位 17）。"
             "閾值對應筆數：2.0=202, 2.5=371, 3.0=536, 3.5=676",
    )
    parser.add_argument(
        "--vocalverse-technique-max", type=float, default=None,
        help="（僅 --dataset vocalverse）pro 「技巧」維度單獨 cap（與其他 max AND 組合）",
    )
    parser.add_argument(
        "--vocalverse-breath-max", type=float, default=None,
        help="（僅 --dataset vocalverse）pro 「氣息控制」維度單獨 cap",
    )
    parser.add_argument(
        "--vocalverse-emotion-max", type=float, default=None,
        help="（僅 --dataset vocalverse）pro 「情感」維度單獨 cap（次要 signal）",
    )
    parser.add_argument(
        "--vocalverse-timbre-max", type=float, default=None,
        help="（僅 --dataset vocalverse）pro 「音色」維度單獨 cap。"
             "**不建議使用**：音色屬 physiological 特性、由 spk_emb 鎖；過濾音色等於"
             "拒絕「音色不夠 pop」的業餘者，違反 NSVB-ZH 設計",
    )
    parser.add_argument(
        "--vocalverse-mos-max", type=float, default=None,
        help="（僅 --dataset vocalverse）amateur MOS 單獨 cap（次要 corroborator）。"
             "建議與 --vocalverse-amateur-score-max 組合用，例如 amateur_score≤3.0 + mos≤3.5",
    )
    parser.add_argument(
        "--vocalverse-label-dir", default=None,
        help="VocalVerse label 目錄路徑覆寫。預設自動找 "
             "{vocalverse_root}/VocalVerse_Datasets-human_labels/",
    )
    parser.add_argument(
        "--vocalverse-chunk-sec", type=float, default=None,
        help="（僅 --dataset vocalverse）⭐ 把每首歌切成 N 個 chunk_sec 秒 chunks，"
             "解決 VV 長 recording vs M4 短 snippet 的失衡（M4 21K snippets vs "
             "VV 536 songs ≈ 39:1）。推薦 5.0（VV 切完 ≈ 21K chunks 與 M4 對等）。"
             "None=不切（VV 保持整首 200s 一個 .npz；訓練 random crop 會浪費 IO）",
    )
    # 向後相容：舊參數名 → 對應到 mos-max
    parser.add_argument(
        "--vocalverse-mos-threshold", type=float, default=None,
        help="(deprecated) 與 --vocalverse-mos-max 等價；舊腳本相容用",
    )
    args = parser.parse_args()

    # 對應 data 目錄的實際資料夾名（M4Singer 大小寫敏感）
    dataset_dir_name = {
        "m4singer": "m4singer",
        "vocalverse": "VocalVerse",
    }[args.dataset]

    dataset_root = Path(args.data_root) / dataset_dir_name
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {dataset_root}")
        sys.exit(1)

    out_root = Path(args.out_root)

    # 構造 FilterCriteria；舊 --vocalverse-mos-threshold 與新 --vocalverse-mos-max 取較嚴者
    from nsvb.data.vocalverse_mos import FilterCriteria
    legacy_mos = args.vocalverse_mos_threshold
    if legacy_mos is not None and args.vocalverse_mos_max is None:
        print("[binarize] WARNING: --vocalverse-mos-threshold is deprecated; "
              "use --vocalverse-mos-max", flush=True)
    effective_mos = args.vocalverse_mos_max if args.vocalverse_mos_max is not None else legacy_mos

    vocalverse_filter = FilterCriteria(
        amateur_score_max=args.vocalverse_amateur_score_max,
        technique_max=args.vocalverse_technique_max,
        breath_max=args.vocalverse_breath_max,
        emotion_max=args.vocalverse_emotion_max,
        timbre_max=args.vocalverse_timbre_max,
        mos_max=effective_mos,
    )

    run_binarize(
        dataset_root=dataset_root,
        out_root=out_root,
        dataset_name=args.dataset,
        skip_existing=not args.no_skip,
        max_samples=args.max_samples,
        device=args.device,
        whisper_model_name=args.whisper_model,
        dereverb=not args.no_dereverb,
        vocalverse_filter=vocalverse_filter,
        vocalverse_label_dir=Path(args.vocalverse_label_dir) if args.vocalverse_label_dir else None,
        vocalverse_chunk_sec=args.vocalverse_chunk_sec,
    )


if __name__ == "__main__":
    main()
