"""
nsvb/data/dataset.py
======================

【這支檔案做什麼】
torch Dataset wrapper around binarizer.py 預存的 .npz files。
讀進 mel / F0 / voicing / register / PPG / spk_emb / phoneme_id，提供：
    - BinarizedNSVBDataset           單一 dataset (m4singer 或 vocalverse)
    - JointAmateurProDataset          Stage 2 用，獨立 amateur / pro 抽樣 dataloader
    - collate_fn                      把 list of samples 打包成 batch + 處理變長 padding

【為什麼讀 .npz 而不用 IndexedDataset】
- NSVB 原版用 .data + .idx 二進位 packed（讀快、寫複雜、出錯難 debug）
- 我們用一首歌一個 .npz：開發期可以 `np.load(file).files` 直接看內容，
  訓練期 SSD 隨機讀取速度也 OK（PyTorch DataLoader num_workers 多進程平行）
- 缺點：~10000 個小檔對檔案系統有 inode 壓力；若以後成 bottleneck 再 pack

【樣本切片策略】
- max_frames=600 切片（mel 172 fps × ~3.5 sec），dataloader 隨機取 [0, T - max_frames] 起點
- 為什麼隨機切：避免每 epoch 都看歌曲開頭，加強模型泛化
- 為什麼 600：3.5 sec ≈ 一個短句 phrase，足夠捕捉 vibrato / 共鳴細節，但不超 GPU memory
- short clip 不切：若 T <= max_frames 直接全用，不 pad（pad 在 collate 階段做）

【為什麼 PPG 在 dataset 上轉 fp32】
- 存 .npz 是 fp16（省磁碟），但訓練要 fp32（pytorch op 多數 fp32 較穩）
- 在 __getitem__ 上轉，dataloader worker 用 CPU 處理；GPU forward 不用再轉

【phoneme_id 為何用 .get() 寬鬆模式】
- 第一次 binarize 完還沒跑 cluster_ppg，.npz 裡沒有 phoneme_id
- 此時 dataset 回傳 phoneme_id = -1 全填，Stage 1 不依賴 phoneme_id 仍能訓
- Stage 2 訓 D_z 才依賴 phoneme_id，那時已經跑過 cluster_ppg
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BinarizedNSVBDataset(Dataset):
    """
    讀單一 dataset (m4singer 或 vocalverse) 的 .npz 檔。

    Args:
        binarized_root: 含 .npz 的目錄
        max_frames:     單筆樣本切片長度（mel frame 數）
        seed:           shuffle seed（每 epoch 再 reseed 由 sampler 控制）
    """

    # 為什麼讓 phoneme_id 預設 -1 sentinel：
    # binarize 第一輪沒 phoneme_id，讓 dataset 仍能載入；
    # Stage 2 task 在每 batch 檢查 phoneme_id != -1 才啟用 D_z
    PHONEME_ID_PADDING = -1

    def __init__(
        self,
        binarized_root: Path,
        max_frames: int = 600,
        min_frames: int = 32,
        split_file: Optional[Path] = None,
    ):
        """
        Args:
            split_file:
                若指定，從該檔讀 item_id list（一行一個，無 .npz 副檔名），
                只保留 list 內的 .npz。檔內可混合多 dataset 的 item_id；
                本 dataset 只匹配自己目錄下符合的（其他 dataset 的 item_id 自然不會被 glob 到）。
                See scripts/make_splits.py。
        """
        self.root = Path(binarized_root)
        self.max_frames = max_frames
        self.min_frames = min_frames

        # 為什麼用 sorted：跨平台 / 跨機器 dataset index 順序一致
        candidate_paths = sorted(self.root.glob("*.npz"))
        if not candidate_paths:
            raise FileNotFoundError(f"No .npz under {self.root}")

        # 為什麼用 set lookup 而非 list:~21K 檔 × ~21K split entries 的 O(N²) 會慢爆;
        # set 是 O(1) per check
        if split_file is not None:
            split_path = Path(split_file)
            with open(split_path) as f:
                allowed = {line.strip() for line in f if line.strip()}
            candidate_paths = [p for p in candidate_paths if p.stem in allowed]
            if not candidate_paths:
                raise FileNotFoundError(
                    f"No .npz under {self.root} matched split file {split_path}; "
                    f"確認 split 檔內含本 dataset 的 item_id"
                )
            print(f"[dataset] {self.root.name}: filtered to "
                  f"{len(candidate_paths)} samples via {split_path.name}")
        self.npz_paths = candidate_paths

        # 預讀每個樣本的 mel 長度（過濾過短的，給 sampler 用）
        # 為什麼預讀：DataLoader 啟動時就過濾，後續每個 batch 都不會踩到太短樣本
        # 開銷：~10000 首歌 × np.load(allow_pickle=True) ~= 30 sec，一次性
        self._lengths = []
        valid_paths = []
        for p in self.npz_paths:
            try:
                with np.load(p, allow_pickle=True) as data:
                    n = int(data["mel"].shape[0])
                if n >= min_frames:
                    self._lengths.append(n)
                    valid_paths.append(p)
            except (KeyError, OSError, ValueError):
                # 損壞 / 缺 mel key 的 .npz 跳過
                continue
        self.npz_paths = valid_paths

        print(f"[dataset] {self.root.name}: {len(self.npz_paths)} valid samples "
              f"(mean T_mel={np.mean(self._lengths):.0f}, "
              f"max={max(self._lengths)}, min={min(self._lengths)})")

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            sample dict containing
                mel:           [T, NUM_MELS]    float32
                f0:            [T]              float32 Hz, unvoiced=0
                voicing:       [T]              float32 [0,1]
                register_soft: [T, 5]           float32
                register_id:   [T]              int64 (-1=unvoiced)
                ppg:           [T, ppg_dim]     float32 (從 fp16 轉)
                spk_emb:       [256]            float32
                phoneme_id:    [T]              int64 (-1 if not yet clustered)
                meta_*:        scalar string / int
        """
        path = self.npz_paths[idx]
        with np.load(path, allow_pickle=True) as data:
            mel = data["mel"].astype(np.float32)       # [T, 80]
            f0 = data["f0"].astype(np.float32)         # [T]
            voicing = data["voicing"].astype(np.float32)
            register_soft = data["register_soft"].astype(np.float32)  # [T, 5]
            register_id = data["register_id"].astype(np.int64)        # [T]
            ppg = data["ppg"].astype(np.float32)       # [T, ppg_dim] (從 fp16 上轉)
            spk_emb = data["spk_emb"].astype(np.float32)               # [256]
            # phoneme_id 可能尚未存在（cluster_ppg 還沒跑）
            if "phoneme_id" in data.files:
                phoneme_id = data["phoneme_id"].astype(np.int64)
            else:
                phoneme_id = np.full(mel.shape[0], self.PHONEME_ID_PADDING, dtype=np.int64)

            meta_dataset = str(data["meta_dataset"])
            meta_speaker = str(data["meta_speaker_id"])
            meta_item = str(data["meta_item_id"])

        T = mel.shape[0]

        # 隨機切 max_frames 段
        # 為什麼用 np.random 而非全域 seed：DataLoader worker 用各自 seed，
        # 同一 epoch 不同 worker 切點不同，加強隨機性
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames + 1)
            end = start + self.max_frames
            mel = mel[start:end]
            f0 = f0[start:end]
            voicing = voicing[start:end]
            register_soft = register_soft[start:end]
            register_id = register_id[start:end]
            ppg = ppg[start:end]
            phoneme_id = phoneme_id[start:end]

        return {
            "mel": torch.from_numpy(mel),
            "f0": torch.from_numpy(f0),
            "voicing": torch.from_numpy(voicing),
            "register_soft": torch.from_numpy(register_soft),
            "register_id": torch.from_numpy(register_id),
            "ppg": torch.from_numpy(ppg),
            "spk_emb": torch.from_numpy(spk_emb),
            "phoneme_id": torch.from_numpy(phoneme_id),
            "meta_dataset": meta_dataset,
            "meta_speaker_id": meta_speaker,
            "meta_item_id": meta_item,
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    把 list of samples 打包成 batch，pad 到 batch 內最長長度。

    為什麼 pad 而非 truncate：
      truncate 損失資訊；pad+mask 在 transformer/conv 都常見處理。
      下游 model / loss 看 mel_mask 忽略 padding 區。

    Padding 規則：
      mel:           pad 0           （ln 0 = mel silence floor）
      f0:            pad 0           （unvoiced = 0 慣例）
      voicing:       pad 0
      register_soft: pad 0
      register_id:   pad -1          （unvoiced sentinel）
      ppg:           pad 0
      spk_emb:       不變（固定 256 維）
      phoneme_id:    pad -1
      mel_mask:      返回 [B, T_max]，1=valid, 0=pad（給 model / loss 用）
    """
    # 計算 batch 內最長 T
    T_max = max(s["mel"].shape[0] for s in batch)
    B = len(batch)

    # 為什麼用 zeros + 條件填 -1：
    # zeros 為大宗 pad value 無爭議；
    # register_id / phoneme_id 用 -1 是 task 自定 sentinel
    mel_dim = batch[0]["mel"].shape[1]
    ppg_dim = batch[0]["ppg"].shape[1]
    reg_dim = batch[0]["register_soft"].shape[1]

    out = {
        "mel":           torch.zeros(B, T_max, mel_dim, dtype=torch.float32),
        "f0":            torch.zeros(B, T_max, dtype=torch.float32),
        "voicing":       torch.zeros(B, T_max, dtype=torch.float32),
        "register_soft": torch.zeros(B, T_max, reg_dim, dtype=torch.float32),
        "register_id":   torch.full((B, T_max), -1, dtype=torch.int64),
        "ppg":           torch.zeros(B, T_max, ppg_dim, dtype=torch.float32),
        "phoneme_id":    torch.full((B, T_max), -1, dtype=torch.int64),
        "spk_emb":       torch.stack([s["spk_emb"] for s in batch]),
        "mel_mask":      torch.zeros(B, T_max, dtype=torch.float32),
        "mel_lengths":   torch.tensor([s["mel"].shape[0] for s in batch], dtype=torch.long),
        "meta_dataset":   [s["meta_dataset"] for s in batch],
        "meta_speaker_id": [s["meta_speaker_id"] for s in batch],
        "meta_item_id":  [s["meta_item_id"] for s in batch],
    }
    for i, s in enumerate(batch):
        T = s["mel"].shape[0]
        out["mel"][i, :T] = s["mel"]
        out["f0"][i, :T] = s["f0"]
        out["voicing"][i, :T] = s["voicing"]
        out["register_soft"][i, :T] = s["register_soft"]
        out["register_id"][i, :T] = s["register_id"]
        out["ppg"][i, :T] = s["ppg"]
        out["phoneme_id"][i, :T] = s["phoneme_id"]
        out["mel_mask"][i, :T] = 1.0
    return out


if __name__ == "__main__":
    # 自我測試（需要先跑過 binarizer.py，且至少有 1 個 .npz 在 data/binarized_smoke/m4singer/）
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/binarized_smoke/m4singer")
    if not root.exists():
        print(f"NOTE: {root} not exist; skipping self-test")
        sys.exit(0)

    ds = BinarizedNSVBDataset(root, max_frames=200)
    print(f"len(ds) = {len(ds)}")

    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:18s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:18s} {v!r}")

    batch = collate_fn([ds[0], ds[0 if len(ds) == 1 else 1 % len(ds)]])
    print("\nbatch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:18s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:18s} {v!r}")
