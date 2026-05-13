"""
nsvb/data/cluster_ppg.py
==========================

【這支檔案做什麼】
Phase 0 第二階段：把 binarizer.py 抽出來的 fp16 PPG（Whisper hidden state, 1280 維）
做 k-means 分群，得到「離散音素 ID」 (phoneme_id, int16)，寫回每個 .npz。
最終 .npz 會新增一個 key：

    phoneme_id  [T_mel]  int16   每 frame 的離散 cluster id

【為什麼要做這一步（D_z 的離散音素條件需要它）】
Plan A' 中 D_z 的條件包含「Discrete Phoneme ID」（透過 nn.Embedding 餵入）：
  - 連續 PPG 給 D_z 條件會讓條件本身洩漏太多語言學細節，
    D_z 容易用「PPG 細節吻合度」當捷徑而非真實品質判斷
  - 離散 phoneme_id 只保留「這是哪一類發音」這個粗訊息，逼 D_z 去學品質特徵
  - 與 Soft Register Bucket 對 F0 的離散化動機完全一致：條件粗化 → 防 D_z 走捷徑

【為什麼用 k-means 而非 Whisper 的 token decoder】
- Whisper decoder 輸出 BPE subword token，不是 phoneme，對歌聲幾乎無意義
  （歌聲拉長母音、連音 → token 化會把單字切碎到無從對應）
- HuBERT-style discrete unit (k-means on hidden state) 是 self-supervised speech
  領域驗證過的「離散音素」近似方法，不需語言學標註
- k-means 對 Whisper hidden state 的 phonetic layer (layer 8) 跑出的 cluster
  在實證上對應到語言學音素的精細變體（如同一個 /a/ 的不同口型/共振峰位置）

【為什麼 K=200】
- 中文音素數量：~23 聲母 + ~35 韻母 × 5 聲調 ≈ 200（細節）
  - 但純粹音素辨識 ~80 類就夠（不分聲調的 zhuyin / pinyin 級別）
- Whisper hidden state 同時捕捉了 prosody / 共發音 / 音高效應，
  經驗上 K=200 才能讓「同 cluster 內」聚集得夠純（變異 < 變異閾值）
- HuBERT base 用 K=100, large 用 K=500；200 是中間值，平衡細節與類別噪聲
- 若 JSD 檢查失敗（VocalVerse vs M4Singer phoneme_id 分布差太多），
  可下調至 K=100 — 越粗 → 越容易兩 dataset 有相同分布

【為什麼用 MiniBatchKMeans 而非標準 KMeans】
- 標準 KMeans 要把所有 frames 載入記憶體：~10000 首 × ~25000 frames × 1280 維 × 2 byte
  = ~640 GB，不可能
- MiniBatchKMeans 用 streaming，每次只看 batch_size 個樣本，記憶體 ~MB
- sklearn 實作對 d=1280 的 1M 樣本約 30-60 秒收斂

【為什麼要先 sample 而非用全部 frames 訓練】
- 全部 frames ~250M，連 MiniBatchKMeans 也太慢（隨機讀取磁碟 IO bottleneck）
- 每首歌 sample 200 frames：~2M frames，8 GB fp32 in-memory，可全載
- 抽樣不損失品質：phoneme cluster 的形狀由 millions of frames 決定，
  從 250M 降到 2M 仍遠超 K=200 的需求（每 cluster 平均 ~10000 樣本）

【兩階段 pipeline】
- Stage A: 對所有 .npz 做 frame sampling，fit k-means → 存 centroids 到磁碟
- Stage B: 對所有 .npz 做 predict（用已 fit 的 centroids），把 phoneme_id 寫回
- 分兩階段的好處：Stage A 有時可手動檢查 cluster 數量、視覺化是否合理；
  確認後才跑 Stage B（耗時長）

【為什麼 phoneme_id 用 int16 而非 int8】
- int8 範圍 [-128, 127]，K=200 超出
- uint8 範圍 [0, 255] 可以，但若未來想試 K=500 就要再改 dtype，麻煩
- int16 [−32k, 32k] 一勞永逸；儲存空間每首歌只多 ~25 KB，可忽略
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from nsvb.utils.audio_config import WHISPER_HIDDEN_DIM


# ── 預設參數 ────────────────────────────────────────────
# K：cluster 數量（見檔頭「為什麼 K=200」說明）
DEFAULT_K = 200

# 每首歌抽樣多少 frames 用於 fit k-means
# 為什麼 200：~10000 首歌 × 200 = 2M frames，足以收斂；
#            單首歌均勻抽避免長歌獨佔分布
DEFAULT_FRAMES_PER_SONG = 200

# MiniBatchKMeans 的 batch size
# 為什麼 8192：sklearn 預設 1024，但 d=1280 高維下 batch 大一點 reassignment 較穩定；
#              8192 × 1280 × 4 = 40 MB，記憶體可接受
DEFAULT_KMEANS_BATCH = 8192

# 為什麼 max_iter=100：sklearn 預設 100；MiniBatchKMeans 通常 30-50 iter 已收斂
DEFAULT_MAX_ITER = 100

# k-means 收斂判定 tolerance
# 為什麼 1e-4：與 sklearn 預設一致；對 d=1280 的尺度合理
DEFAULT_TOL = 1e-4

# 隨機種子
# 為什麼固定：k-means 對 init 敏感，不固定種子可能導致 phoneme_id 對應在
#            不同次跑時整個 permute（不影響功能但難以對比中間結果）
RANDOM_STATE = 42


# ── Stage A: 收集 PPG samples 並 fit k-means ──────────
def collect_ppg_samples(
    npz_paths: List[Path],
    frames_per_song: int = DEFAULT_FRAMES_PER_SONG,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """
    從每個 .npz 隨機抽 `frames_per_song` 個 PPG frames（避開首尾各 5 frame
    的 Whisper 邊界 artifact），合併回傳 [N_total, 1280] fp32 array。

    為什麼避開首尾 5 frames：
      Whisper chunk 邊界附近 hidden state 受 padding 干擾，分布可能異常；
      雖然 ppg_whisper.py 已用 overlap-take-middle 處理，但保險再切 5 frame。

    為什麼 fp32 in-memory：
      .npz 存 fp16 是為了磁碟空間，但 k-means 計算用 fp32 數值精度更穩；
      sklearn 對 fp16 也支援但 internal 仍轉 fp32。
    """
    rng = np.random.default_rng(seed)
    pieces = []

    for i, p in enumerate(npz_paths):
        try:
            with np.load(p) as data:
                ppg = data["ppg"]  # [T, 1280] fp16
        except (KeyError, OSError) as e:
            print(f"  [skip] {p.name}: {e}")
            continue

        T = ppg.shape[0]
        # 邊界 5 frame 各砍掉
        if T <= 10:
            # 短歌（< 0.12s）跳過
            continue
        valid = np.arange(5, T - 5)
        if len(valid) == 0:
            continue

        n = min(frames_per_song, len(valid))
        idx = rng.choice(valid, size=n, replace=False)
        pieces.append(ppg[idx].astype(np.float32))

        if (i + 1) % 100 == 0:
            print(f"  collected {i+1}/{len(npz_paths)} songs, "
                  f"running total = {sum(pc.shape[0] for pc in pieces)} frames")

    if not pieces:
        raise RuntimeError("No PPG frames collected — check .npz files exist and have 'ppg' key")

    samples = np.concatenate(pieces, axis=0)
    print(f"Total samples for k-means: {samples.shape}")
    return samples


def fit_kmeans(
    samples: np.ndarray,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_KMEANS_BATCH,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    seed: int = RANDOM_STATE,
) -> MiniBatchKMeans:
    """
    在 [N, 1280] 上 fit MiniBatchKMeans。

    為什麼用 init='k-means++'：
      sklearn 預設；對高維分布隨機 init 容易收斂到爛 minimum，
      k-means++ 用 D^2 加權選 init points，empirical 比純隨機好很多。

    為什麼 reassignment_ratio=0.01（預設）：
      MiniBatchKMeans 會週期性 reassign 樣本少的 cluster 到新 init，
      避免 cluster collapse；預設值對歌聲 PPG 經驗上 OK。
    """
    print(f"Fitting MiniBatchKMeans: K={k}, batch={batch_size}, max_iter={max_iter}")
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        init="k-means++",
        n_init=3,           # 跑 3 次 init 取最好；保護 against bad init
        verbose=1,          # 印每 iter inertia 變化
    )
    km.fit(samples)
    print(f"k-means done. inertia={km.inertia_:.2f}")
    return km


# ── Stage B: 把 phoneme_id 寫回每個 .npz ──────────────
def assign_phoneme_id(
    npz_paths: List[Path],
    centroids: np.ndarray,
    overwrite: bool = False,
) -> int:
    """
    對每個 .npz：載入 ppg → 用 centroids 分群 → 把 phoneme_id 寫回 .npz。

    Args:
        centroids: [K, 1280] fp32，k-means 訓練好的中心
        overwrite: True 則覆寫已有 phoneme_id；False 則 skip 已含 phoneme_id 的檔

    Returns:
        實際處理的檔案數量

    為什麼用「整檔重寫」而非 in-place append：
      .npz 檔本身是 zip 壓縮 archive，標準 numpy API 沒有 in-place append；
      重寫整檔簡單可靠，讀寫成本可接受（單檔 ~50 MB，SSD ~1s）。

    為什麼預測用 fp32 而非 fp16：
      euclidean distance 在 fp16 對 d=1280 很容易 underflow 到 0；fp32 安全。
    """
    centroids_f32 = centroids.astype(np.float32)
    processed = 0

    for i, p in enumerate(npz_paths):
        with np.load(p, allow_pickle=True) as data:
            keys = list(data.files)
            if "phoneme_id" in keys and not overwrite:
                continue

            # 全部讀入 dict（np.savez 不支援 partial update）
            payload = {k: data[k] for k in keys}

        ppg = payload["ppg"].astype(np.float32)  # [T, 1280]
        # 直接算 squared euclidean distance：
        #   dist[t, k] = ||ppg[t] - centroids[k]||^2
        # 為什麼不直接用 km.predict：
        #   sklearn predict 內部要 wrap 成 KMeans state，這裡用純 numpy 更快
        # 用展開式：||a-b||^2 = ||a||^2 - 2 a·b + ||b||^2
        ppg_norm = np.sum(ppg ** 2, axis=1, keepdims=True)        # [T, 1]
        cen_norm = np.sum(centroids_f32 ** 2, axis=1)             # [K]
        dot = ppg @ centroids_f32.T                                # [T, K]
        dist = ppg_norm + cen_norm[None, :] - 2.0 * dot           # [T, K]
        phoneme_id = np.argmin(dist, axis=1).astype(np.int16)     # [T]

        payload["phoneme_id"] = phoneme_id

        # 重新存回（用 _compressed 與原 binarizer 一致）
        np.savez_compressed(p, **payload)
        processed += 1

        if processed % 50 == 0:
            print(f"  assigned {processed}/{len(npz_paths)} files")

    return processed


# ── 工具：列出已存在的 .npz ────────────────────────────
def list_npz(root: Path) -> List[Path]:
    """
    列出 `root` 下所有 .npz（遞迴）。

    為什麼遞迴：
      binarizer 預設輸出結構 data/binarized/{dataset}/{item_id}.npz；
      若未來分子目錄（如按 speaker），仍能掃到。
    """
    return sorted(root.rglob("*.npz"))


# ── CLI ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-means cluster PPG → phoneme_id")
    parser.add_argument(
        "--binarized-root",
        default="data/binarized",
        help="binarizer.py 輸出根目錄（會掃所有子目錄的 .npz）",
    )
    parser.add_argument(
        "--centroids-out",
        default="data/binarized/ppg_kmeans_centroids.npy",
        help="k-means centroids 儲存路徑",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--frames-per-song",
        type=int,
        default=DEFAULT_FRAMES_PER_SONG,
        help="Stage A 每首歌抽多少 frames",
    )
    parser.add_argument(
        "--stage",
        choices=["fit", "assign", "all"],
        default="all",
        help="fit=只 fit k-means; assign=只用既有 centroids 寫回 phoneme_id; all=兩者",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="即使 .npz 已含 phoneme_id 仍覆寫",
    )
    args = parser.parse_args()

    root = Path(args.binarized_root)
    if not root.exists():
        raise FileNotFoundError(f"binarized root not found: {root}")

    npz_paths = list_npz(root)
    if not npz_paths:
        raise RuntimeError(f"No .npz under {root}; run binarizer.py first")
    print(f"Found {len(npz_paths)} .npz files under {root}")

    centroids_path = Path(args.centroids_out)
    centroids_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage A: fit ──
    if args.stage in ("fit", "all"):
        samples = collect_ppg_samples(npz_paths, frames_per_song=args.frames_per_song)
        km = fit_kmeans(samples, k=args.k)
        np.save(centroids_path, km.cluster_centers_.astype(np.float32))
        print(f"Saved centroids to {centroids_path} shape={km.cluster_centers_.shape}")

    # ── Stage B: assign ──
    if args.stage in ("assign", "all"):
        if not centroids_path.exists():
            raise FileNotFoundError(
                f"Centroids missing: {centroids_path}. Run --stage fit first."
            )
        centroids = np.load(centroids_path)
        assert centroids.shape[1] == WHISPER_HIDDEN_DIM, (
            f"Centroids dim {centroids.shape[1]} ≠ WHISPER_HIDDEN_DIM {WHISPER_HIDDEN_DIM}"
        )
        n_done = assign_phoneme_id(npz_paths, centroids, overwrite=args.overwrite)
        print(f"Assigned phoneme_id to {n_done}/{len(npz_paths)} files")


if __name__ == "__main__":
    main()
