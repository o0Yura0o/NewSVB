"""
nsvb/utils/jsd_check.py
========================

【這支檔案做什麼】
計算兩個資料集（VocalVerse 業餘 vs M4Singer 職業）在以下兩個維度的 Jensen-Shannon
Divergence：
  1. Register 頻率分布（5 個聲區 bucket 的 frame 級頻率）
  2. Phoneme 頻率分布（discrete phoneme ID 的 frame 級頻率）

並提供「不達標 → 重採樣」的索引產生器，用於 Phase 0 資料策展。

【為什麼需要它】
NSVB-ZH 是 unpaired 訓練，D_z（z 層判別器）以 register + phoneme 為條件區分 real/fake。
如果兩個資料集在這兩個條件上的邊際分布差太大：
  - D_z 不需要看 z 內容，只用「這 frame 在哪個 register/phoneme bucket」就能猜出
    real(pro) 或 fake(amateur)，這是分布捷徑（distribution shortcut）。
  - 結果：M 收不到 z 層級的有效梯度，訓練退化成 register/phoneme 平衡器。

JSD < 0.05 是經驗閾值（自然對數下 [0, log 2 ≈ 0.693] 範圍內，0.05 表示兩分布
相對接近）。不達標時，必須對較大的資料集做下採樣，使兩邊邊際分布對齊。

【為什麼用 JSD 而非 KL】
  - JSD 對稱（JSD(P,Q)=JSD(Q,P)），KL 不對稱。我們關心兩個資料集的「相似度」而非
    哪個是 reference，所以對稱量更合適。
  - JSD 有上界（log 2，自然對數下），便於設定通用閾值。KL 沒有上界。
  - JSD = 0.5*KL(P‖M) + 0.5*KL(Q‖M)，其中 M=(P+Q)/2，數值上更穩定（M 不會有 0 機率）。

【為什麼這支用 numpy 而不是 torch】
資料策展是 Phase 0 一次性離線腳本，不在訓練 loop 內，不需要 GPU/autograd。
numpy 更輕量，輸出也方便 dump 成 JSON 報告。
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# JSD 閾值（達標代表兩資料集邊際分布夠接近）
# 0.05 來自經驗：在 5-bucket register 上 0.05 約等於最大差距 bucket 的頻率差 < 5%
JSD_THRESHOLD = 0.05


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    計算兩個離散分布的 JSD（自然對數，上界 log 2 ≈ 0.693）。

    Args:
        p, q: shape [K]，K 為類別數，每個元素是該類別的機率（總和=1）
        eps:  數值穩定下限，避免 log(0)

    Returns:
        jsd: 標量 float，[0, log 2]

    為什麼加 eps 而不是直接過濾 0：
      P 或 Q 可能在某個 bucket 機率為 0（例如某資料集完全沒有最高聲區），
      但 M=(P+Q)/2 該 bucket 仍可能 > 0。直接過濾會造成 P 與 M 維度不一致。
      加 eps 對結果影響可忽略（< 1e-10）。
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps

    # 重新正規化（加 eps 後總和會略大於 1）
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    # KL(p ‖ m) = Σ p * log(p/m)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def frame_freq_distribution(ids: np.ndarray, num_classes: int,
                             ignore_value: int = -1) -> np.ndarray:
    """
    把 frame-level discrete ID array 轉成類別頻率分布。

    Args:
        ids:          [N_total_frames]，整個資料集拼起來的離散 ID（可含 ignore_value）
        num_classes:  類別總數 K
        ignore_value: 視為「無效 frame」的標籤（如 unvoiced=-1）

    Returns:
        dist: [K]，每個類別的 frame 比例（總和=1）

    為什麼用 frame 級而非 utterance 級：
      Frame 級頻率反映「歌手實際在每個聲區/音素上花的時間」，這正是 D_z 看到的條件分布。
      Utterance 級（每首歌取一個主導 bucket）會洗掉很多細節，與 D_z 實際接收的訊號不符。
    """
    valid = ids[ids != ignore_value]
    if valid.size == 0:
        # 完全空資料集，回傳 uniform 避免 JSD 退化
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)

    counts = np.bincount(valid.astype(np.int64), minlength=num_classes)
    return counts.astype(np.float64) / counts.sum()


def check_jsd_pair(
    name_a: str, ids_a: np.ndarray,
    name_b: str, ids_b: np.ndarray,
    num_classes: int,
    label: str = "register",
    ignore_value: int = -1,
    threshold: float = JSD_THRESHOLD,
) -> Dict:
    """
    對兩個資料集做 JSD 檢查，輸出可讀報告 + 是否達標。

    Args:
        name_a/name_b: 資料集名稱（純記錄用，例如 "VocalVerse"、"M4Singer"）
        ids_a/ids_b:   各自拼成的 [N_total_frames] discrete ID
        num_classes:   K
        label:         "register" 或 "phoneme"，影響報告格式
        threshold:     JSD 閾值

    Returns:
        report dict: {
            'label': str, 'jsd': float, 'pass': bool,
            'dist_a': list, 'dist_b': list, 'threshold': float
        }
    """
    dist_a = frame_freq_distribution(ids_a, num_classes, ignore_value)
    dist_b = frame_freq_distribution(ids_b, num_classes, ignore_value)
    jsd = jensen_shannon_divergence(dist_a, dist_b)

    return {
        "label": label,
        "name_a": name_a,
        "name_b": name_b,
        "jsd": jsd,
        "pass": jsd < threshold,
        "threshold": threshold,
        "dist_a": dist_a.tolist(),
        "dist_b": dist_b.tolist(),
    }


def resample_indices_to_match(
    ids_per_utt: List[np.ndarray],
    target_dist: np.ndarray,
    num_classes: int,
    ignore_value: int = -1,
    max_drop_ratio: float = 0.5,
    seed: int = 42,
) -> List[int]:
    """
    JSD 不達標時的「下採樣」工具：從一個資料集挑選 utterance 子集，
    使其邊際分布盡量接近 target_dist。

    Args:
        ids_per_utt:    list[np.ndarray]，每個元素是一首歌的 [T_i] discrete ID
        target_dist:    [K]，要對齊的目標分布（通常是另一個資料集的分布）
        num_classes:    K
        ignore_value:   無效標籤
        max_drop_ratio: 最多丟掉比例（防止全丟），預設最多丟 50%
        seed:           隨機種子，便於重現

    Returns:
        keep_indices: list[int]，保留的 utterance index

    演算法：貪婪對齊
      1. 計算每首歌對「對齊度」的邊際貢獻（utterance 分布 vs target 的 KL 距離）
      2. 從最不對齊的 utterance 開始丟，直到 JSD 達標或丟夠 max_drop_ratio
      3. 每丟一首重新計算總體分布，避免一次丟太多造成過度修正

    為什麼選貪婪而非 LP optimization：
      LP 解最佳化但實作複雜（需 scipy.optimize.linprog）；歌曲數量級 ~10k，
      貪婪一次 pass 已足夠把 JSD 從 ~0.1 拉到 < 0.05，且結果可解釋（明確指出哪首被丟）。
    """
    rng = np.random.default_rng(seed)
    n_utts = len(ids_per_utt)

    # 預計算每首歌的 utt-level 分布
    utt_dists = []
    for ids in ids_per_utt:
        utt_dists.append(frame_freq_distribution(ids, num_classes, ignore_value))
    utt_dists = np.stack(utt_dists)  # [N, K]

    # 預計算每首歌的有效 frame 數（unvoiced 不算）
    utt_weights = np.array([
        (ids != ignore_value).sum() for ids in ids_per_utt
    ], dtype=np.float64)

    keep_mask = np.ones(n_utts, dtype=bool)

    def current_dist():
        # 用 frame 數加權平均得到當前資料集分布
        w = utt_weights * keep_mask
        if w.sum() == 0:
            return np.full(num_classes, 1.0 / num_classes)
        return (utt_dists * w[:, None]).sum(axis=0) / w.sum()

    initial_jsd = jensen_shannon_divergence(current_dist(), target_dist)
    if initial_jsd < JSD_THRESHOLD:
        # 已達標，全部保留
        return list(range(n_utts))

    max_drops = int(n_utts * max_drop_ratio)
    n_dropped = 0

    while n_dropped < max_drops:
        cur = current_dist()
        cur_jsd = jensen_shannon_divergence(cur, target_dist)
        if cur_jsd < JSD_THRESHOLD:
            break

        # 過度代表的 bucket（cur > target）：丟掉這個 bucket 為主導的 utterance
        excess = cur - target_dist  # [K]，正值表示過量
        # 對每首歌打分：分數越高代表丟掉它越能改善（與 excess 正相關）
        # 只考慮還沒被丟的歌
        candidate_mask = keep_mask
        if not candidate_mask.any():
            break

        # 分數 = utterance 分布與 excess 的內積（同方向 = 該歌主導過量 bucket）
        scores = (utt_dists * excess[None, :]).sum(axis=1)
        scores = np.where(candidate_mask, scores, -np.inf)

        # 加微小隨機擾動避免 ties 偏差
        scores = scores + rng.normal(0, 1e-6, size=scores.shape)

        worst = int(np.argmax(scores))
        keep_mask[worst] = False
        n_dropped += 1

    return [i for i in range(n_utts) if keep_mask[i]]


def write_report(report: List[Dict], out_path: str):
    """
    把 JSD 報告寫成 JSON，便於 Phase 0 自動化腳本判斷是否進入 Phase 1。

    為什麼用 JSON 而非純文字：
      Phase 0 的 makefile/腳本可以用 jq 或 python 直接讀 'pass' 欄位決定是否繼續，
      純文字 log 還要 grep 解析容易出錯。
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 自我測試：模擬兩個假資料集，一個分布相近、一個分布很歪
    rng = np.random.default_rng(0)

    # Case 1：兩邊都接近均勻 → JSD 應該很小
    a = rng.choice(5, size=10000, p=[0.20, 0.22, 0.20, 0.20, 0.18])
    b = rng.choice(5, size=10000, p=[0.21, 0.20, 0.20, 0.21, 0.18])
    r1 = check_jsd_pair("VocalVerse_sim", a, "M4Singer_sim", b, num_classes=5)
    print(f"Case 1 (similar):   JSD={r1['jsd']:.4f}  pass={r1['pass']}")

    # Case 2：A 偏低音 B 偏高音 → JSD 應該明顯大
    a = rng.choice(5, size=10000, p=[0.50, 0.30, 0.15, 0.04, 0.01])
    b = rng.choice(5, size=10000, p=[0.01, 0.04, 0.15, 0.30, 0.50])
    r2 = check_jsd_pair("VocalVerse_low", a, "M4Singer_high", b, num_classes=5)
    print(f"Case 2 (skewed):    JSD={r2['jsd']:.4f}  pass={r2['pass']}")

    # Case 3：含 unvoiced（-1）的測試
    a_with_uv = np.concatenate([a, np.full(2000, -1)])
    r3 = check_jsd_pair("VocalVerse_uv", a_with_uv, "M4Singer_uv", b, num_classes=5)
    print(f"Case 3 (with uv):   JSD={r3['jsd']:.4f}  (應與 Case 2 接近)")

    # Case 4：重採樣對齊 (把 Case 2 的 a 拆成 1000 首歌，挑子集對齊到 b 的分布)
    print("\nCase 4: resample_indices_to_match")
    utt_size = 100
    n_utts = len(a) // utt_size
    ids_per_utt = [a[i*utt_size:(i+1)*utt_size] for i in range(n_utts)]
    target = frame_freq_distribution(b, 5)
    keep = resample_indices_to_match(ids_per_utt, target, num_classes=5)

    new_ids = np.concatenate([ids_per_utt[i] for i in keep])
    new_jsd = jensen_shannon_divergence(frame_freq_distribution(new_ids, 5), target)
    print(f"  原始 JSD = {r2['jsd']:.4f}")
    print(f"  重採樣後 JSD = {new_jsd:.4f}  (保留 {len(keep)}/{n_utts} 首歌)")
