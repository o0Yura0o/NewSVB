"""
nsvb/utils/soft_bucket.py
==========================

【這支檔案做什麼】
把 frame-level F0（Hz, shape [B, T]）轉成 5 維 soft register vector（shape [B, T, 5]），
作為 D_z（z 層判別器）的「聲區條件」(register condition)。

【為什麼需要它】
NSVB-ZH 是 unpaired 訓練，D_z 必須收條件才能避免兩種失敗：
  1. F0 shortcut：若直接餵連續 F0，amateur F0（走音）vs pro F0（準）分布不同，
     D_z 會用「F0 品質」當捷徑，M 拿不到有效梯度。
  2. 全域共鳴濾波攻擊：若 D_z 完全無條件，M 可能把所有 amateur 都套上「最高分的
     pro 技術指紋」（例如低音也加頭腔共鳴），造成 mode collapse。

解法：把 F0 量化成 5 個 register bucket（中心在 C3/G3/D4/A4/E5）。
資訊量只有 ~2.3 bit，連續 F0 的細節（vibrato、走音半音）都跨不過一個 bucket，
因此既不會洩漏 F0 品質，又能強制 M 在「同聲區內向 pro 靠攏」（低音保持胸聲、
高音長頭聲）。

【為什麼用 Soft Gaussian 而不是 Hard bucket】
Hard quintile bucket 會在邊界處有不連續：F0 緊貼 bucket 邊界做 vibrato 時，
register one-hot 會在兩個 bucket 之間跳動，給 D_z 一個假訊號。
Soft Gaussian（log-Hz space, σ=0.3）讓邊界處 smooth 過渡，vibrato/portamento
跨 bucket 時 weight 是漸變的，D_z 不會學到這個假邊界。

【為什麼用 log-Hz 而不是線性 Hz】
人耳對音高的感知是對數的（一個八度 = log(2)），相鄰音名間距在 log-Hz 下等距。
σ=0.3 log-Hz ≈ 半個八度的 1σ 寬度，剛好讓相鄰 bucket 有約 50% 的 weight overlap，
落在邊界中點時兩個 bucket 各拿 ~0.5。
"""

import math
import torch


# 5 個 register bucket 中心，覆蓋人聲常用音域 C3 ~ E5（約 130 Hz ~ 660 Hz）
# 選擇依據：男低音胸聲區、男高音/女低音混聲區、女高音頭聲區，間距約一個五度（~7 半音）
#   C3 = 130.81 Hz   男低音胸聲核心
#   G3 = 196.00 Hz   男聲混聲過渡
#   D4 = 293.66 Hz   女聲胸聲 / 男高音頭聲
#   A4 = 440.00 Hz   女聲混聲核心
#   E5 = 659.25 Hz   女高音頭聲
BUCKET_CENTERS_HZ = torch.tensor([130.81, 196.00, 293.66, 440.00, 659.25],
                                  dtype=torch.float32)

# 預先計算 log-Hz 中心（避免 forward 重複算 log）
BUCKET_CENTERS_LOG = torch.log(BUCKET_CENTERS_HZ)

# Gaussian 寬度（log-Hz space）
# σ=0.3 ≈ 半個八度的 1σ，讓相鄰 bucket 邊界有 ~50% overlap
SIGMA_LOG = 0.3

# 數值穩定下限：F0 < EPS 視為 unvoiced（靜音段、子音段）
F0_EPS = 1e-6

NUM_BUCKETS = len(BUCKET_CENTERS_HZ)


def soft_bucketize_f0(f0: torch.Tensor) -> torch.Tensor:
    """
    把 F0（Hz）轉成 soft register vector。

    Args:
        f0: [B, T] 或 [T]，單位 Hz；unvoiced frame 應為 0 或負值

    Returns:
        register: [B, T, 5] 或 [T, 5]
                  voiced frame：5 個 bucket 的 Gaussian weight（總和=1）
                  unvoiced frame：全 0 向量（讓 D_z 知道「這 frame 沒音高條件」）

    為什麼 unvoiced 給全 0 而不是 uniform：
      Uniform (1/5 each) 會被 D_z 當成一個合法的 register state，可能誤導判別。
      全 0 等同於「告訴 D_z：這 frame 不要用 register 條件去判斷」，
      搭配 D_z 的條件分支設計（後續實作時，register 是 additive bias），
      0 向量自然不貢獻偏置，等同於 frame-wise 條件遮罩。
    """
    # 自動偵測裝置
    centers_log = BUCKET_CENTERS_LOG.to(f0.device)

    # 1. voiced mask：F0 有效值才參與 bucketize
    voiced = f0 > F0_EPS  # [..., T]

    # 2. 取 log-F0；unvoiced 位置先用 EPS 避免 log(0) 報錯，後面會被 mask 掉
    log_f0 = torch.log(f0.clamp(min=F0_EPS))  # [..., T]

    # 3. 計算每個 frame 對 5 個 bucket 中心的距離（log-Hz space）
    # log_f0[..., None]: [..., T, 1]
    # centers_log:       [5]   會自動 broadcast 成 [..., T, 5]
    dist = log_f0.unsqueeze(-1) - centers_log

    # 4. Gaussian kernel：weight = exp(-0.5 * (dist/σ)²)
    # 距離越近 weight 越大；σ 控制平滑程度
    weights = torch.exp(-0.5 * (dist / SIGMA_LOG) ** 2)  # [..., T, 5]

    # 5. 正規化（每個 frame 的 5 個 weight 加總=1）
    # 為什麼要 normalize：D_z 後續做 conditional bias 時假設條件是 probability-like，
    #                    norm 後 5 維和為 1，避免不同 F0 的條件信號強度不一致
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # 6. 把 unvoiced frame 整列清零
    # voiced[..., None] broadcast 到 [..., T, 5]
    weights = torch.where(
        voiced.unsqueeze(-1),
        weights,
        torch.zeros_like(weights),
    )

    return weights


def hz_to_register_id(f0: torch.Tensor) -> torch.Tensor:
    """
    Hard bucket（僅用於統計、JSD 計算等需要離散標籤的場合）。

    為什麼留這個：
      Phase 0 的資料策展要計算 register 頻率分布的 JSD（M4Singer vs OpenSinger），
      JSD 的輸入需要離散類別頻率，所以這裡用 argmax 取最近的 bucket id。
      訓練時不要用這個函式——訓練必須用 soft 版本。

    Args:
        f0: [B, T] 或 [T]，Hz

    Returns:
        ids: 同 shape，int64
             voiced frame：0~4
             unvoiced frame：-1（呼叫端可選擇過濾掉）
    """
    centers_log = BUCKET_CENTERS_LOG.to(f0.device)

    voiced = f0 > F0_EPS
    log_f0 = torch.log(f0.clamp(min=F0_EPS))
    dist = (log_f0.unsqueeze(-1) - centers_log).abs()  # 距離取絕對值
    ids = dist.argmin(dim=-1).to(torch.int64)

    # unvoiced 標 -1，呼叫端用 ids[ids >= 0] 過濾
    ids = torch.where(voiced, ids, torch.full_like(ids, -1))
    return ids


def register_distribution(f0: torch.Tensor) -> torch.Tensor:
    """
    計算一個資料集的 register 頻率分布（5 維 prob，總和=1）。

    用途：Phase 0 的 JSD 檢查
      JSD(M4Singer's register dist, OpenSinger's register dist) < 0.05

    為什麼這個檢查必要：
      如果兩個資料集的 register 分布差太多（例如 M4Singer 偏低音、OpenSinger 偏高音），
      D_z 學到的「real vs fake」邊界會完全與 register 分布耦合，等於用聲區頻率當捷徑，
      M 拿不到有效梯度。

    Args:
        f0: [N_total_frames] 或 [B, T]，整個資料集（或一個 split）的 F0 拼起來

    Returns:
        dist: [5]，每個 bucket 的 voiced frame 比例（unvoiced 不計）
    """
    ids = hz_to_register_id(f0).flatten()
    voiced_ids = ids[ids >= 0]

    if voiced_ids.numel() == 0:
        # 全部 unvoiced，回傳均勻分布以避免下游 NaN
        return torch.full((NUM_BUCKETS,), 1.0 / NUM_BUCKETS)

    counts = torch.bincount(voiced_ids, minlength=NUM_BUCKETS).to(torch.float32)
    dist = counts / counts.sum()
    return dist


if __name__ == "__main__":
    # 自我測試：印出幾個典型音高的 soft register
    # 為什麼留這個：開發時驗證行為直觀，避免 import 後才發現邊界處有 bug
    test_freqs = torch.tensor([
        0.0,      # unvoiced → 全 0
        130.81,   # 正中 bucket 0 → 主要 bucket 0
        163.0,    # C3 與 G3 之間 → bucket 0/1 各半
        196.0,    # 正中 bucket 1
        440.0,    # 正中 bucket 3
        659.25,   # 正中 bucket 4
        800.0,    # 高於 E5 → 仍偏向 bucket 4（最近）
    ]).unsqueeze(0)  # [1, 7]

    weights = soft_bucketize_f0(test_freqs)
    print("F0 (Hz)  ->  Soft Register Weights [C3, G3, D4, A4, E5]")
    for i, hz in enumerate(test_freqs[0]):
        w = weights[0, i].tolist()
        print(f"  {hz.item():7.2f}  ->  [{', '.join(f'{x:.3f}' for x in w)}]")

    ids = hz_to_register_id(test_freqs)
    print("\nHard register IDs (-1 = unvoiced):", ids[0].tolist())
