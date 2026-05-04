"""
nsvb/inference/dtw_warp.py
============================

【這支檔案做什麼】
Mode B 推理用的 DTW + gather 時間軸對齊工具。

對應 NSVB 原論文「EHSADTW」邏輯（Energy-aware Hierarchical Singing Audio DTW）的
精簡版：對 amateur 與 pro reference 抽出的 frame-level 特徵做 DTW，得到一個
「對 pro 時間軸的每個 frame，要從 amateur 時間軸取哪個 frame」的索引向量，
配合 torch.gather 把 amateur z 重採到 pro 時長。

【為什麼需要這個 module（Mode B 的核心）】
Mode B 的需求：
    - 輸出長度 = T_p（跟 pro 參考的時長走，能直接配 pro 伴奏）
    - 仍保留 amateur 的音色 / 內容指紋（過 M 後的 z_a'）
所以要把長度為 T_z_a 的 z_a' 重採到長度 T_z_p 的時間軸，且重採方式必須對應
語音內容（不是均勻 stretch — 業餘可能某句拖慢、某句搶拍）。

DTW 解這問題的 standard answer：
    1. 對 (mel_a, mel_p_ref) 算 cost matrix
    2. 跑 DP 找最低 cost path → 每個 pro frame 對應一個 amateur frame
    3. 用 torch.gather(z_a', dim=time, index=path) 取出 z_a'_warped (T_z_p)

【為什麼 DTW 用 mel 而非 F0】
- F0 在 unvoiced 段是 0，距離函式碰到 0 vs 0 沒資訊量；
  整段 unvoiced silent 會讓 DTW 隨機亂走
- mel 包含完整聲學特徵（共振峰 + 共鳴 + voicing 強度），對 silent / 子音 / 母音
  全部有意義
- 與 NSVB 原 EHSADTW 一致（NSVB 用 mel + 能量 hierarchical，這裡簡化成 mel only
  就足夠 Phase 3 主力使用情境）

【為什麼簡化 EHSADTW】
原版 EHSADTW 有「層級」（先粗 align 整段 phrase 邊界、再細 align frame）的設計，
複雜度高、實作 ~200 行。Phase 3 主要使用情境是 user 提供同首歌錄音，
amateur 與 pro 大致對齊（都唱同樣旋律），單層 frame DTW 已經能處理 ±20% 的
時長差。等模式 B 真的有人在用了再加層級。

【為什麼用 librosa.sequence.dtw 而非自寫】
- librosa 內部 cython 實作 cost ~10x 於 pure python
- 已經處理過邊界、step pattern、normalization 等細節
- DTW 不是這專案的研究主題，標準工具跑得好就夠

【shape contract】
    f_a: [T_a, D]  amateur 特徵（mel-rate, T_mel_a）
    f_p: [T_p, D]  pro reference 特徵（mel-rate, T_mel_p）
    回傳:
        path_to_a: [T_p] int64   — 對 pro frame t, 從 amateur frame path_to_a[t] 取
        cost_norm: float          — 平均 cost / T_p（debug 用，越低越像）

注意：
- DTW 在 mel-rate 跑（T_mel）；warp 索引要轉成 latent rate (T_z = T_mel / down) 時，
  下游 warp_latent 會做 stride 對應
- z_a' 的時間軸是 T_z_a = T_mel_a / 4（latent_strides=4），所以 path_to_a 算出的
  是 T_p × 1 的 mel-rate index，下游要除 4 並 round 回 latent index
"""

from typing import Tuple

import numpy as np
import torch


def dtw_path_mel(
    mel_a: np.ndarray,
    mel_p: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, float]:
    """
    對 amateur / pro mel 跑 DTW，回傳「pro frame → amateur frame」mapping。

    Args:
        mel_a:  [T_a, NUM_MELS]   業餘音檔 mel
        mel_p:  [T_p, NUM_MELS]   pro reference mel
        metric: librosa.sequence.dtw 的 step distance metric
                'euclidean' = L2 per frame；對 mel 自然
                'cosine'   = 1 - cos sim；對 normed mel 也合理

    Returns:
        path_to_a:  [T_p] int64    每個 pro frame 對應的 amateur frame index
        cost_norm:  float           dtw_cost / T_p

    為什麼用 ndarray 而非 tensor：
      librosa 走 numpy；輸入轉到 numpy 再交給 librosa；輸出回 numpy。
      下游 warp_latent 才轉 tensor。

    為什麼用 dtw 預設 step pattern：
      librosa 預設 weights_add 與 weights_mul 對 (1,1)/(1,0)/(0,1) 三步 cost 一致；
      允許「重複」（同一 amateur frame 對多個 pro frame）與「跳過」（連續 pro frame
      共享 amateur）。對「pro 拖長段、amateur 唱快」與反過來都能 robust 處理。
    """
    import librosa

    # librosa.sequence.dtw 期望 [D, T]（feature × time）
    # 為什麼 transpose：librosa 約定第一軸是 feature；我們 mel 是 [T, F]
    A = mel_a.T.astype(np.float32)   # [F, T_a]
    B = mel_p.T.astype(np.float32)   # [F, T_p]

    # subseq=False：要做完整 sequence alignment，不是子序列匹配
    # backtrack=True：回傳 wp（warp path）讓我們知道 frame 對應
    # 為什麼 metric='euclidean'：mel 是 log scale，euclidean 對應「dB 差」感知合理
    cost_matrix, wp = librosa.sequence.dtw(
        X=A, Y=B,
        metric=metric,
        subseq=False,
        backtrack=True,
    )
    # wp: [N_steps, 2] (i, j)，從 (T_a-1, T_p-1) backtrack 到 (0, 0)
    # 每個 step 是 (a_index, p_index)；可能多個 step 共享同 p_index（amateur 拖長）
    # 也可能多個 step 共享同 a_index（amateur 短、pro 長）

    # 我們要 [T_p] 長度的「pro→amateur」mapping
    # 為什麼取最後一個遇到的 a_index：
    #   wp 中同一 p_index 可能對應多個 a_index（amateur 拖時相當於多個連續 a frame
    #   都走到這個 p frame）；取最後遇到的 = 最 forward-in-time 的 amateur frame，
    #   讓重採後的 z_a 在 amateur 時間軸上盡量「往前推進」，避免重複舊 frame 太多
    T_p = mel_p.shape[0]
    path_to_a = np.full(T_p, -1, dtype=np.int64)
    # wp 是反向（從 end 到 start），翻過來變正向
    for a_idx, p_idx in wp[::-1]:
        if 0 <= p_idx < T_p:
            path_to_a[p_idx] = a_idx
    # 任何沒被填到的 frame（理論上 DTW 會覆蓋全部，但保險）→ 從鄰居填
    if (path_to_a < 0).any():
        # forward fill
        last_valid = 0
        for t in range(T_p):
            if path_to_a[t] < 0:
                path_to_a[t] = last_valid
            else:
                last_valid = path_to_a[t]

    # 確保 amateur index 在合法範圍
    T_a = mel_a.shape[0]
    path_to_a = np.clip(path_to_a, 0, T_a - 1)

    cost_norm = float(cost_matrix[-1, -1] / max(T_p, 1))
    return path_to_a, cost_norm


def warp_latent(
    z: torch.Tensor,
    path_mel_to_a: np.ndarray,
    latent_down_factor: int,
) -> torch.Tensor:
    """
    把 mel-rate 的 path_to_a 轉成 latent-rate index，並用 torch.gather warp z。

    Args:
        z:                   [B, latent, T_z_a]   amateur latent (= M(z_a))
        path_mel_to_a:       [T_mel_p] int64    每個 pro mel frame 對應的 amateur mel frame
        latent_down_factor:  encoder 下採倍數（NSVB 預設 4）

    Returns:
        z_warped: [B, latent, T_z_p]
                  T_z_p = ceil(T_mel_p / latent_down_factor)

    為什麼要先 mel-rate path → latent-rate path：
      DTW 在 mel-rate 跑（特徵密、對齊細）；但 z 是 latent-rate（已 down 4x）。
      簡單做法：對每個 pro latent frame t，從 path 對應的 mel 區間 [t*4, (t+1)*4)
      取代表 frame（取中位 frame 做為 amateur latent frame index）。

    為什麼用 torch.gather 而非 indexing：
      torch.gather 對 batched tensor 的 axis=2 取 index 是 framework 原生支援，
      autograd 友善（雖然推理 no_grad，但與訓練端共用同樣 op 寫法不易出 bug）；
      [B, latent, T_z_a] 直接用 [..., index] 也行但 batch 維度需手動 expand。
    """
    B, C, T_z_a = z.shape
    T_mel_p = path_mel_to_a.shape[0]
    T_z_p = (T_mel_p + latent_down_factor - 1) // latent_down_factor

    # ── Step 1: 對每個 pro latent frame，找代表 amateur mel frame ──
    # 為什麼取 stride 中央 frame：
    #   區間 [t*down, (t+1)*down) 對應一個 amateur mel slice；取中央 frame index
    #   作為 amateur 端的代表，比起取頭/尾不易受邊界 artifact 影響
    rep_mel_idx = np.minimum(
        np.arange(T_z_p) * latent_down_factor + latent_down_factor // 2,
        T_mel_p - 1,
    )
    a_mel_idx = path_mel_to_a[rep_mel_idx]                     # [T_z_p] amateur mel index
    a_z_idx = a_mel_idx // latent_down_factor                  # [T_z_p] amateur latent index
    a_z_idx = np.clip(a_z_idx, 0, T_z_a - 1)

    # ── Step 2: torch.gather 把 z 沿 time 軸取 index ──
    # gather 期望 index shape == output shape；要 broadcast 到 [B, C, T_z_p]
    idx_t = torch.from_numpy(a_z_idx).long().to(z.device)       # [T_z_p]
    idx_expand = idx_t.view(1, 1, T_z_p).expand(B, C, T_z_p)    # [B, C, T_z_p]
    z_warped = torch.gather(z, dim=2, index=idx_expand)
    return z_warped


def warp_feature_mel_rate(
    feat: torch.Tensor,
    path_mel_to_a: np.ndarray,
) -> torch.Tensor:
    """
    對 mel-rate 的 amateur 特徵（如 ppg, register, mel）做同樣 warp 到 T_mel_p。

    Args:
        feat:           [B, T_a, D]   amateur 特徵（time-major）
        path_mel_to_a:  [T_mel_p] int64

    Returns:
        feat_warped: [B, T_mel_p, D]

    為什麼提供這個輔助：
      Mode B 的設計選擇之一是「PPG 用 amateur 經 warp 到 T_p 時間軸」（與 z 一致）；
      這需要在 mel-rate 跑同樣的 gather。獨立函式避免在 pipeline 重複寫 expand 邏輯。
    """
    B, T_a, D = feat.shape
    T_p = path_mel_to_a.shape[0]
    a_idx = np.clip(path_mel_to_a, 0, T_a - 1)
    idx_t = torch.from_numpy(a_idx).long().to(feat.device)        # [T_p]
    idx_expand = idx_t.view(1, T_p, 1).expand(B, T_p, D)
    return torch.gather(feat, dim=1, index=idx_expand)


if __name__ == "__main__":
    # 自我測試：合成 amateur (T=100) + pro (T=80) 線性 mel；
    # DTW 應該找到接近線性 stretch 的 path，warp 後 shape 對得上
    np.random.seed(0)
    T_a, T_p, F = 100, 80, 80
    # amateur 與 pro 是同一段「向上 ramp」mel，但 amateur 拖了 1.25 倍
    base = np.linspace(-3, 0, T_p)[:, None] * np.linspace(0, 1, F)[None, :]
    mel_p = base + np.random.randn(T_p, F) * 0.05
    # 把 pro 線性 stretch 1.25 倍生成 amateur
    mel_a = np.zeros((T_a, F), dtype=np.float32)
    for t in range(T_a):
        src = min(int(t / T_a * T_p), T_p - 1)
        mel_a[t] = mel_p[src] + np.random.randn(F) * 0.05

    path, cost = dtw_path_mel(mel_a.astype(np.float32), mel_p.astype(np.float32))
    print(f"DTW path shape: {path.shape}  (expect [{T_p}])")
    print(f"DTW cost / T_p: {cost:.4f}")
    print(f"path range: [{path.min()}, {path.max()}]  (expect within [0, {T_a-1}])")
    # 線性 stretch 下，path[t] ≈ t * 1.25
    expected = np.arange(T_p) * (T_a / T_p)
    err = np.abs(path - expected).mean()
    print(f"path 與 linear-stretch 期望 mean abs diff: {err:.2f} frames")

    # 測 warp_latent
    B, C, T_z_a = 2, 128, T_a // 4
    z = torch.randn(B, C, T_z_a)
    z_warped = warp_latent(z, path, latent_down_factor=4)
    print(f"z_warped shape: {z_warped.shape}  (expect [{B},{C},{(T_p + 3) // 4}])")