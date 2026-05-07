"""
nsvb/model/losses.py
======================

【這支檔案做什麼】
集中放 Stage 2 訓練要用的 loss helpers：
    1. hinge_d_loss / hinge_g_loss     — D_z 與 D_mel 的對抗 loss
    2. PatchNCELoss                     — z 與 M(z) 的 frame 對應內容鎖
    3. l_identity_pro                   — M(z_p) ≈ z_p 的 L1 約束
    4. l_ppg                            — output PPG 與 input PPG 的 L1 約束

不在這裡：
    - mel L1 / L2（在 task/stage1.py）
    - KL（FVAE 內部 closed-form）

【設計原則】
- 全部 stateless function 或輕量 nn.Module（PatchNCE 需要學習投影）
- 不耦合特定 model；輸入是 tensor，輸出是 scalar loss
- 為什麼 PatchNCE 包成 Module 而其他純 function：
    PatchNCE 內含 learnable projection head（2-layer MLP），
    必須是 nn.Module 才能加到 optimizer parameter list 裡訓練
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Adversarial losses (hinge) ──────────────────────────
def hinge_d_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for discriminator (D_z 或 D_mel 都用這個)。

    為什麼 hinge 而非 BCE：
        BCE 的 sigmoid 在 D 過強時飽和（gradient 消失），G 學不動；
        hinge 在 |score| > 1 截斷，gradient 永遠 alive；
        對 NSVB 這種 unpaired adv 訓練實證更穩

    L = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
    """
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_g_loss(fake_score: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator (M)，non-saturating form。

    L = -E[D(fake)]    (M 想讓 D 給 fake 高分)
    """
    return -fake_score.mean()


# ── PatchNCE loss ────────────────────────────────────────
class PatchNCELoss(nn.Module):
    """
    PatchNCE 對應 frame 鎖：z 與 M(z) 在「同 time index」的 frame 應比「不同 time index」
    的 frame 更相似。源於 CUT (Park et al. 2020)。

    為什麼用 PatchNCE 而非 L_identity 或純 L_PPG：
        1. L_identity (M(z) ≈ z) 過嚴：M 學不動任何 pro-style 修飾
        2. L_PPG 在 mel 出口檢查內容，太晚（M 已經做完所有改動才知道內容跑掉）
        3. PatchNCE 在 z 層級鎖「frame 對應關係」，**M 可以改變每個 frame 的內容
           特徵，但每 frame 仍與自己 best match**——既保留內容對齊，又允許 z 變化
        4. 不需要任何外部 reference（純 z vs M(z) 對比），訓練便宜

    為什麼有 learnable projection head：
        z 維度 128 直接做 cosine 相似度信噪比低；
        投影到 64 維（更稀疏）後 negative pairs 之間 cosine 差別更顯著，
        contrastive signal 更強。CUT 論文證實 projection 比 raw feature 好

    為什麼 negatives 只在「同 batch item 內取」而非跨 batch：
        跨 batch 的 frame 來自不同歌、不同歌手，naturally 已經很不同——
        contrastive 太簡單 → 學不到細節；
        batch 內取 negatives，model 必須區分「同首歌不同 frame」，學到 fine-grained content invariance

    Args:
        latent_dim:   z 的通道數（128）
        proj_dim:     projection head 出口維度（64）
        temperature:  contrastive 溫度（0.07，CUT 預設）
        num_patches:  每筆 batch item 隨機抽多少 frames 計算 contrastive（128）
    """

    def __init__(self,
                 latent_dim: int = 128,
                 proj_dim: int = 64,
                 temperature: float = 0.07,
                 num_patches: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, proj_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(proj_dim, proj_dim, kernel_size=1),
        )
        self.temperature = temperature
        self.num_patches = num_patches

    def _sample_indices(self, T: int, B: int, device,
                        mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        隨機從每個 batch item 的 [0, T) 抽 num_patches 個 index。

        Args:
            mask: [B, T] (1=valid, 0=pad) or None
        Returns:
            idx: [B, N]

        為什麼用 torch.randint 而非 random.sample：
            randint 是 with replacement，不會 OOB；min(num_patches, T) 已 clamp。
            random.sample 是 without replacement，T < num_patches 時會 raise。

        T 太小時的訊號退化：
            T < 4 時 contrastive 幾乎沒 negative pair（每個 query 最多看到 3 個 key），
            訊號太弱對 gradient 沒貢獻。發出警告（不 raise）讓 caller 知道該調
            max_frames 或 batch_size。
        """
        N = min(self.num_patches, T)
        if T < 4 and not getattr(self, "_warned_small_T", False):
            import warnings
            warnings.warn(
                f"PatchNCE: T_z={T} < 4，contrastive 訊號將極弱。"
                f"建議將 max_frames 提升至 ≥ 64（latent down=4 → T_z ≥ 16）",
                RuntimeWarning,
            )
            self._warned_small_T = True
        if mask is None:
            return torch.randint(0, T, (B, N), device=device)
        # 為什麼 valid frames 不夠就 fallback 到 random：
        #   全 padding 的 batch item 不該存在（dataloader 已過濾 < min_frames）；
        #   保險起見 fallback 避免 zero-sized indexing crash
        idx_list = []
        for b in range(B):
            valid = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                pick = torch.randint(0, T, (N,), device=device)
            else:
                pick = valid[torch.randint(0, valid.numel(), (N,), device=device)]
            idx_list.append(pick)
        return torch.stack(idx_list, dim=0)

    def forward(self,
                z: torch.Tensor,
                z_mapped: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z:        [B, C, T]   M 的輸入
            z_mapped: [B, C, T]   M(z) = z + Δ(z)
            mask:     [B, T] or None    1=valid, 0=padding (z 的 frame rate)

        Returns:
            loss: scalar
        """
        B, C, T = z.shape
        proj_q = self.proj(z)              # [B, D, T]   query (M 的輸入端)
        proj_k = self.proj(z_mapped)       # [B, D, T]   key   (M 的輸出端)

        idx = self._sample_indices(T, B, z.device, mask)  # [B, N]

        # 從同一組 idx 取 q 與 k（保證 positive pair 是同 time 對應）
        D = proj_q.size(1)
        N = idx.size(1)
        idx_exp = idx.unsqueeze(1).expand(-1, D, -1)        # [B, D, N]
        q_feat = torch.gather(proj_q, 2, idx_exp).permute(0, 2, 1).reshape(B * N, D)
        k_feat = torch.gather(proj_k, 2, idx_exp).permute(0, 2, 1).reshape(B * N, D)

        # L2 normalize 後做 cosine similarity
        q_feat = F.normalize(q_feat, dim=-1).view(B, N, D)
        k_feat = F.normalize(k_feat, dim=-1).view(B, N, D)

        # 同 batch item 內：q[i] vs k[j] for all j；對角 j=i 是 positive，其它是 negative
        logits = torch.einsum("bnd,bmd->bnm", q_feat, k_feat) / self.temperature  # [B, N, N]
        targets = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        loss = F.cross_entropy(logits.reshape(B * N, N), targets.reshape(-1))
        return loss


# ── L_identity_pro ───────────────────────────────────────
def l_identity_pro(z_p: torch.Tensor, z_p_mapped: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    M(z_p) ≈ z_p 的 L1 約束。20% batches 隨機抽中時加進 G 的 loss。

    為什麼必要（從 ver1 借鑑）：
        Stage 2 訓練 M 主要看 amateur z，可能讓 M 在 pro 端的行為自由發揮（漂移）；
        即便我們已用 spk_emb 鎖音色（Risk 4 第一層保險），M 對 pro z 加奇怪偏移仍可能
        被 D_z 接受（因為 D_z 看到 pro real 與 M(z_p)≈real 都是 pro 區，無明顯差別）。
        L_identity_pro 額外要求 M 在 pro 輸入近恆等，**逼 M 把「修飾」侷限在 amateur 區**

    為什麼 stochastic（20% prob）而非 always-on：
        always-on 限制 M 對 pro 端的微調空間（pro 聲音本身也有些瑕疵）；
        20% 偶爾被叫去看 pro 端，平均效果是「不要漂太遠」，又不犧牲 amateur 端訓練

    Args:
        z_p / z_p_mapped: [B, C, T]
        mask:             [B, T] or None  (z 的 frame rate)
    """
    diff = (z_p - z_p_mapped).abs()           # [B, C, T]
    if mask is None:
        return diff.mean()
    mask_e = mask.unsqueeze(1)                 # [B, 1, T]
    n_valid = mask.sum().clamp(min=1.0)
    return (diff * mask_e).sum() / (n_valid * z_p.shape[1])


# ── L_PPG ────────────────────────────────────────────────
def l_ppg(ppg_pred: torch.Tensor, ppg_target: torch.Tensor,
          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    輸出 mel 對應 PPG 與輸入 mel 對應 PPG 的 L1 距離。

    為什麼必要：
        PatchNCE 在 z 層鎖內容，但 z → mel 經過 decoder 還可能輕微偏；
        L_PPG 在 mel 出口端「再驗證一次」內容沒跑掉，是雙重保險。

    為什麼 caller 負責提取 PPG 而非這裡內含 Whisper：
        在 training loop 中跑 Whisper 太重（每 step 一次 forward Whisper-large-v3
        是 ~3 秒，訓練拖累 10x+）；
        實作策略：caller 訓練一個輕量 mel→PPG predictor，或在 Stage 2 task 階段
        直接用 PPG 預存的「ground truth PPG」當 ppg_target，而 ppg_pred 由
        快速 mel→PPG 模組現算（後續實作）

    Args:
        ppg_pred:   [B, T_mel, ppg_dim]    輸出 mel 重抽的 PPG（caller 提供）
        ppg_target: [B, T_mel, ppg_dim]    輸入 mel 對應的 GT PPG（從 binarize 讀出）
        mask:       [B, T_mel] or None     mel rate

    Returns:
        loss: scalar
    """
    # 為什麼對齊到較短長度：
    #   ppg_pred 與 ppg_target 可能差 ±1 frame（reconstruct 端 wav 長度差幾 sample）
    T = min(ppg_pred.shape[1], ppg_target.shape[1])
    a = ppg_pred[:, :T]
    b = ppg_target[:, :T]
    diff = (a - b).abs()
    if mask is None:
        return diff.mean()
    mask_a = mask[:, :T].unsqueeze(-1)
    n_valid = mask[:, :T].sum().clamp(min=1.0)
    return (diff * mask_a).sum() / (n_valid * a.shape[-1])


if __name__ == "__main__":
    # 自我測試各 loss
    B, C, T = 4, 128, 80

    # PatchNCE
    nce = PatchNCELoss(latent_dim=C, proj_dim=64, num_patches=64)
    z = torch.randn(B, C, T)
    z_m = z + 0.01 * torch.randn_like(z)  # ≈ z
    mask = torch.ones(B, T)
    mask[:, T // 2:] = 0  # 後半 padding
    loss_nce = nce(z, z_m, mask)
    print(f"PatchNCE (z ≈ z+ε): {loss_nce.item():.4f}  (應該很小，z 與 M(z) 幾乎一致)")

    # 隨機 z2 vs z 的 PatchNCE 對比
    z_random = torch.randn(B, C, T)
    loss_nce_random = nce(z, z_random, mask)
    print(f"PatchNCE (z vs random): {loss_nce_random.item():.4f}  (應該大很多)")

    # hinge losses
    real_s = torch.randn(B, 1, T)
    fake_s = torch.randn(B, 1, T)
    print(f"hinge_d: {hinge_d_loss(real_s, fake_s).item():.4f}")
    print(f"hinge_g: {hinge_g_loss(fake_s).item():.4f}")

    # l_identity_pro
    z_p = torch.randn(B, C, T)
    z_p_m = z_p + 0.05 * torch.randn_like(z_p)
    print(f"l_identity_pro: {l_identity_pro(z_p, z_p_m, mask).item():.4f}")

    # l_ppg
    ppg_a = torch.randn(B, T * 2, 384)
    ppg_b = ppg_a + 0.1 * torch.randn_like(ppg_a)
    mask_mel = torch.ones(B, T * 2)
    print(f"l_ppg (small noise): {l_ppg(ppg_a, ppg_b, mask_mel).item():.4f}")
