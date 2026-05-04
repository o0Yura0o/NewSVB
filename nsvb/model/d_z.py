"""
nsvb/model/d_z.py
===================

【這支檔案做什麼】
Stage 2 的 z 層判別器 D_z。輸入：
    z:              [B, latent_dim, T_z]
    soft_register:  [B, T_z, K=5]    (從 audio 抽 F0 算的軟 bucket，downsampled 到 z 的 frame rate)
    phoneme_id:     [B, T_z] long    (cluster_ppg 算的離散音素 id, -1=padding/unknown)

輸出：
    per-frame critic score [B, 1, T_z]    (PatchGAN 風格)

【為什麼 D_z 要 conditional】
無條件 D_z 容易讓 M 用「全域共鳴濾波」攻擊（把所有業餘 mel 套上「最高分 pro 技術指紋」
例如低音也加頭腔共鳴）。conditional D_z 強制 M 在「相同聲區、相同音素」context 下
比較，逼 M 學到「對的位置做對的修飾」。

條件選擇：
    - Soft register（5 buckets, σ=0.3 log-Hz）：聲區資訊，防 F0 shortcut
    - Phoneme ID（k-means 200 cluster）：音素 context，防 PPG 細節洩漏

【為什麼 condition 從 channel 軸 concat 而非 projection】
- Channel concat 讓 condition 參與每個 conv 感受野
- Projection（Miyato 2018）對 class label 有效，但 register 是連續軟向量、phoneme 是
  embedding 向量，不適合 projection trick；channel concat 更通用

【為什麼用 spectral_norm + LeakyReLU(0.2) + hinge loss（沒寫在這裡，loss 在 losses.py）】
- spectral_norm 限制 D 的 Lipschitz，穩定 GAN 訓練
- LeakyReLU(0.2) 讓 D 在負值區仍有梯度，避免 dying ReLU
- hinge loss 在訓練中後期（D 強時）給 G 更穩 gradient（vs. BCE 的 saturating）
- 上述 3 件事是 NSVB ver1 的成熟組合，移植過來

【為什麼最後 head 是 1x1 conv 出 [B, 1, T_z]】
- PatchGAN 風格：每個 z frame 各自評一個分數
- 對應 hinge loss 直接 .mean() over T_z（每 frame 同等權重）
- 比 global pooling + scalar 出單一分數，提供更密的 gradient signal
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscriminatorZ(nn.Module):
    """
    Args:
        latent_dim:           z 的通道數（128）
        soft_register_dim:    K（5）
        phoneme_vocab_size:   k-means cluster K（200，含 padding=0 用 -1 sentinel 額外處理）
        phoneme_embed_dim:    phoneme embedding 向量維度（32）
        hidden_dim:           內部 conv 通道（256）
        num_layers:           中間層數（4，最後再加 1x1 head 出 logit）
        kernel_size:          conv 感受野（5；對應 z frame rate 43 fps，5 frame ≈ 116 ms 局部）

    為什麼 kernel=5：
        43 fps × 5 frames ≈ 116 ms 感受野，正好覆蓋 1-2 個音素 transition；
        kernel=3 太短只看單音素，kernel=7 過大會混太多上下文
    """

    def __init__(self,
                 latent_dim: int = 128,
                 soft_register_dim: int = 5,
                 phoneme_vocab_size: int = 200,
                 phoneme_embed_dim: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 kernel_size: int = 5):
        super().__init__()

        # 為什麼 phoneme embedding 多預留 1 個 slot 給 padding：
        #   phoneme_id 來自 cluster_ppg，正常範圍 [0, K-1]；padding / unvoiced / 未知用 -1
        #   forward 時把 -1 clip 成 padding_idx=0 並讓 embedding(0) 學成「無 phoneme info」
        # 所以 vocab 多 1 slot：[0=PAD, 1..K]
        # 但 cluster_ppg 預設輸出 [0, K-1]，要在 forward 把所有 id 偏移 +1
        self.phoneme_embed = nn.Embedding(
            num_embeddings=phoneme_vocab_size + 1,
            embedding_dim=phoneme_embed_dim,
            padding_idx=0,
        )

        cond_dim = soft_register_dim + phoneme_embed_dim
        padding = kernel_size // 2

        # 中間層 stack
        layers = []
        in_ch = latent_dim + cond_dim
        for _ in range(num_layers):
            layers.append(spectral_norm(
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=padding)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = hidden_dim

        # 最後 head：1x1 conv 出 per-frame logit
        layers.append(spectral_norm(nn.Conv1d(in_ch, 1, kernel_size=1)))
        self.net = nn.Sequential(*layers)

    def forward(self,
                z: torch.Tensor,
                soft_register: torch.Tensor,
                phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:              [B, latent_dim, T_z]
            soft_register:  [B, T_z, K]
            phoneme_ids:    [B, T_z] long, 範圍 [-1, vocab_size-1]
                            (-1 = padding / unvoiced / unknown)

        Returns:
            score: [B, 1, T_z]   per-frame critic logit
        """
        # phoneme_ids: -1 → 0 (padding_idx), [0, K-1] → [1, K]
        # 為什麼 +1 偏移：embedding 0 留給 padding，所有有效 id +1 平移
        ph_ids_shifted = (phoneme_ids + 1).clamp(min=0)
        ph_emb = self.phoneme_embed(ph_ids_shifted)             # [B, T_z, P]

        # condition concat: [B, T_z, K+P]
        cond = torch.cat([soft_register, ph_emb], dim=-1)
        cond = cond.transpose(1, 2)                              # [B, K+P, T_z]

        # z + cond concat along channel
        x = torch.cat([z, cond], dim=1)                          # [B, latent+cond, T_z]
        return self.net(x)


if __name__ == "__main__":
    # 自我測試：shape + spectral_norm 是否生效
    B, C_z, T_z = 4, 128, 80
    K_REG = 5
    K_PH = 200
    P_EMB = 32

    D = DiscriminatorZ(
        latent_dim=C_z, soft_register_dim=K_REG,
        phoneme_vocab_size=K_PH, phoneme_embed_dim=P_EMB,
        hidden_dim=256, num_layers=4, kernel_size=5,
    )
    n_params = sum(p.numel() for p in D.parameters())
    print(f"D_z params: {n_params/1e6:.3f}M")

    z = torch.randn(B, C_z, T_z)
    soft_register = torch.softmax(torch.randn(B, T_z, K_REG), dim=-1)
    # 部分 frames 用 -1 模擬 padding / unvoiced
    phoneme_ids = torch.randint(0, K_PH, (B, T_z))
    phoneme_ids[:, ::5] = -1  # 每 5 個 frame 1 個 padding

    score = D(z, soft_register, phoneme_ids)
    print(f"score shape: {score.shape}  (expect [{B},1,{T_z}])")
    print(f"score range: [{score.min().item():.3f}, {score.max().item():.3f}]")

    # 驗證 spectral_norm 生效：應該有 weight_orig / weight_u 等內部 buffer
    sn_count = sum(1 for n, _ in D.named_buffers() if "weight_u" in n)
    print(f"spectral_norm-applied conv layers: {sn_count}  (expect 5)")
