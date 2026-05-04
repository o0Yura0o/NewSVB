"""
nsvb/model/m_mapping.py
=========================

【這支檔案做什麼】
Stage 2 的 mapping function M：把業餘 latent z_a 映射到「pro 風格」latent z_a'。
殘差結構：f(z) = z + Δ(z)，初始化 Δ 近零，所以 M 起步 ≈ identity。

【為什麼用 residual + kernel=1】
1. **Warp-invariant**：kernel=1 的 pointwise conv 對時間軸上任何 reorder 都是同樣處理；
   推理 Mode B 中 z_a' 經過 DTW + gather warp 後，再過一次 M 不會引入時間錯位
   （M 只看單 frame，不需要 temporal context）
2. **Identity-preserving**：殘差結構讓 M 在訓練早期 ≈ identity，逐漸學「Δ 偏移」；
   防止 M 一開始就把 z 推到亂七八糟的地方破壞訓練
3. **可選 kernel=3**：若訓練到 30k 步 ‖Δ‖/‖z‖ 仍 < 3%，代表 M 太保守，
   切 kernel=3 引入 local context（仍是窄感受野，與 warp 部分相容）

【為什麼最後一層 conv 初始化 std=1e-2】
- Δ 起點要小，但不能完全 0（不然第一次 backprop grad 全 0 學不動）
- std=1e-2 讓 Δ(z) 的初始輸出 magnitude 約 z 的 1%，幾乎是 identity 但有一點 signal
- 來自 NSVB-ZH ver1 的 init_delta_scale 慣例

【為什麼用 GroupNorm 而非 BatchNorm】
- BatchNorm 對 batch 統計敏感，z 在不同 batch 統計差會晃
- GroupNorm 不依賴 batch，更穩；NSVB ver1 也選此

【為什麼最後一層之前每層 GELU 而非 ReLU】
- z 是 mean-centered Gaussian-ish 分布（VAE 性質），GELU 平滑通過 0 點
- ReLU 在負值區直接 0，破壞 z 分布的對稱性
"""

import torch
import torch.nn as nn


class _ChannelLayerNorm(nn.Module):
    """
    LayerNorm over channel dim 給 1D conv tensor [B, C, T] 用。

    為什麼自寫：
        torch.nn.LayerNorm 預設對最後一個 dim normalize，對 [B, C, T] 是 normalize T；
        我們要 normalize C，所以自寫 transpose-style version。
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class ResidualM(nn.Module):
    """
    Residual mapping function M: z → z + Δ(z)。

    Args:
        latent_dim:        z 的通道數（NSVB 預設 128）
        hidden_dim:        Δ 內部隱通道（256）
        kernel_size:       1（pointwise, warp-invariant）或 3（更廣感受野，輕微犧牲 warp invariance）
        num_layers:        Δ 網路深度（4 with hidden→hidden→hidden→latent_dim）
        norm:              "group" / "layer"
        init_delta_scale:  最後一層 conv weight 初始化 std（1e-2）
    """

    def __init__(self,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 kernel_size: int = 1,
                 num_layers: int = 4,
                 norm: str = "group",
                 init_delta_scale: float = 1e-2):
        super().__init__()
        assert kernel_size in (1, 3), "kernel_size must be 1 (warp-invariant) or 3"
        assert num_layers >= 2, "num_layers must be >= 2 (in + out)"

        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        layers = []
        in_ch = latent_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding))
            if i < num_layers - 1:
                # 中間層接 norm + activation；最後一層只 conv（讓 Δ 直接出 latent）
                if norm == "group":
                    # 為什麼 min(8, out_ch)：若 hidden=256，groups=8 每組 32 通道；
                    # 通道少時 cap 在 out_ch 避免 groups > channels
                    groups = min(8, out_ch)
                    layers.append(nn.GroupNorm(groups, out_ch))
                elif norm == "layer":
                    layers.append(_ChannelLayerNorm(out_ch))
                else:
                    raise ValueError(f"Unknown norm: {norm}")
                layers.append(nn.GELU())
            in_ch = out_ch

        self.delta = nn.Sequential(*layers)

        # 為什麼把最後一層 weight 初始化成 N(0, 1e-2)：
        #   讓 Δ(z) 起點 magnitude 約 ‖z‖ × 1e-2，M ≈ identity；
        #   bias 設 0 確保 voiced/unvoiced 邊界沒有 systematic shift
        with torch.no_grad():
            last_conv = self.delta[-1]
            assert isinstance(last_conv, nn.Conv1d)
            nn.init.normal_(last_conv.weight, mean=0.0, std=init_delta_scale)
            nn.init.zeros_(last_conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, T_z]
        Returns:
            z + Δ(z): [B, latent_dim, T_z]
        """
        return z + self.delta(z)

    def delta_only(self, z: torch.Tensor) -> torch.Tensor:
        """
        只回傳 Δ(z)，用於監控 ‖Δ‖ / ‖z‖ ratio。

        為什麼提供這個：
          訓練 log 期望看「M 漂移程度」，定義為 ‖Δ(z)‖ / ‖z‖；
          直接 forward 拿不到 Δ（已加進 z），開這個 helper 監控用
        """
        return self.delta(z)


if __name__ == "__main__":
    # 自我測試：驗證 init 時 M ≈ identity，shape 正確
    B, C, T = 4, 128, 80
    M = ResidualM(latent_dim=C, hidden_dim=256, kernel_size=1, num_layers=4)
    n_params = sum(p.numel() for p in M.parameters())
    print(f"ResidualM params: {n_params/1e6:.3f}M")

    z = torch.randn(B, C, T)
    z_mapped = M(z)
    delta = M.delta_only(z)
    print(f"z shape:       {z.shape}")
    print(f"z + Δ shape:   {z_mapped.shape}")
    print(f"Δ shape:       {delta.shape}")
    print(f"‖Δ‖ / ‖z‖:    {(delta.norm() / z.norm()).item():.4f}  (init 期望 < 0.05)")
    print(f"‖z + Δ - z‖:  {(z_mapped - z - delta).abs().max().item():.6e}  (應該 0)")