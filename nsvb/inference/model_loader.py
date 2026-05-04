"""
nsvb/inference/model_loader.py
================================

【這支檔案做什麼】
從 Stage 2 ckpt（內含對 Stage 1 ckpt 的引用）載入推理需要的全部模型：
    SVBVAEZh (CVAE backbone, frozen)
    ResidualM (mapping network, frozen)
    HifiGanNSFGenerator (vocoder, frozen)
回傳一個 InferenceModels 容器，方便 pipeline 函式直接拿來用。

【為什麼把三個模型一起載入】
推理路徑永遠是 encoder → M → decoder → vocoder，三者必須一起就位才能跑。
分散在不同 loader 函式只會增加 caller 的協調成本。Vocoder 雖獨立 ckpt，
但同樣是「推理必要」資源，包進來統一管理。

【Stage 2 ckpt 的結構（與 stage2.save_ckpt 對應）】
    {
        "step": int,
        "M":        state_dict,
        "D_z":      state_dict,    (推理不用)
        "D_mel":    state_dict,    (推理不用)
        "patchnce": state_dict,    (推理不用)
        "opt_*":    state_dict,    (推理不用)
        "config":   Stage2Config 的 __dict__,
        "stage1_ckpt": str (絕對或相對路徑指回 Stage 1 ckpt),
    }

【為什麼 Stage 2 ckpt 不直接含 Stage 1 weight】
- ckpt 容量考量：Stage 1 CVAE ≈ 30 MB，每 5000 步存一次 Stage 2 ckpt 內含 CVAE
  會浪費磁碟（Stage 1 是 frozen，永遠不變）
- 兩階段獨立部署：可換 Stage 1 ckpt 而不用重訓 Stage 2（理論上）
- 推理時透過 stage1_ckpt 路徑回頭載 → 一次組裝完整 forward 路徑
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from nsvb.model.svb_vae_zh import SVBVAEZh
from nsvb.model.m_mapping import ResidualM


@dataclass
class InferenceModels:
    """推理所需的全套 frozen 模型。

    為什麼用 dataclass：
      pipeline 函式只需 `models.cvae`, `models.M`, `models.vocoder`，
      不用記住 dict key string 拼錯時 silent 失敗的問題。
    """
    cvae: SVBVAEZh
    M: ResidualM
    vocoder: Callable                              # (mel [1,80,T], f0 [1,T]) → wav [1, T*hop]
    device: torch.device
    config: dict                                    # Stage2Config dict


def _load_vocoder(
    vocoder_ckpt: Path,
    device: torch.device,
) -> Callable:
    """
    載入 HifiGAN-NSF vocoder（與 vocoder_identity_test.load_vocoder 同一套邏輯）。

    為什麼不直接 import scripts/vocoder_identity_test：
      script 檔不應作為 library 來 import（沒有 nsvb. namespace、可能有 sys.exit
      副作用）；把這 ~10 行邏輯複製過來，inference 模組自包含。
    """
    from nsvb.backbone.vocoder import HifiGanNSFGenerator

    state = torch.load(str(vocoder_ckpt), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    if "model_gen" in sd:
        sd = sd["model_gen"]

    g = HifiGanNSFGenerator(c_out=1, num_mels=80, audio_sample_rate=22050)
    g.load_state_dict(sd, strict=True)
    g.to(device).eval()
    g.remove_weight_norm()  # 推理移除 weight_norm 加速 ~10%

    @torch.no_grad()
    def _vocode(mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [1, NUM_MELS, T_mel]
            f0:  [1, T_mel]   Hz, 連續值 (0=unvoiced 但已 log-interp)
        Returns:
            wav: [1, T_audio]  T_audio = T_mel * 128
        """
        wav = g(mel, f0)            # [1, 1, T_audio]
        return wav.squeeze(1)        # [1, T_audio]

    return _vocode


def load_inference_models(
    stage2_ckpt: Path,
    vocoder_ckpt: Path,
    device: Optional[str] = None,
    stage1_ckpt: Optional[Path] = None,
) -> InferenceModels:
    """
    主入口：載入 Stage 2 + Stage 1 + Vocoder。

    Args:
        stage2_ckpt:  Stage 2 ckpt 路徑（含 M weight + 對 Stage 1 路徑的引用）
        vocoder_ckpt: HifiGAN-NSF vocoder ckpt
        device:       'cuda' / 'cpu' / None (auto)
        stage1_ckpt:  覆寫 Stage 2 ckpt 內紀錄的 Stage 1 路徑
                       （訓練機與推理機路徑不同時很有用，例如訓練在 Linux 但 demo 在 Windows）

    Returns:
        InferenceModels（cvae / M / vocoder 全 eval mode、frozen）

    為什麼 stage1_ckpt 是 optional override：
      Stage 2 ckpt 內存的 stage1_ckpt 路徑是訓練時的絕對 / 相對路徑，遷移到別台
      機器後可能找不到檔案；提供 override 讓使用者明確指路。
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Step 1: 載 Stage 2 ckpt ──
    s2 = torch.load(str(stage2_ckpt), map_location="cpu", weights_only=False)
    cfg = s2.get("config", {})

    # ── Step 2: 解析 Stage 1 ckpt 位置 ──
    if stage1_ckpt is None:
        s1_path_str = s2.get("stage1_ckpt") or cfg.get("stage1_ckpt")
        if not s1_path_str:
            raise ValueError(
                "Stage 2 ckpt 沒有 stage1_ckpt 欄位且未提供 stage1_ckpt 參數；"
                "無法找到 CVAE backbone weight"
            )
        stage1_ckpt = Path(s1_path_str)
    if not stage1_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 1 ckpt not found: {stage1_ckpt}\n"
            f"提示：Stage 2 ckpt 內紀錄的路徑是訓練機上的 path；"
            f"在不同機器推理時用 stage1_ckpt 參數覆寫"
        )

    # ── Step 3: 構建 SVBVAEZh + 載入 Stage 1 weight ──
    cvae = SVBVAEZh(
        num_mels=cfg.get("num_mels", 80),
        ppg_dim=cfg.get("ppg_dim", 1280),
        spk_emb_dim=cfg.get("spk_emb_dim", 256),
        latent_size=cfg.get("latent_size", 128),
        hidden_size=cfg.get("hidden_size", 192),
        enc_n_layers=cfg.get("enc_n_layers", 8),
        dec_n_layers=cfg.get("dec_n_layers", 4),
    ).to(device_t)
    s1 = torch.load(str(stage1_ckpt), map_location="cpu", weights_only=False)
    cvae.load_state_dict(s1["model"], strict=True)
    cvae.eval()
    for p in cvae.parameters():
        p.requires_grad = False
    print(f"[infer-loader] Stage 1 CVAE loaded from {stage1_ckpt}", flush=True)

    # ── Step 4: 構建 ResidualM + 載入 Stage 2 M weight ──
    M = ResidualM(
        latent_dim=cfg.get("latent_size", 128),
        hidden_dim=cfg.get("m_hidden_dim", 256),
        kernel_size=cfg.get("m_kernel_size", 1),
        num_layers=cfg.get("m_num_layers", 4),
        init_delta_scale=cfg.get("m_init_delta_scale", 1e-2),
    ).to(device_t)
    M.load_state_dict(s2["M"], strict=True)
    M.eval()
    for p in M.parameters():
        p.requires_grad = False
    print(f"[infer-loader] Stage 2 M loaded from {stage2_ckpt} (step={s2.get('step', '?')})",
          flush=True)

    # ── Step 5: 載 Vocoder ──
    vocoder = _load_vocoder(vocoder_ckpt, device_t)
    print(f"[infer-loader] Vocoder loaded from {vocoder_ckpt}", flush=True)

    return InferenceModels(
        cvae=cvae, M=M, vocoder=vocoder,
        device=device_t, config=cfg,
    )