"""
nsvb/utils/transfer_weights.py
================================

【這支檔案做什麼】
從 NSVB pretrained ckpt 把 FVAE backbone weights 遷移到我們的 SVBVAEZh model。
省下從零訓 Stage 1 CVAE 的 ~1 週 GPU 時間。

支援的來源：
    - NSVB `1030_vae_mle/model_ckpt_steps_*.ckpt`：英文歌聲 200k 步 CVAE pretrained

【為什麼能 transfer（即使我們是中文 + 不同 condition source）】
NSVB 1030 ckpt 的 vae_model 結構：
    encoder.pre_net    Conv1d(80→192, kernel=8, stride=4)     # mel → hidden
    encoder.wn         WaveNet residual stack
    encoder.out_proj   Conv1d(192→256, kernel=1)              # → (mean, log_std)*128
    decoder.pre_net    ConvTranspose1d(128→192, kernel=4, stride=4)
    decoder.wn         WaveNet residual stack
    decoder.out_proj   Conv1d(192→80, kernel=1)               # hidden → mel
    g_pre_net          Conv1d(256→256, kernel=8, stride=4)    # condition 下採

驗證 (compare_keys 函式)：我們 SVBVAEZh.fvae 的 88 個 keys 全部與 NSVB
vae_model.* 一一對應（shape 完全相同）；NSVB 多 16 個 encoder.poolings.* 是
prior_glow / attention pooling 殘留，我們的 architecture 不用，丟棄即可。

【哪些 weights 適合 transfer】
✅ encoder.pre_net / wn / out_proj   ← mel→hidden 是 language-agnostic
✅ decoder.pre_net / wn / out_proj   ← 同理
🟡 g_pre_net                          ← 形狀對但 condition source 不同（NSVB 是 FastSpeech2
                                          hidden, ours 是 PPG+F0+spk）。仍 transfer 後 fine-tune
                                          也比 random init 收斂快
❌ encoder.wn.cond_layer / decoder.wn.cond_layer
                                       ← 這些把 condition g 投影到 WN 各層，weights 假設
                                          NSVB 的 condition 分布。雖然 shape 對，但語意完全
                                          不同，**這層強烈建議 reset 為 random init**

【但在實作上：先全 transfer，再以小 lr fine-tune】
完全 reset cond_layer 損失某些泛化（很多 mel→hidden 的學習仍可借）；
全 transfer + small lr 讓 cond_layer 自己調整，實證上比 partial reset 好。
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def load_nsvb_vae_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    從 NSVB 1030_vae_mle ckpt 抽出 vae_model.* 子集，去 prefix 後回傳。

    為什麼回傳 stripped state_dict 而不是直接 load：
      讓 caller 看到 keys 後再決定怎麼餵給目標 model（彈性）；
      也便於先 print 對齊狀況、debug 用。

    Returns:
        state: {'encoder.pre_net.0.weight': ..., 'g_pre_net.0.weight': ..., ...}
        對應 SVBVAEZh.fvae 的命名空間
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected ckpt structure: {type(sd)}")

    # 抓 vae_model.* keys 並去 prefix
    out = {}
    for k, v in sd.items():
        if k.startswith("vae_model."):
            out[k[len("vae_model."):]] = v
    if not out:
        raise ValueError(f"No 'vae_model.*' keys in ckpt {ckpt_path}; "
                         f"maybe wrong ckpt? Found prefixes: "
                         f"{set(k.split('.')[0] for k in sd.keys())}")
    return out


def transfer_to_svbvae(model: nn.Module, ckpt_path: str,
                       verbose: bool = True) -> Tuple[int, int, int]:
    """
    把 NSVB ckpt 的 vae_model.* 注入到 SVBVAEZh.fvae（model.fvae）。

    Args:
        model:     SVBVAEZh instance
        ckpt_path: NSVB 1030_vae_mle ckpt 路徑

    Returns:
        (n_loaded, n_skipped_shape, n_skipped_missing)
        n_loaded:           成功載入的 weight 數
        n_skipped_shape:    名字相符但 shape 不對的數量（理論上應該 0）
        n_skipped_missing:  我們有但 ckpt 沒有的 keys 數（會用 random init）

    為什麼用 strict=False 風格手動 load 而非直接 load_state_dict：
      load_state_dict(strict=False) 對 mismatch 只 warning 不 raise；
      我們手動載：對齊不上的明確報出，可以審視；同時對齊得上的 100% load 進去
    """
    if not hasattr(model, "fvae"):
        raise AttributeError("model has no .fvae attribute; expected SVBVAEZh")

    nsvb_state = load_nsvb_vae_state(ckpt_path)
    target_state = model.fvae.state_dict()

    n_loaded = 0
    n_skipped_shape = 0
    skipped_shape_keys = []

    new_state = {}
    for k in target_state.keys():
        if k in nsvb_state:
            if nsvb_state[k].shape == target_state[k].shape:
                new_state[k] = nsvb_state[k]
                n_loaded += 1
            else:
                # shape 不符（理論上不該發生，因為前面已驗證）
                new_state[k] = target_state[k]  # 保持 random init
                n_skipped_shape += 1
                skipped_shape_keys.append((k, target_state[k].shape, nsvb_state[k].shape))
        else:
            new_state[k] = target_state[k]  # 保持 random init

    n_skipped_missing = len(target_state) - n_loaded - n_skipped_shape

    # strict=True 因為 new_state 與 target 同 keys / 同 shape
    model.fvae.load_state_dict(new_state, strict=True)

    if verbose:
        print(f"[transfer] {ckpt_path}")
        print(f"  total keys in target SVBVAEZh.fvae: {len(target_state)}")
        print(f"  loaded from NSVB:    {n_loaded}")
        print(f"  shape mismatches:    {n_skipped_shape}")
        print(f"  not in NSVB ckpt:    {n_skipped_missing}")
        if skipped_shape_keys:
            print("  shape mismatch detail (kept random init):")
            for k, ours, theirs in skipped_shape_keys[:5]:
                print(f"    {k}: ours={ours} vs theirs={theirs}")

    return n_loaded, n_skipped_shape, n_skipped_missing


if __name__ == "__main__":
    # 自我測試 / CLI：載入 NSVB 1030_vae_mle ckpt 注入到我們 SVBVAEZh，驗證有多少 weight 對得上
    # 用法：python -m nsvb.utils.transfer_weights /path/to/model_ckpt_steps_200000.ckpt
    import sys
    from nsvb.model.svb_vae_zh import SVBVAEZh

    if len(sys.argv) < 2:
        print("Usage: python -m nsvb.utils.transfer_weights <nsvb_ckpt_path>")
        print("  例：python -m nsvb.utils.transfer_weights "
              "checkpoints/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt")
        sys.exit(2)

    ckpt_path = sys.argv[1]
    model = SVBVAEZh(num_mels=80, ppg_dim=1280, spk_emb_dim=256)
    n_loaded, n_shape_mm, n_missing = transfer_to_svbvae(model, ckpt_path)
    print(f"\nResult: loaded={n_loaded}, shape_mm={n_shape_mm}, missing={n_missing}")