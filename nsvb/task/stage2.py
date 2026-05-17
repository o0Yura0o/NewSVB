"""
nsvb/task/stage2.py
=====================

【這支檔案做什麼】
NSVB-ZH Phase 2 — Stage 2 mapping training task。
凍結 Stage 1 訓好的 SVBVAEZh (φ/θ)，初始化 ResidualM 與 DiscriminatorZ，
復用 Stage 1 D_mel（real 改為只看 pro），訓練 M 把業餘 z 映射到 pro 風格。

【訓練流程（每 step）】
  1. 各自抽 amateur batch_a / pro batch_p（兩個獨立 dataloader）
  2. 凍結 φ 計算 z_a, z_p
  3. 用同 stride 把 register / phoneme 從 mel rate 下採到 z 的 frame rate
  4. 更新 D_z：real=z_p (cond reg_p, ph_p), fake=M(z_a).detach() (cond reg_a, ph_a)
  5. 更新 D_mel（low LR）：real=mel_p, fake=θ(M(z_a)) decode 出的 mel
  6. 更新 M（同時更新 PatchNCE 投影 head）：
       L_adv_z          (D_z 對 M(z_a) 給高分；warmup 前不開)
       L_PatchNCE(z_a, M(z_a))
       L_adv_mel * 0.2  (D_mel 對 decode(M(z_a)) 給高分)
       L_identity_pro * 0.1  (20% batches 抽中時)
       L_PPG            （TODO: 待 fast mel→PPG predictor 完成後啟用）

【為什麼凍結 φ/θ】
- Stage 1 已經把 CVAE 訓好（mel↔z 映射穩定）
- Stage 2 動 M 即可達成「修飾 z」目標；連 φ/θ 一起更新會讓 z 分布漂移，
  D_z 訊號變不穩，難以收斂

【為什麼 D_z warmup 5000 步不傳梯度給 M】
- D_z 是 from scratch 初始化，前期判斷不準（real/fake 都隨機猜）；
  讓 D_z 自己先學 5000 步穩定後，再開放 G 接收 D_z 訊號
- M 在 warmup 期還是訓的（PatchNCE + adv_mel + identity_pro 給 signal），
  學的是「近恆等修飾」（M ≈ identity 微調）

【TTUR (Heusel 2017)】
- opt_M  = 1e-4
- opt_Dz = 4e-4 (4x M)
- opt_Dmel = 1e-5 (微調)
給 D_z headroom 學識別，M 才有清晰梯度往對的方向走

【為什麼有兩個 DataLoader (amateur, pro)】
- 兩 dataset 樣本數可能不同，混在一起隨機抽會 bias 大 dataset
- 各自獨立 sampler 讓每個 batch step 都拿到 ~equal mix
- itertools.cycle 把兩邊各自無限 cycling

【L_PPG 為什麼 TODO】
- 在 training loop 中跑 Whisper-large-v3 forward 太慢（~3 sec/batch，訓練拖累 10x+）
- 需要一個輕量 mel→PPG predictor（mel 直接預測 1280-d PPG）
- 暫時 lambda_ppg=0；後續若觀察到 PatchNCE 不夠鎖內容再加
- PatchNCE + L_adv_mel + L_identity_pro 三層內容保護 在 NSVB ver1 已經 work
"""

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nsvb.data.dataset import BinarizedNSVBDataset, collate_fn
from nsvb.model.svb_vae_zh import SVBVAEZh
from nsvb.model.m_mapping import ResidualM
from nsvb.model.d_z import DiscriminatorZ
from nsvb.model.losses import (
    hinge_d_loss, hinge_g_loss, PatchNCELoss,
    l_identity_pro,
)
from nsvb.backbone.multi_window_disc import Discriminator as DMel


# ── 工具：條件 mel rate → z rate downsample ─────────────
def downsample_register(register_soft: torch.Tensor, T_z: int) -> torch.Tensor:
    """
    register_soft [B, T_mel, K=5] → [B, T_z, K]，用 linear interpolate。

    為什麼用 interpolate 而非 avg_pool：
        T_mel / 4 不一定恰好 = T_z（pre_net Conv1d 計算 floor((T-4)/4)+1）；
        interpolate(size=T_z) 自動對齊到 z 的精確 frame 數
    """
    x = register_soft.transpose(1, 2)  # [B, K, T_mel]
    x = F.interpolate(x, size=T_z, mode="linear", align_corners=False)
    return x.transpose(1, 2)  # [B, T_z, K]


def downsample_phoneme_id(phoneme_id: torch.Tensor, T_z: int,
                          stride: int = 4) -> torch.Tensor:
    """
    phoneme_id [B, T_mel] long → [B, T_z]，用 stride sampling。

    為什麼 stride 而非 mode pool：
        phoneme_id 是 categorical，連續 stride 對應的就是 cluster 主導 frame，
        簡單也夠 representative；mode pool 額外計算成本但收益小

    為什麼處理長度不對齊：
        stride 後可能 ±1 frame 與 T_z 不符，pad/trim 確保一致
    """
    pid_z = phoneme_id[:, ::stride]
    if pid_z.shape[1] > T_z:
        pid_z = pid_z[:, :T_z]
    elif pid_z.shape[1] < T_z:
        pad_amt = T_z - pid_z.shape[1]
        pid_z = F.pad(pid_z, (0, pad_amt), value=-1)
    return pid_z


def downsample_mask(mel_mask: torch.Tensor, T_z: int) -> torch.Tensor:
    """mel_mask [B, T_mel] → z_mask [B, T_z]，用 nearest neighbour（mask 是 binary）。"""
    x = mel_mask.unsqueeze(1)  # [B, 1, T_mel]
    x = F.interpolate(x, size=T_z, mode="nearest")
    return x.squeeze(1)


# ── Config ───────────────────────────────────────────────
@dataclass
class Stage2Config:
    """Stage 2 hyperparameters。可被 CLI override。"""
    # ── 資料 ──
    binarized_root: str = "data/binarized"
    amateur_dataset: str = "vocalverse"   # amateur 那邊
    pro_dataset: str = "m4singer"          # pro 那邊
    # 為什麼 1500：VocalVerse 5s chunks ≈ 860 frames + M4Singer 9s outlier ≈ 1500
    # 涵蓋兩 dataset 所有樣本不觸發 random crop（隨 binarize 端 chunk VV 同步調整）
    max_frames: int = 1500
    batch_size: int = 16
    num_workers: int = 4

    # ── 模型 ──
    num_mels: int = 80
    ppg_dim: int = 1280
    spk_emb_dim: int = 256
    latent_size: int = 128
    hidden_size: int = 192
    enc_n_layers: int = 8
    dec_n_layers: int = 4

    # ── M ──
    m_kernel_size: int = 1
    m_hidden_dim: int = 256
    m_num_layers: int = 4
    m_init_delta_scale: float = 1e-2

    # ── D_z ──
    register_dim: int = 5
    phoneme_vocab_size: int = 200
    phoneme_embed_dim: int = 32
    dz_hidden_dim: int = 256
    dz_num_layers: int = 4
    dz_kernel_size: int = 5

    # ── PatchNCE ──
    patchnce_proj_dim: int = 64
    patchnce_temp: float = 0.07
    patchnce_num_patches: int = 128

    # ── Loss 權重 ──
    lambda_adv_z: float = 1.0
    lambda_patchnce: float = 1.0
    lambda_adv_mel: float = 0.2
    lambda_identity_pro: float = 0.1
    identity_pro_prob: float = 0.2
    lambda_ppg: float = 0.0  # TODO: 實作 fast mel→PPG predictor 後啟用

    # ── D_mel real source（Risk 2 fallback）──
    # 預設 False：D_mel real 只看 pro mel（rebuild_checklist §C 設計，把 D_mel 從
    #            「自然度」升級成「pro 自然度」，提供 mel 層 pro-direction 推力）
    # 設 True 啟用 fallback：D_mel real 改餵 pro+amateur 混合（每 step 兩邊各半），
    #            退化成 ver1 的「自然度判別器」，犧牲 mel 層 pro-direction 訊號換取
    #            「絕不鼓勵去殘響」的安全
    # 何時啟用：訓中 monitor_audio_quality 連續兩次顯示 unvoiced_concentration > 0.65
    #          時 resume + 啟此 flag（救火，不是預設）。
    # 前提：Risk 2 L2（dereverb 對兩邊都做）正確執行 → D_mel pro-only 安全。
    #       若無法保證 dereverb，從一開始就建議啟此 flag
    dmel_mix_amateur_real: bool = False

    # ── D_mel freeze（v2 redo:Risk §二.3 amateur F0 conditioning confound 救火）─
    # 為什麼 freeze:fake mel decoder condition 用 f0_a(業餘),real mel 來自 pro,
    #   → D_mel 容易用「f0_a 軌跡不像 pro」當捷徑判 fake,造成 l_adv_mel 結構性 floor,
    #   M 為了縮這個 floor 改去動 latent 時間結構(觀察:Δ/z ~0.93、tdr ~0.77,
    #   兩者都嚴重超出 training_flow §3.6.1 的健康範圍 [0.03, 0.30] / < 0.3)。
    # 凍 D_mel + 把 λ_adv_mel 從 0.2 降到 0.05:把 mel 層 adversarial pressure 削弱
    #   到只剩「prior signal」量級,M 主要從 PatchNCE + L_adv_z(latent space)學,
    #   不再被 D_mel 的 F0 捷徑帶偏。
    # 副作用:D_mel 不再更新,失去「跟隨 M 進步」的能力 → 訓練後期 D_mel 可能變弱;
    #   但本次目標是「先讓 M 不亂改 latent」,自然度先暫時不靠 D_mel 推。
    freeze_d_mel: bool = False

    # ── Optimizer (TTUR) ──
    lr_m: float = 1e-4
    lr_dz: float = 4e-4
    lr_dmel: float = 1e-5
    beta1: float = 0.5     # NSVB GAN 慣例
    beta2: float = 0.999

    # ── Warmup / Schedule ──
    d_z_warmup_steps: int = 5000

    # ── 訓練 ──
    max_steps: int = 120000
    log_interval: int = 50
    save_interval: int = 5000
    grad_clip: float = 1.0

    # ── 音質監控（Risk 2 補強 P2）─────────────
    # 為什麼提供：訓練 loss 數字看不出 M 是否走「修錄音環境」捷徑；
    # 隔幾千步抽幾個 amateur 樣本算「M 修飾的 spectral 分布」是否健康
    audio_quality_monitor_interval: int = 5000  # 每 N 步跑一次
    audio_quality_monitor_n_samples: int = 4    # 每次抽幾個 sample
    audio_quality_log_dir: str = "checkpoints/stage2/audio_monitor"

    # ── 路徑 ──
    stage1_ckpt: str = "checkpoints/stage1/stage1_latest.pt"
    ckpt_dir: str = "checkpoints/stage2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Train split 過濾(避免 Stage 2 訓練吃到 hold-out test 樣本)──
    # 為什麼只接 train.txt 不接 val.txt:Stage 2 是 GAN-style 對抗訓練,沒有單一
    # 「val loss」是無爭議的最佳 ckpt 判準(D_z accuracy 漂、l_adv 是相對 score),
    # 所以這裡只用 train split 排除 test。val/test 樣本留給推理 evaluation。
    split_dir: str = "data/binarized/splits"

    # ── M 健康檢查（兩種 failure mode）─────────────────────
    # Failure mode A：M 太保守（delta_over_z 過低）— 學不出 pro-style 修飾
    # Failure mode B：M 抹平既有時間軌跡（temporal_diff_ratio 過高）— 顫音/滑音被殺
    # 30000 步後若兩條件任一觸發 → print 一次性警告
    # 為什麼 30000 步：L_adv_z 在 5000 步 warmup 結束後正式 active，再給 25000 步
    # 讓 M 學會基本 mapping；30000 仍異常表示需要介入
    health_check_step: int = 30000
    # A：delta_over_z 移動平均 < 0.03 → M 沒動 z（kernel=1 表達力不夠 → 切 kernel=3）
    delta_low_threshold: float = 0.03
    # B：temporal_diff_ratio 移動平均 > 1.0 → M 改變的時間導數量級超過 z 本身的時間
    # 導數（典型 sign of trajectory destruction）
    # 為什麼 1.0：z 的時間導數本身具有特定 magnitude；若 M(z)-z 的時間導數差距 >
    # z 自身的時間導數，等於 M 把時間結構整個重寫；< 0.3 健康
    temporal_diff_high_threshold: float = 1.0
    # 向後相容 alias（舊欄位名）
    delta_health_check_step: int = 30000
    delta_health_check_threshold: float = 0.03


# ── 工具：兩 dataloader 無限 yield ──────────────────────
def infinite_loader(dl: DataLoader) -> Iterator[dict]:
    """無限 yield batches:dl 跑完一輪後回頭重 iterate。

    ⚠ **不要**用 `itertools.cycle(dl)`:
      cycle 會把每個 yield 出來的 batch 存進內部 saved list 供下一輪 replay。
      我們在 train_step 用 `_to_device(batch)` **就地 mutate dict** 把 CPU
      tensor 換成 GPU tensor → cycle 的 cache 變成「一堆指向 GPU tensor 的 dict」
      → 每步累積 ~150MB GPU 不釋放,~200 步就把 40GB A100 吃光 OOM。

    plain `yield from dl` 不 cache,generator stack frame 只留當前 batch 的
    reference,前一個 batch 走出 stack 後 Python GC 即可回收 → 記憶體穩態。
    """
    while True:
        yield from dl


# ── Trainer ──────────────────────────────────────────────
class Stage2Trainer:
    def __init__(self, cfg: Stage2Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

        # ── Dataset：兩邊各自獨立(若 splits/train.txt 存在則只吃 train 切分)──
        split_dir = Path(cfg.split_dir) if cfg.split_dir else None
        train_split_file = None
        if split_dir is not None:
            t = split_dir / "train.txt"
            if t.exists():
                train_split_file = t
                print(f"[stage2] using train split: {t}")
            else:
                print(f"[stage2] {t} 不存在 — fallback 用全 dataset(test 樣本也會被訓到)")
        self.ds_a = BinarizedNSVBDataset(
            Path(cfg.binarized_root) / cfg.amateur_dataset,
            max_frames=cfg.max_frames, split_file=train_split_file,
        )
        self.ds_p = BinarizedNSVBDataset(
            Path(cfg.binarized_root) / cfg.pro_dataset,
            max_frames=cfg.max_frames, split_file=train_split_file,
        )
        self.dl_a = DataLoader(
            self.ds_a, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            drop_last=True, pin_memory=(cfg.device == "cuda"),
        )
        self.dl_p = DataLoader(
            self.ds_p, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            drop_last=True, pin_memory=(cfg.device == "cuda"),
        )
        self.iter_a = infinite_loader(self.dl_a)
        self.iter_p = infinite_loader(self.dl_p)

        # ── Model：CVAE 凍結，M / D_z 從零訓，D_mel 從 Stage 1 復用 ──
        self.cvae = SVBVAEZh(
            num_mels=cfg.num_mels, ppg_dim=cfg.ppg_dim, spk_emb_dim=cfg.spk_emb_dim,
            latent_size=cfg.latent_size, hidden_size=cfg.hidden_size,
            enc_n_layers=cfg.enc_n_layers, dec_n_layers=cfg.dec_n_layers,
        ).to(self.device)
        # 從 Stage 1 ckpt 載入 CVAE state（注意：載入 svb_vae_zh 全部，不只 fvae）
        s1 = torch.load(cfg.stage1_ckpt, map_location="cpu", weights_only=False)
        self.cvae.load_state_dict(s1["model"], strict=True)
        # 凍結
        for p in self.cvae.parameters():
            p.requires_grad = False
        self.cvae.eval()
        print(f"[stage2] loaded Stage 1 CVAE from {cfg.stage1_ckpt} (frozen)")

        self.M = ResidualM(
            latent_dim=cfg.latent_size,
            hidden_dim=cfg.m_hidden_dim,
            kernel_size=cfg.m_kernel_size,
            num_layers=cfg.m_num_layers,
            init_delta_scale=cfg.m_init_delta_scale,
        ).to(self.device)

        self.D_z = DiscriminatorZ(
            latent_dim=cfg.latent_size,
            soft_register_dim=cfg.register_dim,
            phoneme_vocab_size=cfg.phoneme_vocab_size,
            phoneme_embed_dim=cfg.phoneme_embed_dim,
            hidden_dim=cfg.dz_hidden_dim,
            num_layers=cfg.dz_num_layers,
            kernel_size=cfg.dz_kernel_size,
        ).to(self.device)

        # D_mel 從 Stage 1 ckpt 載入
        self.D_mel = DMel(
            time_lengths=(32, 64, 128), freq_length=cfg.num_mels,
            hidden_size=128, norm_type="bn", reduction="sum",
        ).to(self.device)
        if "d_mel" in s1:
            self.D_mel.load_state_dict(s1["d_mel"], strict=True)
            print("[stage2] loaded D_mel from Stage 1 ckpt")
        else:
            print("[stage2] WARNING: stage1 ckpt has no d_mel; D_mel from scratch")

        self.patchnce = PatchNCELoss(
            latent_dim=cfg.latent_size, proj_dim=cfg.patchnce_proj_dim,
            temperature=cfg.patchnce_temp, num_patches=cfg.patchnce_num_patches,
        ).to(self.device)

        # ── Optimizers (TTUR) ──
        # 為什麼 M + patchnce 共用 opt_m：兩者都是 G 路徑，loss 一起算 backward 一次
        m_params = list(self.M.parameters()) + list(self.patchnce.parameters())
        self.opt_m = torch.optim.Adam(m_params, lr=cfg.lr_m, betas=(cfg.beta1, cfg.beta2))
        self.opt_dz = torch.optim.Adam(self.D_z.parameters(), lr=cfg.lr_dz,
                                        betas=(cfg.beta1, cfg.beta2))
        # freeze_d_mel:不建 opt_dmel 也把 D_mel param requires_grad 關掉(forward 仍可
        # 跑、仍能對 M 提供 gradient,但本身 weight 不會被更新)
        if cfg.freeze_d_mel:
            for p in self.D_mel.parameters():
                p.requires_grad = False
            self.D_mel.eval()
            self.opt_dmel = None
        else:
            self.opt_dmel = torch.optim.Adam(self.D_mel.parameters(), lr=cfg.lr_dmel,
                                             betas=(cfg.beta1, cfg.beta2))

        self.step = 0
        n_m = sum(p.numel() for p in self.M.parameters()) / 1e6
        n_dz = sum(p.numel() for p in self.D_z.parameters()) / 1e6
        print(f"[stage2] init done: M={n_m:.2f}M, D_z={n_dz:.2f}M, "
              f"lr(M/Dz/Dmel)={cfg.lr_m:.0e}/{cfg.lr_dz:.0e}/"
              f"{'frozen' if cfg.freeze_d_mel else f'{cfg.lr_dmel:.0e}'}")
        print(f"[stage2] lambda(adv_z/adv_mel/patchnce)="
              f"{cfg.lambda_adv_z}/{cfg.lambda_adv_mel}/{cfg.lambda_patchnce}", flush=True)
        print(f"[stage2] D_mel real source: "
              f"{'pro+amateur (Risk 2 fallback)' if cfg.dmel_mix_amateur_real else 'pro only (default)'}"
              f"{' [FROZEN]' if cfg.freeze_d_mel else ''}",
              flush=True)

    # ── 共用：把一個 batch 餵 frozen CVAE.encoder 拿 z 與下採條件 ─
    def _encode_and_downsample(self, batch: dict):
        """
        Returns:
            z:                [B, latent, T_z]   from frozen φ(x)
            mel_recon_full:    [B, T_mel, NUM_MELS]   for fake mel use
            register_z:        [B, T_z, K]
            phoneme_z:         [B, T_z] long
            mask_z:            [B, T_z]
            mel:               [B, T_mel, NUM_MELS]   原始
            mel_mask:          [B, T_mel]
            spk_emb / ppg / f0: 維持原樣（decoder 要）
        """
        with torch.no_grad():
            out = self.cvae(batch["mel"], batch["ppg"], batch["f0"],
                            batch["spk_emb"], batch["mel_mask"], infer=False)
            # 為什麼用 m_q（posterior mean）而非採樣 z：
            #   M 訓練時 deterministic 較穩；採樣 z 引入 noise，gradient signal 變糟
            z = out["m_q"]  # [B, latent, T_z]

        T_z = z.shape[-1]
        register_z = downsample_register(batch["register_soft"], T_z)
        phoneme_z = downsample_phoneme_id(batch["phoneme_id"], T_z)
        mask_z = downsample_mask(batch["mel_mask"], T_z)
        return {
            "z": z, "register_z": register_z, "phoneme_z": phoneme_z,
            "mask_z": mask_z,
        }

    def _decode_with_mapped_z(self, z_mapped: torch.Tensor, batch: dict):
        """
        把 M(z_a) 餵回 frozen θ 解碼出 mel。
        為什麼這層不 with no_grad：M 經過 z_mapped → θ → mel，gradient 要回傳到 M
        （但 θ 本身 frozen，gradient 會 flow through it 不更新它的 weight）
        """
        # θ 內部需要 condition g（mel rate, 與 stage 1 同樣）
        g = self.cvae.condition(batch["ppg"], batch["f0"], batch["spk_emb"])
        # mask 還是 mel rate
        mel_mask_expand = batch["mel_mask"].unsqueeze(1)
        mel_recon = self.cvae.fvae.decoder(z_mapped, mel_mask_expand, g)
        # transpose 回 time-major
        return mel_recon.transpose(1, 2)  # [B, T_mel, NUM_MELS]

    def _to_device(self, batch: dict) -> dict:
        for k in ["mel", "ppg", "f0", "spk_emb", "mel_mask",
                  "register_soft", "register_id", "phoneme_id"]:
            batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch

    def train_step(self) -> dict:
        # ── 抽兩 batch ──
        ba = self._to_device(next(self.iter_a))
        bp = self._to_device(next(self.iter_p))

        # ── 凍結 CVAE 拿 z + 下採條件 ──
        side_a = self._encode_and_downsample(ba)
        side_p = self._encode_and_downsample(bp)
        z_a, z_p = side_a["z"], side_p["z"]

        # ── (1) Update D_z ──
        with torch.no_grad():
            z_a_mapped_detach = self.M(z_a)
        real_score = self.D_z(z_p, side_p["register_z"], side_p["phoneme_z"])
        fake_score = self.D_z(z_a_mapped_detach.detach(),
                              side_a["register_z"], side_a["phoneme_z"])
        d_z_loss = hinge_d_loss(real_score, fake_score)
        self.opt_dz.zero_grad()
        d_z_loss.backward()
        self.opt_dz.step()

        # ── (2) Update D_mel (real = pro mel only by default) ──
        # freeze_d_mel:整段 skip,既省 forward+backward 又省 update。D_mel 在 (3c) 仍會
        # forward(以給 M 梯度),但 weight 凍住
        d_mel_loss_val = 0.0
        if self.cfg.lambda_adv_mel > 0 and not self.cfg.freeze_d_mel:
            with torch.no_grad():
                z_a_mapped_for_mel = self.M(z_a)
                fake_mel = self._decode_with_mapped_z(z_a_mapped_for_mel, ba)
            # Risk 2 fallback：若啟用 dmel_mix_amateur_real，real 餵 pro+amateur 混合；
            # 用 cat 一次餵進 D_mel 比餵兩次平均更穩（梯度尺度同步）
            if self.cfg.dmel_mix_amateur_real:
                real_mel = torch.cat([bp["mel"], ba["mel"]], dim=0)  # [2B, T, NUM_MELS]
            else:
                real_mel = bp["mel"]
            real_ret = self.D_mel(real_mel)
            fake_ret = self.D_mel(fake_mel.detach())
            d_mel_loss = hinge_d_loss(real_ret["y"], fake_ret["y"])
            self.opt_dmel.zero_grad()
            d_mel_loss.backward()
            self.opt_dmel.step()
            d_mel_loss_val = d_mel_loss.item()

        # ── (3) Update M (含 PatchNCE proj head) ──
        z_a_mapped = self.M(z_a)
        m_loss_dict = {}

        # 3a. L_PatchNCE：z_a vs M(z_a)
        l_nce = self.patchnce(z_a, z_a_mapped, side_a["mask_z"])
        m_total = self.cfg.lambda_patchnce * l_nce
        m_loss_dict["l_nce"] = l_nce.item()

        # 3b. L_adv_z（warmup 期間設 0）
        d_z_active = self.step >= self.cfg.d_z_warmup_steps
        if d_z_active:
            fake_score_g = self.D_z(z_a_mapped, side_a["register_z"], side_a["phoneme_z"])
            l_adv_z = hinge_g_loss(fake_score_g)
            m_total = m_total + self.cfg.lambda_adv_z * l_adv_z
            m_loss_dict["l_adv_z"] = l_adv_z.item()
        else:
            m_loss_dict["l_adv_z"] = 0.0

        # 3c. L_adv_mel
        if self.cfg.lambda_adv_mel > 0:
            mel_g = self._decode_with_mapped_z(z_a_mapped, ba)
            mel_g_score = self.D_mel(mel_g)["y"]
            l_adv_mel = hinge_g_loss(mel_g_score)
            m_total = m_total + self.cfg.lambda_adv_mel * l_adv_mel
            m_loss_dict["l_adv_mel"] = l_adv_mel.item()
        else:
            m_loss_dict["l_adv_mel"] = 0.0

        # 3d. L_identity_pro（隨機 20% 抽中）
        # 為什麼 stochastic 不是 bug（設計意圖）：
        #   - 設計目的：M 偶爾被叫去看 pro 端「不要漂太遠」，但不犧牲 amateur 端訓練
        #   - 「100% × 0.02」不等價：每步固定觸發改變 gradient direction 性質，
        #     會把 M 訓成「總是試圖 identity-on-pro」（pro 聲也有些瑕疵應允許微調）
        # 為什麼動量擾動量級小：
        #   - lambda × E[l_id_pro] ≈ 0.1 × 0.01·||z|| ≈ 0.001·||z||，比主 loss
        #     (l_nce/l_adv_z ~0.5–2) 小 1–2 數量級
        #   - opt_m 用 Adam β1=0.5（GAN 慣例），動量半衰期 ~1 步即輕量
        # 注意：本配方尚未經過實機驗證；若實際訓練 m_total 出現「每 5 步一個 spike」
        #      週期性震盪，考慮改為 100% 觸發 + 較小常數權重（例如 0.02）
        if (self.cfg.lambda_identity_pro > 0
                and torch.rand(1).item() < self.cfg.identity_pro_prob):
            z_p_mapped = self.M(z_p)
            l_id_pro = l_identity_pro(z_p, z_p_mapped, side_p["mask_z"])
            m_total = m_total + self.cfg.lambda_identity_pro * l_id_pro
            m_loss_dict["l_id_pro"] = l_id_pro.item()
        else:
            m_loss_dict["l_id_pro"] = 0.0

        # ── 監控：M 動多少 + 動的方向對不對 ──
        with torch.no_grad():
            delta = self.M.delta_only(z_a)
            delta_ratio = (delta.norm() / (z_a.norm() + 1e-6)).item()

            # temporal_diff_ratio：M 是否抹平既有時間軌跡（顫音/滑音）
            # Δ_t z = z[:, :, 1:] - z[:, :, :-1]  → z 的時間導數
            # 比較 M(z) 與 z 的時間導數差異：理想情況 M 只動空間方向（spectral envelope
            # 對應的 z 元件），時間導數的"變化"應與原 z 差不多 → ratio < 0.3
            # 若 M 把 z 抹平成 const → M(z) 的時間導數變小 → ratio 飆高
            # 若 M 把 z 加噪 → 同樣 ratio 飆高
            dz_t = z_a[:, :, 1:] - z_a[:, :, :-1]
            dM_t = z_a_mapped[:, :, 1:] - z_a_mapped[:, :, :-1]
            temporal_diff_ratio = (
                (dM_t - dz_t).abs().mean() / (dz_t.abs().mean() + 1e-6)
            ).item()
        m_loss_dict["delta_over_z"] = delta_ratio
        m_loss_dict["temporal_diff_ratio"] = temporal_diff_ratio

        self.opt_m.zero_grad()
        m_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.M.parameters()) + list(self.patchnce.parameters()),
            max_norm=self.cfg.grad_clip,
        )
        self.opt_m.step()

        return {
            "m_total": m_total.item(),
            "d_z": d_z_loss.item(),
            "d_mel": d_mel_loss_val,
            **m_loss_dict,
        }

    def save_ckpt(self, tag: str = "latest"):
        path = Path(self.cfg.ckpt_dir) / f"stage2_{tag}.pt"
        state = {
            "step": self.step,
            "M": self.M.state_dict(),
            "D_z": self.D_z.state_dict(),
            "D_mel": self.D_mel.state_dict(),
            "patchnce": self.patchnce.state_dict(),
            "opt_m": self.opt_m.state_dict(),
            "opt_dz": self.opt_dz.state_dict(),
            # freeze_d_mel:opt_dmel = None,寫 None 不寫 dict
            "opt_dmel": self.opt_dmel.state_dict() if self.opt_dmel is not None else None,
            "config": self.cfg.__dict__,
            "stage1_ckpt": self.cfg.stage1_ckpt,
        }
        torch.save(state, path)
        print(f"[stage2] ckpt saved: {path}", flush=True)

    def load_ckpt(self, ckpt_path: str):
        """
        從 ckpt 載入 Stage 2 完整訓練狀態（M/D_z/D_mel + 三個 optimizer + step）。

        為什麼不重載 CVAE：
          CVAE 已在 __init__ 從 stage1_ckpt 載過且凍結；ckpt 內也沒存 CVAE state。
          Resume 只動 stage 2 自己的可訓練組件。

        為什麼 PatchNCE 也存：
          PatchNCE 內部有 projection head（一個 MLP），它的 weight 隨訓練更新；
          不恢復會讓 contrastive loss 一夕重置，M 收到大跳變的梯度信號。
        """
        print(f"[stage2] resume from {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.M.load_state_dict(state["M"], strict=True)
        self.D_z.load_state_dict(state["D_z"], strict=True)
        self.D_mel.load_state_dict(state["D_mel"], strict=True)
        if "patchnce" in state:
            self.patchnce.load_state_dict(state["patchnce"], strict=True)
        self.opt_m.load_state_dict(state["opt_m"])
        self.opt_dz.load_state_dict(state["opt_dz"])
        # freeze_d_mel:本 run opt_dmel=None,跳過。若 ckpt 是非凍結時存的,opt_dmel
        # state 仍會留在檔案內,本 run 不載入即可(D_mel weight 仍會被載)
        if self.opt_dmel is not None and state.get("opt_dmel") is not None:
            self.opt_dmel.load_state_dict(state["opt_dmel"])
        self.step = int(state.get("step", 0))
        print(f"[stage2] resumed at step={self.step}", flush=True)

    @torch.no_grad()
    def monitor_audio_quality(self, n_samples: int = 4):
        """
        Risk 2 補強 P2：訓中音質監控。

        每隔 audio_quality_monitor_interval 步抽 N 個 amateur 樣本：
            1. 跑 M + decode 得 mel_modified
            2. 算 Δ_mel = mel_modified - mel_gt（對 mel 而非 z 算）
            3. 計算「Δ-energy 集中度」指標：
                  - Δ 在 voiced frames 平均能量
                  - Δ 在 unvoiced frames 平均能量
                  - **比例 = unvoiced_E / (voiced_E + unvoiced_E)**
                  - 比例越高 → M 把修飾集中在 noise/reverb 段 → Risk 2 警訊
            4. 存 mel spectrogram 對比 npz（GT / modified / Δ）供視覺檢查

        為什麼不存 wav 給人耳聽：
            wav 需要 vocoder forward，每次抽都跑 vocoder 太慢；
            mel-level 對比已經能看出 Risk 2 跡象（Δ 集中在 silent / 邊界段）；
            真要聽，從 ckpt 跑獨立 inference script 即可

        為什麼不每 step 都跑：
            音質監控是 sanity check 而非 loss signal，每 5000 步抽幾筆即可
        """
        from pathlib import Path

        log_dir = Path(self.cfg.audio_quality_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 從 amateur dataset 抽 n_samples 個（用 self.iter_a 拿到的下個 batch 切前 n 個）
        ba = self._to_device(next(self.iter_a))
        n = min(n_samples, ba["mel"].shape[0])

        # 跑 forward (eval mode)
        self.M.eval()
        try:
            side_a = self._encode_and_downsample(ba)
            z_a = side_a["z"][:n]
            z_a_mapped = self.M(z_a)
            mel_modified = self._decode_with_mapped_z(z_a_mapped, {
                "ppg": ba["ppg"][:n], "f0": ba["f0"][:n],
                "spk_emb": ba["spk_emb"][:n], "mel_mask": ba["mel_mask"][:n],
            })  # [n, T_mel, NUM_MELS]

            # 也餵 z_a 原值（不過 M）給 decoder 算「baseline mel」
            # 為什麼要這個 baseline：M 的 Δ 直接對比 z 餵 decoder 出的 baseline 才公平，
            # 而不是對比 GT mel（GT mel 含原 wav 細節，decoder 重建本身就有差）
            mel_baseline = self._decode_with_mapped_z(z_a, {
                "ppg": ba["ppg"][:n], "f0": ba["f0"][:n],
                "spk_emb": ba["spk_emb"][:n], "mel_mask": ba["mel_mask"][:n],
            })

            mask_n = ba["mel_mask"][:n]                        # [n, T_mel]
            voiced_n = (ba["f0"][:n] > 0).float()               # [n, T_mel] voiced=1, unvoiced=0
            valid_voiced = voiced_n * mask_n                    # voiced 且非 padding
            valid_unvoiced = (1 - voiced_n) * mask_n            # unvoiced 且非 padding

            delta_mel = mel_modified - mel_baseline             # [n, T_mel, NUM_MELS]
            delta_energy_per_frame = delta_mel.pow(2).sum(-1)   # [n, T_mel]

            # 平均「voiced frame 上 Δ 能量」 vs 「unvoiced frame 上 Δ 能量」
            voiced_e = (delta_energy_per_frame * valid_voiced).sum() / valid_voiced.sum().clamp(min=1)
            unvoiced_e = (delta_energy_per_frame * valid_unvoiced).sum() / valid_unvoiced.sum().clamp(min=1)
            unvoiced_concentration = unvoiced_e / (voiced_e + unvoiced_e + 1e-10)

            # 為什麼用「百分比」報告：絕對能量受 z 規模影響，
            # 比例 0.5 = 隨機分布，> 0.6 = M 略偏向 unvoiced（Risk 2 警訊）

            # ── voiced 段 Δ-mel 的時間頻譜拆分 ──
            # M 在 voiced 段做的「好事」應該是改 spectral envelope / formant 結構
            # （低時間頻率變化：跨數十 ms 的共鳴調整，frame-to-frame 變化小）
            # 不該大幅改 F0-related trajectory（高時間頻率變化：顫音/滑音的時間細節）
            # 拆法：對 Δ_mel 沿時間軸做 1-frame 差分得「高時間頻率分量」；
            #       (Δ_mel[t] + Δ_mel[t+1]) / 2 = 「低時間頻率分量」（envelope shift）
            #       voiced_spectral_ratio = low_E / (low_E + high_E)
            # 經 synthetic test 校準的閾值：
            #   ≥ 0.7：M 改動以 envelope shift 為主（健康，例如統一加強共鳴）
            #   0.4–0.7：marginal
            #   < 0.4：M 改動以高頻時間振盪為主（可能在動 F0 trajectory，警訊）
            # 為什麼閾值不對稱：random noise 自然落 ~0.2，純 envelope 改動接近 1.0；
            # 「健康」靠近高端，「失敗」是大幅偏離高端
            delta_mel_hf = delta_mel[:, 1:, :] - delta_mel[:, :-1, :]
            delta_mel_lf = (delta_mel[:, 1:, :] + delta_mel[:, :-1, :]) / 2
            valid_voiced_pair = (valid_voiced[:, 1:] * valid_voiced[:, :-1])
            voiced_hf_e = (delta_mel_hf.pow(2).sum(-1) * valid_voiced_pair).sum() / valid_voiced_pair.sum().clamp(min=1)
            voiced_lf_e = (delta_mel_lf.pow(2).sum(-1) * valid_voiced_pair).sum() / valid_voiced_pair.sum().clamp(min=1)
            voiced_spectral_ratio = (
                voiced_lf_e / (voiced_lf_e + voiced_hf_e + 1e-10)
            ).item()

            print(f"[stage2-monitor step {self.step}] "
                  f"Δ_voiced_E={voiced_e.item():.4f}  "
                  f"Δ_unvoiced_E={unvoiced_e.item():.4f}  "
                  f"unvoiced_concentration={unvoiced_concentration.item():.3f}  "
                  f"(< 0.55 良好, > 0.65 Risk2 警訊) "
                  f"voiced_spectral_ratio={voiced_spectral_ratio:.3f} "
                  f"(≥ 0.7 envelope-dominated 健康；< 0.4 M 改 F0 trajectory 警訊)",
                  flush=True)

            # 存 spectrogram（小，每次只一張 sample）
            sample_npz = log_dir / f"step{self.step:06d}_sample0.npz"
            import numpy as np
            np.savez_compressed(
                sample_npz,
                mel_gt=ba["mel"][0].cpu().numpy(),
                mel_baseline=mel_baseline[0].cpu().numpy(),
                mel_modified=mel_modified[0].cpu().numpy(),
                delta_mel=delta_mel[0].cpu().numpy(),
                f0=ba["f0"][0].cpu().numpy(),
                voiced=voiced_n[0].cpu().numpy(),
                step=np.array(self.step),
                unvoiced_concentration=np.array(unvoiced_concentration.item()),
                voiced_spectral_ratio=np.array(voiced_spectral_ratio),
            )
        finally:
            self.M.train()

    def fit(self):
        from tqdm import tqdm
        pbar = tqdm(
            total=self.cfg.max_steps, initial=self.step,
            ncols=120, ascii=True, dynamic_ncols=False,
            desc="stage2",
        )
        print(f"[stage2] start training: device={self.device} "
              f"step={self.step}/{self.cfg.max_steps}", flush=True)
        t0 = time.time()
        running = {}
        last_logged_step = self.step
        # 健康檢查 — 30k 步後若 Δ/z 過低 (mode A) 或 temporal_diff_ratio 過高 (mode B)
        # 各 print 一次性警告
        # 為什麼用 list of recent values 而非單點：避免單步雜訊導致 false positive
        delta_recent: list = []
        tdr_recent: list = []
        delta_warned = False
        tdr_warned = False
        while self.step < self.cfg.max_steps:
            metrics = self.train_step()
            self.step += 1
            pbar.update(1)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

            if self.step % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                n = max(self.step - last_logged_step, 1)
                rate = n / max(elapsed, 1e-6)
                avg = {k: v / n for k, v in running.items()}
                log_str = (f"[step {self.step:6d} {rate:.1f}it/s] "
                           + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
                pbar.write(log_str)
                pbar.set_postfix({
                    "m": f"{avg.get('m_total', 0):.3f}",
                    "d_z": f"{avg.get('d_z', 0):.3f}",
                    "Δ/z": f"{avg.get('delta_over_z', 0):.3f}",
                    "tdr": f"{avg.get('temporal_diff_ratio', 0):.3f}",
                })
                # 收集移動平均樣本，留待 30k 步後做健康檢查
                delta_recent.append(avg.get("delta_over_z", 0.0))
                tdr_recent.append(avg.get("temporal_diff_ratio", 0.0))
                if len(delta_recent) > 20:
                    delta_recent.pop(0)
                if len(tdr_recent) > 20:
                    tdr_recent.pop(0)
                running = {}
                last_logged_step = self.step
                t0 = time.time()

            # Failure mode A：M 太保守警告（一次性）
            if (not delta_warned
                    and self.step >= self.cfg.health_check_step
                    and len(delta_recent) >= 10
                    and self.cfg.m_kernel_size == 1):
                delta_ma = sum(delta_recent) / len(delta_recent)
                if delta_ma < self.cfg.delta_low_threshold:
                    pbar.write(
                        f"[stage2] ⚠️  Failure mode A: ‖Δ‖/‖z‖ moving avg = {delta_ma:.4f} "
                        f"< threshold {self.cfg.delta_low_threshold} "
                        f"after {self.step} steps with kernel_size=1.\n"
                        f"           M 動的太少（pointwise 表達力不足）。\n"
                        f"           建議：以 --m-kernel-size 3 重新訓練 "
                        f"(可從 stage1 ckpt 接續，M 結構不同無法直接續訓 stage2 ckpt)。"
                    )
                    delta_warned = True

            # Failure mode B：M 抹平時間軌跡警告（一次性）
            if (not tdr_warned
                    and self.step >= self.cfg.health_check_step
                    and len(tdr_recent) >= 10):
                tdr_ma = sum(tdr_recent) / len(tdr_recent)
                if tdr_ma > self.cfg.temporal_diff_high_threshold:
                    pbar.write(
                        f"[stage2] ⚠️  Failure mode B: temporal_diff_ratio moving avg = "
                        f"{tdr_ma:.4f} > threshold {self.cfg.temporal_diff_high_threshold} "
                        f"after {self.step} steps.\n"
                        f"           M 改動的時間導數量級超過 z 自身 → 顫音/滑音等時間"
                        f"結構可能被抹平或重寫。\n"
                        f"           建議：(1) 用 monitor_audio_quality dump 的 npz 視覺化"
                        f" delta_mel 確認；(2) 若確實有，降 lambda_adv_z 或提早停訓；"
                        f"(3) 訓 inference 後跑 F0 trajectory 比對。"
                    )
                    tdr_warned = True

            if self.step % self.cfg.save_interval == 0:
                self.save_ckpt(tag=f"step{self.step}")
                self.save_ckpt(tag="latest")

            # Risk 2 補強 P2：訓中音質監控
            if self.step % self.cfg.audio_quality_monitor_interval == 0:
                self.monitor_audio_quality(
                    n_samples=self.cfg.audio_quality_monitor_n_samples,
                )

        pbar.close()
        self.save_ckpt(tag="final")
        print(f"[stage2] training done at step {self.step}", flush=True)


# ── CLI ──────────────────────────────────────────────────
def main():
    # 同 stage1:確保 stdout 在 pipe-to-tee 場景下 line-buffered,init 階段
    # 預讀 dataset(~15 min)的 print 才不會卡在 buffer 看不到。
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="NSVB-ZH Stage 2 mapping training")
    parser.add_argument("--binarized-root", default="data/binarized")
    parser.add_argument("--amateur-dataset", default="vocalverse")
    parser.add_argument("--pro-dataset", default="m4singer")
    parser.add_argument("--ppg-dim", type=int, default=1280)
    parser.add_argument("--phoneme-vocab-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=1500,
                        help="batch 內樣本長度上限（mel frames）。1500 ≈ 8.7s @ hop=128，"
                             "對應 VV 切 5s chunks + M4 9s outlier 全涵蓋")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--stage1-ckpt", default="checkpoints/stage1/stage1_latest.pt",
                        required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/stage2")
    parser.add_argument("--m-kernel-size", type=int, default=1, choices=[1, 3])
    parser.add_argument("--dmel-mix-amateur-real", action="store_true",
                        help="Risk 2 fallback：D_mel real 改餵 pro+amateur 混合（預設只看 pro）。"
                             "訓中 monitor 顯示 unvoiced_concentration > 0.65 連續兩次時 resume + 啟此 flag 救火；"
                             "犧牲 mel 層 pro-direction 訊號換取「絕不鼓勵去殘響」的安全")
    parser.add_argument("--freeze-d-mel", action="store_true",
                        help="v2 redo:凍結 D_mel 不更新(forward 仍給 M 梯度)。"
                             "目的:Risk §二.3 amateur F0 conditioning confound 救火 — "
                             "Δ/z 0.93 + tdr 0.77 超出健康範圍時把 mel-side 對抗壓力削掉。"
                             "搭配 --lambda-adv-mel 0.05 + --lr-dz 2e-4 使用")
    parser.add_argument("--lambda-adv-mel", type=float, default=None,
                        help="覆寫 Stage2Config.lambda_adv_mel(預設 0.2)。v2 redo 建議 0.05")
    parser.add_argument("--lr-dz", type=float, default=None,
                        help="覆寫 Stage2Config.lr_dz(預設 4e-4 = 4× lr_m,TTUR)。"
                             "v2 redo 建議 2e-4(2× TTUR,削弱 latent 對抗壓力配合 D_mel freeze)")
    parser.add_argument("--resume", default="",
                        help="從現有 Stage 2 ckpt 路徑恢復（M/D_z/D_mel + 三 optimizer + step）。"
                             "可用 'latest' 簡寫，自動找 {ckpt-dir}/stage2_latest.pt")
    parser.add_argument("--split-dir", default="data/binarized/splits",
                        help="train.txt 所在目錄(scripts/make_splits.py 產出)。"
                             "該目錄無 train.txt 時 fallback 用全 dataset")
    args = parser.parse_args()

    cfg_kwargs = dict(
        binarized_root=args.binarized_root,
        amateur_dataset=args.amateur_dataset,
        pro_dataset=args.pro_dataset,
        ppg_dim=args.ppg_dim,
        phoneme_vocab_size=args.phoneme_vocab_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        stage1_ckpt=args.stage1_ckpt,
        ckpt_dir=args.ckpt_dir,
        m_kernel_size=args.m_kernel_size,
        dmel_mix_amateur_real=args.dmel_mix_amateur_real,
        freeze_d_mel=args.freeze_d_mel,
        split_dir=args.split_dir,
    )
    if args.lambda_adv_mel is not None:
        cfg_kwargs["lambda_adv_mel"] = args.lambda_adv_mel
    if args.lr_dz is not None:
        cfg_kwargs["lr_dz"] = args.lr_dz
    cfg = Stage2Config(**cfg_kwargs)
    trainer = Stage2Trainer(cfg)

    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            resume_path = str(Path(cfg.ckpt_dir) / "stage2_latest.pt")
        trainer.load_ckpt(resume_path)

    trainer.fit()


if __name__ == "__main__":
    main()
