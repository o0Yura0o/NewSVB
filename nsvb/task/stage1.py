"""
nsvb/task/stage1.py
=====================

【這支檔案做什麼】
NSVB-ZH Phase 1 — Stage 1 CVAE 預訓練 task。
訓練 SVBVAEZh model 做 mel 自我重建：
    x → φ(x; PPG, F0, spk_emb) → z → θ(z; PPG, F0, spk_emb) → x̂
    L = L_l1(x̂, x) + L_l2(x̂, x) + β · L_KL(q(z|x) ‖ N(0, I))
        (+ λ · L_adv_mel 從 D_mel，可選)

不在這裡做的事：
- M / D_z / PatchNCE / L_PPG （都是 Stage 2，獨立 task）
- 推理（獨立 inference script）

【Stage 1 訓練策略】
- m4singer + VocalVerse 兩邊都當 real（學「自然人聲」分布）
- D_mel 此階段也看兩邊都 real；Stage 2 才把 real 改成只看 pro
- 兩種啟動：
    1. 從零訓 (--init-from-nsvb 不帶)：~50-80k 步至少
    2. 從 NSVB 1030_vae_mle 初始化 (--init-from-nsvb): ~10-30k 步即可達同等品質
       （CVAE backbone 88/88 weights 已預訓練，只需 fine-tune cond_layer
       與重新訓 condition modules）

【為什麼 Stage 1 兩邊都當 real】
Stage 1 目標：CVAE 學「給定 condition 重建 mel」這個基本能力，不分業餘/職業。
若 D_mel real 此時就分流，amateur side 學不到正確 mel，會破壞 latent space 結構。

【為什麼用純 PyTorch 訓練迴圈而非 PyTorch Lightning】
- Lightning 對 NSVB 風格的 multi-optimizer (G/D 對抗) 支援不直接
- 純 PyTorch 迴圈程式碼少、debug 直觀、訓練 log 自己控
- 與 NSVB 原 repo 的 BaseTask（自寫 trainer）哲學相同

【Loss 細節】
1. L_mel = L1 + L2 mix（與 NSVB ssim:0.5|l1:0.5 配方類似）
   - L1 對整體 dB 偏差敏感
   - L2 對 outlier 敏感
   - 兩者皆 mask out padding frames
2. L_KL：FVAE 內部 closed-form Gaussian KL；用 KL warmup（前 N 步 β=0 線性升）
   - 防止 posterior collapse
3. L_adv_mel（optional, weight 0.1）：D_mel hinge loss
   - 提升重建自然度

【Checkpoint 格式】
.pt 含 step, epoch, model.state_dict, d_mel.state_dict, opt_g, opt_d, config dict
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from nsvb.data.dataset import BinarizedNSVBDataset, collate_fn
from nsvb.model.svb_vae_zh import SVBVAEZh
from nsvb.backbone.multi_window_disc import Discriminator as DMel
from nsvb.utils.transfer_weights import transfer_to_svbvae


# ── Loss helpers ────────────────────────────────────────
def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred / target: [B, T, NUM_MELS]
    mask:           [B, T]            1=valid, 0=pad
    """
    diff = (pred - target).abs()
    mask_e = mask.unsqueeze(-1)
    n_valid = mask.sum().clamp(min=1.0)
    return (diff * mask_e).sum() / (n_valid * pred.shape[-1])


def masked_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    為什麼用 L2 而非 SSIM：SSIM 在 GPU 上 differentiable 實作較重，
    Stage 1 目標是把 mel 大致重建出來，L2 已給足 signal；後續若要 SSIM 再加。
    """
    diff_sq = (pred - target).pow(2)
    mask_e = mask.unsqueeze(-1)
    n_valid = mask.sum().clamp(min=1.0)
    return (diff_sq * mask_e).sum() / (n_valid * pred.shape[-1])


def hinge_d_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_g_loss(fake_score: torch.Tensor) -> torch.Tensor:
    return -fake_score.mean()


# ── Config ───────────────────────────────────────────────
@dataclass
class Stage1Config:
    """Stage 1 hyperparameters。可被 CLI override。"""
    # ── 資料 ──
    binarized_root: str = "data/binarized"
    datasets: tuple = ("m4singer", "vocalverse")
    max_frames: int = 600
    batch_size: int = 16
    num_workers: int = 4

    # ── 模型 ──
    num_mels: int = 80
    ppg_dim: int = 1280     # whisper-large-v3=1280, smoke whisper-tiny=384
    spk_emb_dim: int = 256
    latent_size: int = 128
    hidden_size: int = 192
    enc_n_layers: int = 8
    dec_n_layers: int = 4
    fvae_kernel_size: int = 5

    # ── Loss 權重 ──
    lambda_l1: float = 1.0
    lambda_l2: float = 1.0
    lambda_kl: float = 0.01
    kl_warmup_steps: int = 5000
    lambda_adv_mel: float = 0.1
    use_d_mel: bool = True

    # ── Optimizer ──
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.98

    # ── 訓練 ──
    max_steps: int = 200000
    log_interval: int = 50
    save_interval: int = 5000
    grad_clip: float = 1.0

    # ── 路徑 ──
    ckpt_dir: str = "checkpoints/stage1"
    init_from_nsvb: str = ""  # 若非空，從此 ckpt 路徑載入 NSVB FVAE 預訓練權重
    init_lr_scale: float = 0.5  # NSVB init 後用較小 lr fine-tune（gentle）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Trainer ──────────────────────────────────────────────
class Stage1Trainer:
    """
    狀態（model / optim / step / epoch）封裝；支援從 NSVB ckpt 初始化。
    """

    def __init__(self, cfg: Stage1Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

        # ── Dataset ──
        sub_ds: List[BinarizedNSVBDataset] = []
        for ds_name in cfg.datasets:
            root = Path(cfg.binarized_root) / ds_name
            if not root.exists():
                print(f"[stage1] WARNING: {root} not found, skipping")
                continue
            sub_ds.append(BinarizedNSVBDataset(root, max_frames=cfg.max_frames))
        if not sub_ds:
            raise RuntimeError("No dataset found; run binarizer.py first")
        self.dataset = ConcatDataset(sub_ds)

        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=(cfg.device == "cuda"),
        )

        # ── Model ──
        self.model = SVBVAEZh(
            num_mels=cfg.num_mels, ppg_dim=cfg.ppg_dim, spk_emb_dim=cfg.spk_emb_dim,
            latent_size=cfg.latent_size, hidden_size=cfg.hidden_size,
            kernel_size=cfg.fvae_kernel_size,
            enc_n_layers=cfg.enc_n_layers, dec_n_layers=cfg.dec_n_layers,
        ).to(self.device)

        # ── 從 NSVB 1030_vae_mle 初始化（若指定）──
        # 為什麼 init 在 model 構造後、optimizer 構造前：
        #   要先載 weights 才知道 fine-tune lr 的起點；optimizer state 從 init 算起
        if cfg.init_from_nsvb:
            print(f"[stage1] initializing FVAE backbone from NSVB ckpt: {cfg.init_from_nsvb}")
            transfer_to_svbvae(self.model, cfg.init_from_nsvb, verbose=True)

        if cfg.use_d_mel:
            self.d_mel = DMel(
                time_lengths=(32, 64, 128), freq_length=cfg.num_mels,
                hidden_size=128, norm_type="bn", reduction="sum",
            ).to(self.device)
        else:
            self.d_mel = None

        # ── Optimizer ──
        # 為什麼 init_from_nsvb 時降 lr：
        #   weights 已收斂在 mel 重建，大 lr 會 destroy 預訓表徵；
        #   降 lr fine-tune cond_layer 與重新訓 condition modules 即可
        effective_lr_g = cfg.lr_g * (cfg.init_lr_scale if cfg.init_from_nsvb else 1.0)
        self.opt_g = torch.optim.Adam(
            self.model.parameters(), lr=effective_lr_g,
            betas=(cfg.beta1, cfg.beta2), eps=1e-9,
        )
        if self.d_mel is not None:
            self.opt_d = torch.optim.Adam(
                self.d_mel.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999),
            )
        else:
            self.opt_d = None

        self.step = 0
        self.epoch = 0
        print(f"[stage1] init done: lr_g={effective_lr_g:.2e}, "
              f"d_mel={'on' if self.d_mel else 'off'}, "
              f"params(G)={sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def _kl_weight(self) -> float:
        """KL weight linear warmup 0 → lambda_kl 在前 kl_warmup_steps 步。"""
        if self.step >= self.cfg.kl_warmup_steps:
            return self.cfg.lambda_kl
        return self.cfg.lambda_kl * (self.step / max(self.cfg.kl_warmup_steps, 1))

    def train_step(self, batch: dict) -> dict:
        """單 step 訓練：先更新 D，再更新 G。"""
        for k in ["mel", "ppg", "f0", "spk_emb", "mel_mask"]:
            batch[k] = batch[k].to(self.device, non_blocking=True)

        # ── (1) Update D_mel（如果啟用）──
        d_loss_val = 0.0
        if self.d_mel is not None and self.cfg.lambda_adv_mel > 0:
            with torch.no_grad():
                out = self.model(batch["mel"], batch["ppg"], batch["f0"],
                                 batch["spk_emb"], batch["mel_mask"], infer=False)
                fake_mel = out["mel_recon"]
            real_ret = self.d_mel(batch["mel"])
            fake_ret = self.d_mel(fake_mel.detach())
            d_loss = hinge_d_loss(real_ret["y"], fake_ret["y"])
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            d_loss_val = d_loss.item()

        # ── (2) Update G（CVAE） ──
        out = self.model(batch["mel"], batch["ppg"], batch["f0"],
                         batch["spk_emb"], batch["mel_mask"], infer=False)
        mel_recon = out["mel_recon"]
        kl = out["kl"]

        l_l1 = masked_l1(mel_recon, batch["mel"], batch["mel_mask"])
        l_l2 = masked_l2(mel_recon, batch["mel"], batch["mel_mask"])
        beta = self._kl_weight()
        l_total = self.cfg.lambda_l1 * l_l1 + self.cfg.lambda_l2 * l_l2 + beta * kl

        adv_val = 0.0
        if self.d_mel is not None and self.cfg.lambda_adv_mel > 0:
            fake_ret = self.d_mel(mel_recon)
            l_adv = hinge_g_loss(fake_ret["y"])
            l_total = l_total + self.cfg.lambda_adv_mel * l_adv
            adv_val = l_adv.item()

        self.opt_g.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
        self.opt_g.step()

        return {
            "l_total": l_total.item(),
            "l_l1": l_l1.item(),
            "l_l2": l_l2.item(),
            "kl": kl.item(),
            "kl_beta": beta,
            "l_adv_g": adv_val,
            "l_d": d_loss_val,
        }

    def save_ckpt(self, tag: str = "latest"):
        path = Path(self.cfg.ckpt_dir) / f"stage1_{tag}.pt"
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "config": self.cfg.__dict__,
        }
        if self.d_mel is not None:
            state["d_mel"] = self.d_mel.state_dict()
            state["opt_d"] = self.opt_d.state_dict()
        torch.save(state, path)
        print(f"[stage1] ckpt saved: {path}", flush=True)

    def load_ckpt(self, ckpt_path: str):
        """
        從 ckpt 載入完整訓練狀態（model + optimizer + step/epoch）以中段續訓。

        為什麼把 model + optimizer + step 都一起恢復：
          - 只恢復 model 會讓 Adam 動量重設，在原 lr 下 loss 會明顯反彈幾百步
          - step 必須恢復以維持 KL warmup schedule、save_interval、log_interval 對齊
          - epoch 用於 log，不嚴格但保留方便辨識

        為什麼用 strict=True：
          續訓必須是「同一個 model 結構」；若 config 不同會 silent 載入錯誤的 weight，
          訓練表面正常但結果偏差。strict 強制顯露結構不一致。
        """
        print(f"[stage1] resume from {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.opt_g.load_state_dict(state["opt_g"])
        if self.d_mel is not None and "d_mel" in state:
            self.d_mel.load_state_dict(state["d_mel"], strict=True)
            if self.opt_d is not None and "opt_d" in state:
                self.opt_d.load_state_dict(state["opt_d"])
        self.step = int(state.get("step", 0))
        self.epoch = int(state.get("epoch", 0))
        print(f"[stage1] resumed at step={self.step} epoch={self.epoch}", flush=True)

    def fit(self):
        # tqdm 提供 ssh 也能看的 ASCII progress bar；
        # ASCII=True 避免在 Windows / 非 utf-8 terminal 顯示亂碼
        from tqdm import tqdm
        pbar = tqdm(
            total=self.cfg.max_steps, initial=self.step,
            ncols=120, ascii=True, dynamic_ncols=False,
            desc="stage1",
        )
        print(f"[stage1] start training: device={self.device} "
              f"step={self.step}/{self.cfg.max_steps}", flush=True)
        t0 = time.time()
        running = {}
        last_logged_step = self.step
        while self.step < self.cfg.max_steps:
            self.epoch += 1
            for batch in self.loader:
                metrics = self.train_step(batch)
                self.step += 1
                pbar.update(1)

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v

                if self.step % self.cfg.log_interval == 0:
                    elapsed = time.time() - t0
                    n = max(self.step - last_logged_step, 1)
                    rate = n / max(elapsed, 1e-6)
                    avg = {k: v / n for k, v in running.items()}
                    log_str = (f"[step {self.step:6d} ep{self.epoch} {rate:.1f}it/s] "
                               + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
                    pbar.write(log_str)
                    # tqdm 上方狀態列同時顯示重點 loss
                    pbar.set_postfix({
                        "l_total": f"{avg.get('l_total', 0):.3f}",
                        "kl": f"{avg.get('kl', 0):.3f}",
                    })
                    running = {}
                    last_logged_step = self.step
                    t0 = time.time()

                if self.step % self.cfg.save_interval == 0:
                    self.save_ckpt(tag=f"step{self.step}")
                    self.save_ckpt(tag="latest")

                if self.step >= self.cfg.max_steps:
                    break

        pbar.close()
        self.save_ckpt(tag="final")
        print(f"[stage1] training done at step {self.step}", flush=True)


# ── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NSVB-ZH Stage 1 CVAE pretrain")
    parser.add_argument("--binarized-root", default="data/binarized")
    parser.add_argument("--ppg-dim", type=int, default=1280,
                        help="whisper-large-v3=1280, smoke whisper-tiny=384")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--no-adv", action="store_true",
                        help="disable D_mel adversarial loss (純 reconstruction)")
    parser.add_argument("--ckpt-dir", default="checkpoints/stage1")
    parser.add_argument("--init-from-nsvb", default="",
                        help="從 NSVB 1030_vae_mle ckpt 初始化 FVAE backbone "
                             "（強烈建議；省 ~1 週 GPU pretraining）")
    parser.add_argument("--resume", default="",
                        help="從現有 ckpt 路徑恢復訓練（model + optimizer + step）。"
                             "可用 'latest' 為簡寫，會自動找 {ckpt-dir}/stage1_latest.pt")
    args = parser.parse_args()

    cfg = Stage1Config(
        binarized_root=args.binarized_root,
        ppg_dim=args.ppg_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        use_d_mel=not args.no_adv,
        ckpt_dir=args.ckpt_dir,
        init_from_nsvb=args.init_from_nsvb,
    )
    trainer = Stage1Trainer(cfg)

    # 為什麼 resume 在 fit() 之前、init_from_nsvb 之後處理：
    #   resume 會覆蓋 model state；不能與 init_from_nsvb 同時用（resume 暗示「已經訓過」）
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            resume_path = str(Path(cfg.ckpt_dir) / "stage1_latest.pt")
        trainer.load_ckpt(resume_path)

    trainer.fit()


if __name__ == "__main__":
    main()
