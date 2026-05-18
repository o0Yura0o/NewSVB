"""Periodic mel-domain val eval during Stage 2 training (v3 早停 hook).

【為什麼存在】
訓中 loss(`m_total`、`d_z`、`l_adv_*`) 不直接反映「M 真的有沒有做對」。我們已有
[scripts/stage2_mel_eval.py](../../scripts/stage2_mel_eval.py) 在 mel-domain 算
`pro_direction_alignment`,跟主觀品質強相關。把該邏輯包進訓中 hook,每 N 步在固定
val subset 上算一次,**自動存 `stage2_best.pt` by pro_direction_alignment**。

【設計】
- 每 N 步(預設 5000)抽 20 個 val samples,跑 cvae → M → decoder(no vocoder)
- 算每個 sample 的 `pro_direction_alignment`(對 pro_mean_env 的方向 cosine)
- 取平均,若 > best,複製當前 `stage2_latest.pt` → `stage2_best.pt`
- `pro_mean_env` 訓練開始時計算一次(從 splits/train.txt 抽 200 pro sample),
  避免 test contamination 同 stage2_mel_eval.py 的處理
- baseline mel (M=None decoded) 對固定 val subset 算一次 cache,後續 step 直接用

【性能】
- val eval 一次(20 samples × ~0.5s forward + metrics)≈ 10-15s
- 訓中每 5K 步呼叫一次 → 50K 步 = 10 次 eval = 150s overhead
- 佔 50K 步訓練(~3-5h on T4)的 ~1%,可接受

【不破壞訓練】
- M.eval() 期間呼叫,完了 M.train() 還原
- 全程 `with torch.no_grad()`,無梯度洩漏
- cvae 本來就 eval+frozen,baseline mel 一次算完 cache
- 不存 D_z / D_mel / opt 到 stage2_best.pt,只存 inference 必要的(M / cvae 引用)
"""
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nsvb.utils.audio_config import LATENT_DOWN_FACTOR


def _pad_to_multiple(arr: np.ndarray, multiple: int, pad_value: float = 0.0) -> np.ndarray:
    T = arr.shape[0]
    pad = (-T) % multiple
    if pad == 0:
        return arr
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[0] = (0, pad)
    return np.pad(arr, pad_widths, constant_values=pad_value)


def _pick_val_samples(
    val_split_file: Path,
    binarized_root: Path,
    n_samples: int,
    seed: int = 42,
) -> list[Path]:
    """從 val split 抽 N 個樣本,1/4 M4(pro control)+ 3/4 VV(amateur 主訊號)。"""
    val_list = [s.strip() for s in val_split_file.read_text().splitlines() if s.strip()]
    val_vv = [v for v in val_list if "__c" in v]
    val_m4 = [v for v in val_list if "#" in v and v not in val_vv]

    rng = random.Random(seed)
    n_m4 = min(n_samples // 4, len(val_m4))
    n_vv = min(n_samples - n_m4, len(val_vv))
    picks = (rng.sample(val_m4, n_m4) if n_m4 else []) + (rng.sample(val_vv, n_vv) if n_vv else [])

    paths: list[Path] = []
    for item in picks:
        for ds in ("m4singer", "vocalverse"):
            p = binarized_root / ds / f"{item}.npz"
            if p.exists():
                paths.append(p)
                break
    return paths


def _compute_pro_mean_env(
    binarized_root: Path,
    pro_dataset: str,
    n: int = 200,
    seed: int = 42,
    train_split_file: Optional[Path] = None,
) -> np.ndarray:
    """從 pro dataset(train split 內若 train_split_file 給了)抽 N sample 算 mean envelope。"""
    pro_dir = binarized_root / pro_dataset
    if train_split_file is not None and train_split_file.exists():
        items = [s.strip() for s in train_split_file.read_text().splitlines() if s.strip()]
        candidate = [pro_dir / f"{item}.npz" for item in items]
        candidate = [p for p in candidate if p.exists()]
    else:
        candidate = sorted(pro_dir.glob("*.npz"))
    if not candidate:
        raise SystemExit(f"[eval_hook] 找不到 pro samples 算 pro_mean_env")
    rng = random.Random(seed)
    sampled = rng.sample(candidate, min(n, len(candidate)))
    envs = []
    for p in sampled:
        with np.load(p, allow_pickle=True) as d:
            mel = d["mel"].astype(np.float32)
        envs.append(mel.mean(axis=0))
    return np.stack(envs).mean(axis=0).astype(np.float32)


@torch.no_grad()
def _encode_decode(cvae, M_or_none, npz_path: Path, device: torch.device) -> np.ndarray:
    """跑 cvae encoder → (M_or_none) → decoder,回 mel_out [T_orig, NUM_MELS]。"""
    with np.load(npz_path, allow_pickle=True) as d:
        mel_np = d["mel"].astype(np.float32)
        ppg_np = d["ppg"].astype(np.float32)
        f0_np = d["f0"].astype(np.float32)
        spk_np = d["spk_emb"].astype(np.float32)
    T_orig = mel_np.shape[0]
    mel_pad = _pad_to_multiple(mel_np, LATENT_DOWN_FACTOR, pad_value=-10.0)
    ppg_pad = _pad_to_multiple(ppg_np, LATENT_DOWN_FACTOR, pad_value=0.0)
    f0_pad = _pad_to_multiple(f0_np, LATENT_DOWN_FACTOR, pad_value=0.0)

    mel = torch.from_numpy(mel_pad).unsqueeze(0).to(device)
    ppg = torch.from_numpy(ppg_pad).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0_pad).unsqueeze(0).to(device)
    spk = torch.from_numpy(spk_np).unsqueeze(0).to(device)
    mel_mask = torch.zeros_like(f0_t)
    mel_mask[:, :T_orig] = 1.0

    g = cvae.condition(ppg, f0_t, spk)
    mel_chfirst = mel.transpose(1, 2)
    mask_exp = mel_mask.unsqueeze(1)
    g_sqz = cvae.fvae.g_pre_net(g)
    _z_q, m_q, _logs_q, _ = cvae.fvae.encoder(mel_chfirst, mask_exp, g_sqz)
    z_in = m_q if M_or_none is None else M_or_none(m_q)
    mel_out_chfirst = cvae.fvae.decoder(z_in, mask_exp, g)
    mel_out_chfirst = mel_out_chfirst[:, :, :T_orig]
    return mel_out_chfirst.squeeze(0).transpose(0, 1).cpu().numpy()


def _pro_direction_alignment(
    mel_out: np.ndarray, mel_baseline: np.ndarray, pro_mean_env: np.ndarray,
) -> float:
    """同 stage2_mel_eval.py compute_metrics:cos(modification_vec, pro_direction_vec)。"""
    env_out = mel_out.mean(axis=0)
    env_baseline = mel_baseline.mean(axis=0)
    mod_vec = env_out - env_baseline
    dir_vec = pro_mean_env - env_baseline
    mod_norm = np.linalg.norm(mod_vec) + 1e-10
    dir_norm = np.linalg.norm(dir_vec) + 1e-10
    return float(np.dot(mod_vec, dir_vec) / (mod_norm * dir_norm))


class Stage2BestCkptHook:
    """訓中 val eval + best ckpt 保存。

    Usage:
        hook = Stage2BestCkptHook(trainer, binarized_root, val_split_file, ...)
        trainer.val_eval_hook = hook
        trainer.fit()  # fit() 內每 hook.interval 步呼叫 hook(step)
    """

    def __init__(
        self,
        trainer,
        binarized_root: Path,
        val_split_file: Path,
        pro_dataset: str = "m4singer",
        n_samples: int = 20,
        interval: int = 5000,
        pro_mean_n: int = 200,
        seed: int = 42,
    ):
        self.trainer = trainer
        self.interval = interval
        self.device = trainer.device

        # 抽 val subset(固定,訓中不變)
        self.val_paths = _pick_val_samples(
            val_split_file, binarized_root, n_samples, seed=seed,
        )
        if not self.val_paths:
            raise SystemExit(f"[eval_hook] val 抽不到 samples (split={val_split_file})")
        print(f"[eval_hook] val subset: {len(self.val_paths)} samples"
              f" (M4={sum(1 for p in self.val_paths if 'm4singer' in str(p))},"
              f" VV={sum(1 for p in self.val_paths if 'vocalverse' in str(p))})",
              flush=True)

        # 算 pro_mean_env (避免 test contamination,從 train split 抽)
        train_split = val_split_file.parent / "train.txt"
        self.pro_mean_env = _compute_pro_mean_env(
            binarized_root, pro_dataset, n=pro_mean_n,
            seed=seed,
            train_split_file=train_split if train_split.exists() else None,
        )
        print(f"[eval_hook] pro_mean_env computed (n={pro_mean_n}, "
              f"split={'train.txt' if train_split.exists() else 'all'})", flush=True)

        # baseline mels(M=None decoded)在訓開始時 cache 一次
        self.baseline_mels: dict[Path, np.ndarray] = {}
        print(f"[eval_hook] caching baseline mels for {len(self.val_paths)} val samples...",
              flush=True)
        self.trainer.M.eval()  # 雖然這時還沒訓,還是保守 eval mode
        try:
            for p in self.val_paths:
                self.baseline_mels[p] = _encode_decode(
                    self.trainer.cvae, None, p, self.device,
                )
        finally:
            self.trainer.M.train()
        print(f"[eval_hook] baseline mels cached", flush=True)

        # tracker
        self.best_alignment: float = -float("inf")
        self.best_step: int = 0
        self.history: list[dict] = []

    @torch.no_grad()
    def __call__(self, step: int) -> dict:
        """跑 val eval,回 {val_pro_direction_alignment, is_best, best_alignment, best_step}。"""
        self.trainer.M.eval()
        try:
            alignments = []
            for p in self.val_paths:
                mel_out = _encode_decode(self.trainer.cvae, self.trainer.M, p, self.device)
                mel_baseline = self.baseline_mels[p]
                a = _pro_direction_alignment(mel_out, mel_baseline, self.pro_mean_env)
                alignments.append(a)
            avg = float(np.mean(alignments))
        finally:
            self.trainer.M.train()

        is_best = avg > self.best_alignment
        if is_best:
            self.best_alignment = avg
            self.best_step = step
            self._save_best_ckpt()

        self.history.append({"step": step, "alignment": avg, "is_best": is_best})
        return {
            "val_pro_direction_alignment": avg,
            "is_best": is_best,
            "best_alignment": self.best_alignment,
            "best_step": self.best_step,
        }

    def _save_best_ckpt(self):
        """複製 stage2_latest.pt → stage2_best.pt(latest 由 fit() 在 save_interval 存)。
        若 latest 還沒被 save_interval 觸發過,直接呼叫 trainer.save_ckpt('best')。
        """
        latest_path = Path(self.trainer.cfg.ckpt_dir) / "stage2_latest.pt"
        best_path = Path(self.trainer.cfg.ckpt_dir) / "stage2_best.pt"
        if latest_path.exists():
            shutil.copy2(str(latest_path), str(best_path))
        else:
            self.trainer.save_ckpt(tag="best")
