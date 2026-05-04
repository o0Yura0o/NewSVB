# -*- coding: utf-8 -*-
"""
Phase 0 probe: dataset discriminability.

Question
--------
Can a simple classifier, given only the CVAE latent z, predict whether a clip
came from M4Singer or OpenSinger? If yes to a high degree, the D_z adversarial
signal in Stage 2 will be dominated by dataset-level artefacts (singer,
microphone, room, mixing), not by "amateur vs pro".

Method
------
1. Load a half-trained Stage 1 CVAE checkpoint (≥ 50k steps).
2. For N clips from each dataset (default N=1000 each), run the encoder to
   get z, then reduce to utterance-level features:
     - mean(z) over valid frames   (C dims)
     - std(z) over valid frames    (C dims)
   Total feature dim = 2C (≈ 384 for latent_dim=192).
3. Stratified 80/20 split, fit two classifiers:
     - L2-regularised logistic regression
     - Tiny MLP (256-hidden, 2-layer)
   Report both.
4. Decision:
     < 0.60 accuracy  -> Case A : run the base architecture
     0.60 -- 0.75      -> Case B : turn on GRL (grl_lambda 0.1 -> 0.3, warmup 30k)
     > 0.75 accuracy  -> Case C : do NOT run Stage 2 as-is. Options:
                         C1: MOSNet re-labeling (run scripts/mosnet_relabel.py
                             -- not included here, external tool needed)
                         C2: DTW pseudo-pairs (fall back to original NSVB)
                         C3: find a Chinese karaoke dataset (paired naturally)

Usage
-----
python scripts/probe_dataset_discriminability.py \\
    --ckpt  checkpoints/nsvb_zh_stage1/model_ckpt_steps_80000.ckpt \\
    --m4_data_dir  data/binary/m4singer \\
    --open_data_dir data/binary/opensinger \\
    --n_per_dataset 1000 \\
    --out_dir outputs/phase0_probe
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_encoder(ckpt_path: str, device: torch.device):
    """Load the Stage 1 CVAE encoder from an NSVB checkpoint.

    Assumes the checkpoint's state_dict contains the `model.fvae.encoder`
    submodule. Adjust the key prefix if your fork uses a different name.
    """
    from modules.tts.fs import FS_ENCODERS  # noqa: F401, placeholder import
    # Lazy import: build the model using the saved hparams.
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'hyper_parameters' in ckpt:
        hp = ckpt['hyper_parameters']
    else:
        hp = ckpt.get('hparams', {})
    # We use the existing SVBVAE builder to avoid re-implementing.
    from modules.voice_conversion.svb_vae import SVBVAE
    model = SVBVAE(hp)
    state_dict = ckpt.get('state_dict', ckpt)
    # Strip common 'model.' prefix if present.
    state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[probe] Loaded ckpt: {ckpt_path}")
    print(f"[probe] Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    model.eval().to(device)
    return model, hp


@torch.no_grad()
def extract_utterance_feats(model,
                            mel_batch: torch.Tensor,
                            mel_lengths: torch.Tensor,
                            device: torch.device) -> np.ndarray:
    """Run encoder on a batch of mels, return (B, 2C) utterance-level features.

    mel_batch : (B, T, n_mels)
    mel_lengths: (B,) valid lengths
    """
    mel_batch = mel_batch.to(device)
    mel_lengths = mel_lengths.to(device)

    # The original SVBVAE encoder expects (B, T, n_mels) and returns z (B, C, T).
    # This line may need adjustment for your fork -- check fvae.encode signature.
    out = model.fvae.encoder(mel_batch.transpose(1, 2))        # (B, C, T)
    if isinstance(out, tuple):
        z = out[0]
    else:
        z = out

    B, C, T = z.shape
    # Build length mask
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (idx < mel_lengths.unsqueeze(-1)).float()            # (B, T)
    mask_c = mask.unsqueeze(1)                                   # (B, 1, T)
    valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)         # (B, 1)

    mean = (z * mask_c).sum(dim=2) / valid                       # (B, C)
    var = ((z - mean.unsqueeze(-1)) ** 2 * mask_c).sum(dim=2) / valid
    std = torch.sqrt(var + 1e-6)
    feat = torch.cat([mean, std], dim=1)                         # (B, 2C)
    return feat.cpu().numpy()


def load_mels_from_binary(data_dir: str, n: int, seed: int = 0) -> list:
    """Load n (mel, length) pairs from an NSVB-format binary data dir.

    Supports both the original repo's IndexedDataset format and a plain
    directory of .npy mels. Adjust to match your binarizer output.
    """
    from utils.indexed_datasets import IndexedDataset
    ds = IndexedDataset(data_dir)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n]

    items = []
    for i in indices:
        item = ds[i]
        mel = item['mel']
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        items.append((mel.float(), mel.shape[0]))
    return items


def batch_mels(items, batch_size: int = 16):
    """Collate into padded batches."""
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
        max_t = int(lens.max().item())
        n_mels = batch[0][0].size(-1)
        padded = torch.zeros(len(batch), max_t, n_mels)
        for i, (mel, L) in enumerate(batch):
            padded[i, :L] = mel
        yield padded, lens


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def fit_logreg(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_s))
    try:
        auc = roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1])
    except ValueError:
        auc = float('nan')
    return {'test_acc': float(acc), 'test_auc': float(auc)}


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x): return self.net(x)


def fit_mlp(X_tr, y_tr, X_te, y_te, epochs: int = 50, lr: float = 1e-3,
            device: str = 'cpu'):
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = torch.from_numpy(scaler.transform(X_tr)).float().to(device)
    X_te_s = torch.from_numpy(scaler.transform(X_te)).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    model = TinyMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_s)
        loss = F.cross_entropy(logits, y_tr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_te_s).argmax(-1).cpu().numpy()
        probs = F.softmax(model(X_te_s), dim=-1)[:, 1].cpu().numpy()
    acc = accuracy_score(y_te, pred)
    try:
        auc = roc_auc_score(y_te, probs)
    except ValueError:
        auc = float('nan')
    return {'test_acc': float(acc), 'test_auc': float(auc)}


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------
def decide_case(acc: float) -> dict:
    if acc < 0.60:
        return {
            'case': 'A',
            'action': 'Run base architecture. Set use_grl=False.',
            'risk': 'Low. Dataset bias does not dominate z.',
        }
    elif acc < 0.75:
        return {
            'case': 'B',
            'action': ('Turn on GRL during Stage 1. '
                       'grl_lambda: 0.1 -> 0.3, warmup 30k steps. '
                       'Also apply L_identity_pro (weight 0.1, prob 0.2).'),
            'risk': ('Moderate dataset bias. GRL should disentangle it '
                     'from the content/quality axes.'),
        }
    else:
        return {
            'case': 'C',
            'action': ('DO NOT run Stage 2 as-is. Pick one: '
                       'C1 MOSNet re-labeling (recommended); '
                       'C2 DTW pseudo-pairs; '
                       'C3 find a Chinese karaoke paired dataset.'),
            'risk': ('High. D_z will almost certainly learn dataset identity, '
                     'not singing quality.'),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Stage 1 CVAE checkpoint')
    ap.add_argument('--m4_data_dir', required=True,
                    help='Binary data dir for M4Singer split (train).')
    ap.add_argument('--open_data_dir', required=True,
                    help='Binary data dir for OpenSinger split (train).')
    ap.add_argument('--n_per_dataset', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='outputs/phase0_probe')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, _hp = load_encoder(args.ckpt, device)

    # --- Extract features -------------------------------------------------
    print(f"[probe] Extracting M4Singer features (n={args.n_per_dataset}) ...")
    m4_items = load_mels_from_binary(args.m4_data_dir, args.n_per_dataset,
                                     seed=args.seed)
    m4_feats = []
    for mel, lens in batch_mels(m4_items, args.batch_size):
        m4_feats.append(extract_utterance_feats(model, mel, lens, device))
    m4_feats = np.concatenate(m4_feats, axis=0)

    print(f"[probe] Extracting OpenSinger features (n={args.n_per_dataset}) ...")
    op_items = load_mels_from_binary(args.open_data_dir, args.n_per_dataset,
                                     seed=args.seed)
    op_feats = []
    for mel, lens in batch_mels(op_items, args.batch_size):
        op_feats.append(extract_utterance_feats(model, mel, lens, device))
    op_feats = np.concatenate(op_feats, axis=0)

    X = np.concatenate([m4_feats, op_feats], axis=0)
    y = np.concatenate([
        np.zeros(len(m4_feats), dtype=np.int64),   # 0 = M4
        np.ones(len(op_feats), dtype=np.int64),    # 1 = OpenSinger
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # --- Fit classifiers --------------------------------------------------
    print("[probe] Fitting logistic regression ...")
    lr_res = fit_logreg(X_tr, y_tr, X_te, y_te)
    print(f"  LogReg  acc={lr_res['test_acc']:.3f}  auc={lr_res['test_auc']:.3f}")

    print("[probe] Fitting tiny MLP ...")
    mlp_res = fit_mlp(X_tr, y_tr, X_te, y_te, device=args.device)
    print(f"  MLP     acc={mlp_res['test_acc']:.3f}  auc={mlp_res['test_auc']:.3f}")

    # Use the stronger classifier's accuracy for the decision.
    best_acc = max(lr_res['test_acc'], mlp_res['test_acc'])
    decision = decide_case(best_acc)

    # --- Report -----------------------------------------------------------
    report = {
        'ckpt': args.ckpt,
        'n_per_dataset': args.n_per_dataset,
        'feature_dim': int(X.shape[1]),
        'logreg': lr_res,
        'mlp': mlp_res,
        'decision_accuracy': float(best_acc),
        'decision': decision,
    }
    out_json = Path(args.out_dir) / 'probe_report.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"PHASE 0 PROBE RESULT")
    print("=" * 60)
    print(f"Best accuracy: {best_acc:.3f}")
    print(f"Case: {decision['case']}")
    print(f"Action: {decision['action']}")
    print(f"Risk: {decision['risk']}")
    print(f"\nFull report: {out_json}")


if __name__ == '__main__':
    main()
