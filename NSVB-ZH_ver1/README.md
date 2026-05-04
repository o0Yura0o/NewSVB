# NSVB-ZH: Neural Singing Voice Beautifier for Chinese (Unpaired)

Extension of [NeuralSVB (ACL 2022)](https://github.com/MoonInTheRiver/NeuralSVB)
to Chinese singing using **unpaired** M4Singer + OpenSinger data.

> **Important**: The "M4Singer = amateur, OpenSinger = pro" assumption is
> not obviously correct. The two datasets differ in singers, song selection,
> recording environment, and mixing — not just technique. If the latent-space
> discriminator can tell datasets apart easily, it will learn dataset identity
> rather than singing quality. **Phase 0 decides whether the core assumption
> holds** before any Stage 2 training starts.

---

## Architecture

```
┌─────────────────── Stage 1 (CVAE pretrain) ──────────────────────┐
│                                                                    │
│   x  ─▶  fVAE encoder ─▶  z   ─▶  fVAE decoder  ─▶  x̂              │
│                             │                                       │
│                             ▼  (Case B only)                        │
│                       GRL  ·λ                                       │
│                             │                                       │
│                       D_domain  →  {M4, OpenSinger}                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                  ▼ checkpoint (frozen in Stage 2)

┌─────────────────── Stage 2 (mapping + GAN) ──────────────────────┐
│                                                                    │
│   x_a ─▶ encoder ─▶ z_a ─▶  M  ─▶ z_a' ─▶ decoder ─▶ mel_a'        │
│                      │           │                    │            │
│                      └── L_NCE ──┘ ── L_PPG ──────────┘            │
│                                   │                                │
│                                   ▼ conditioned on                 │
│                              (soft register, phoneme)              │
│                                   │                                │
│                                 D_z  ◀── z_p (real, OpenSinger)    │
│                                                                    │
│                                 D_mel ◀── mel_p (reused, low LR)   │
└────────────────────────────────────────────────────────────────────┘
```

**Key design choices**

| Component | Choice | Why |
|---|---|---|
| `M` | residual, kernel=1 | warp-invariant; near-identity at init |
| `D_z` | conditional on (soft F0 register, phoneme id) | prevents F0 / content shortcut |
| Soft register | 5 Gaussian bins in log-Hz, σ=0.3 | smooth gradients, perceptually natural |
| `D_mel` | reused from Stage 1, LR=1e-5 | stabilises but does not dominate |
| `L_identity_pro` | 20% of batches, weight 0.1 | prevents M from drifting on pro inputs |
| TTUR | opt_M=1e-4, opt_Dz=4e-4, opt_Dmel=1e-5 | give D_z headroom to learn |
| Stage 2 warmup | no D_z grad to M for 5k steps | M learns near-identity first |
| Stage 1 GRL | λ 0.1→0.3 over 30k (Case B only) | disentangle dataset from z |

---

## Directory layout

```
NSVB-ZH/
├── README.md                                     ← you are here
├── docs/
│   └── PHASE0_PROBE.md                           ← how to interpret probe results
├── modules/voice_conversion/
│   ├── __init__.py
│   ├── soft_bucket.py                            ← F0 → soft register
│   └── nsvb_zh_model.py                          ← ResidualM, D_z, D_domain, PatchNCE
├── tasks/singing/
│   └── nsvb_zh_task.py                           ← training task (inherits SVBVAEMleTask)
├── data_gen/singing/
│   └── binarize_nsvb_zh.py                       ← M4+OpenSinger → binary, with labels
├── scripts/
│   ├── probe_dataset_discriminability.py         ← PHASE 0 GATE
│   └── vocoder_identity_test.py                  ← PHASE 0 VOCODER CHECK
├── egs/datasets/audio/M4OpenSinger/
│   ├── base.yaml
│   ├── stage1_pretrain.yaml
│   └── stage2_mapping.yaml
└── inference/
    └── nsvb_zh_inference.py                      ← Mode A / B / C
```

All paths are relative to the original NSVB repo root. Drop these files into
your fork at the same relative paths and they will inherit existing utils,
`data_gen.singing.binarize.SingingBinarizer`, and the HifiGAN vocoder.

---

## How to run

### 0. Install

Use the original NSVB environment. Additional requirements:

```bash
pip install scikit-learn scikit-image pyworld
```

### 1. Binarise

```bash
# Expects raw_data_dir/{m4singer,opensinger}/... under raw_data_dir
python data_gen/singing/binarize.py \
    --config egs/datasets/audio/M4OpenSinger/base.yaml
```

### 2. Stage 1 pretrain (half-cook, ~50–80k steps)

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/datasets/audio/M4OpenSinger/stage1_pretrain.yaml \
    --exp_name nsvb_zh_stage1
```

### 3. **PHASE 0 GATE** — decide Case A / B / C

**Probe (required):**

```bash
python scripts/probe_dataset_discriminability.py \
    --ckpt    checkpoints/nsvb_zh_stage1/model_ckpt_steps_80000.ckpt \
    --m4_data_dir   data/binary/m4opensinger/train_m4 \
    --open_data_dir data/binary/m4opensinger/train_open \
    --n_per_dataset 1000 \
    --out_dir outputs/phase0_probe
```

Decision table:

| Probe accuracy | Case | Action |
|---|---|---|
| `< 0.60` | **A** | Keep `use_grl: false`, continue Stage 1 to convergence |
| `0.60 – 0.75` | **B** | Set `use_grl: true` in stage1 YAML, restart / finetune Stage 1 |
| `≥ 0.75` | **C** | **Stop**. Re-label via MOSNet (C1) / DTW pseudo-pairs (C2) / swap to Chinese karaoke dataset (C3) |

**Vocoder identity check (required):**

```bash
python scripts/vocoder_identity_test.py \
    --vocoder_ckpt checkpoints/0109_hifigan_bigpopcs_hop128/model_best.ckpt \
    --wav_dirs m4=data/raw/m4singer open=data/raw/opensinger \
    --n_per_dir 20 --save_wavs \
    --out_dir outputs/phase0_vocoder
```

Pass: mel SSIM ≥ 0.90 **and** F0 RMSE ≤ 10 Hz on both datasets. If fail,
fine-tune the vocoder on Chinese singing before proceeding.

### 4. Full Stage 1 (gate: val mel SSIM ≥ 0.85)

Depending on probe outcome, keep `use_grl` off (Case A) or on (Case B). Train
to convergence.

### 5. Stage 2 mapping

Update `load_ckpt` in `stage2_mapping.yaml` to point at your best Stage 1
ckpt, then:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/datasets/audio/M4OpenSinger/stage2_mapping.yaml \
    --exp_name nsvb_zh_stage2
```

Monitor `_delta_over_z` in the training log. If it stays below 3 % after 30k
steps, switch `m_kernel_size: 3` in the config — M is too conservative.

### 6. Inference

```bash
python inference/nsvb_zh_inference.py \
    --config egs/datasets/audio/M4OpenSinger/stage2_mapping.yaml \
    --stage2_ckpt checkpoints/nsvb_zh_stage2/model_best.ckpt \
    --vocoder_ckpt checkpoints/0109_hifigan_bigpopcs_hop128/model_best.ckpt \
    --mode A \
    --src_wav demo/amateur.wav \
    --out_wav demo/enhanced.wav
```

---

## What is NOT in this repo

- **MOSNet re-labeling (C1)** — requires an external MOSNet / UTMOS ckpt.
  Add `scripts/mosnet_relabel.py` that scores every clip and emits a new
  `dataset_label` in {0 (bottom-quartile), 1 (top-quartile)}.
- **DTW pseudo-pairing (C2)** — the original NSVB `preprocess_align.py` can
  be repurposed; contribution framing changes to "soft-supervised Chinese
  NSVB".
- **Chinese karaoke dataset ingest (C3)** — would need a singer-disjoint
  test split design.

---

## Things you must adjust for your fork

1. `phoneme_vocab_size` in YAML — must match your zh txt_processor output.
2. `ppg_extractor_ckpt` — path to a Chinese PPG model (Whisper / WeNet based).
3. `load_ckpt` in stage2 YAML — path to your Stage 1 best ckpt.
4. `modules.tts.fs.FS_ENCODERS` import in the probe script — change if your
   encoder path differs.
5. `self.model.fvae.encoder(...)` call signature in `run_encoder` — verify
   against your fork's fVAE API.

---

## Licence

Inherits from the upstream NeuralSVB licence. This extension is provided
as-is for research use.
