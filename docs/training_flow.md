# NSVB-ZH 訓練流程文件

本文件涵蓋從**原始 wav 資料**到**訓練完成的 Stage 2 ckpt** 的完整 pipeline，
含每個處理步驟的 input/output tensor shapes、實作細節、與訓練監控指標。

> 配套文件：[rebuild_checklist.md](../rebuild_checklist.md)（架構決策）、
> [risk.md](../risk.md)（風險與 Monitor 列表）

---

## 0. 高層 pipeline 概覽

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 0  資料二進位化  (data/{m4singer,VocalVerse}/*.wav  →  .npz)          │
│   binarizer.py + cluster_ppg.py                                              │
│                                                                              │
│ Phase 0 gate ① JSD(register / phoneme) < 0.05                                │
│ Phase 0 gate ② Vocoder identity test  SSIM ≥ 0.85, F0 RMSE ≤ 20 Hz           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 1  Stage 1 CVAE 預訓練  (.npz → φ θ D_mel ckpt)                        │
│   nsvb/task/stage1.py                                                        │
│   model: SVBVAEZh (FVAE 88/88 weights from NSVB 1030_vae_mle ckpt)           │
│   loss:  L_l1 + L_l2 + β·L_KL  (+ 0.1·L_adv_mel)                              │
│                                                                              │
│ Phase 1 gate  val mel L1 < 0.15 + 聽測樣本品質                                 │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 2  Stage 2 Mapping 訓練  (φ θ frozen, train M + D_z, refine D_mel)     │
│   nsvb/task/stage2.py                                                        │
│   model: ResidualM + DiscriminatorZ + reused D_mel                           │
│   loss:  L_PatchNCE + L_adv_z (warmup 5k) + 0.2·L_adv_mel (+ 0.1·L_id_pro)    │
│                                                                              │
│ Phase 2 monitor  D_z accuracy 0.55-0.75, ‖Δ‖/‖z‖ ratio                        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 3  推理  (Mode A 自動 / Mode B 完全參考；尚未實作)                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Phase 0 — 資料二進位化

### 1.1 輸入

```
data/m4singer/{歌手#歌名}/{idx}.wav        # 業餘? 不，職業（M4Singer 是錄音室專業歌手）
data/VocalVerse/{user_id}/{wav_id}.wav     # 業餘
```

| 資料集 | 檔案格式 | sample rate | mono/stereo | 角色 |
|---|---|---:|---|---|
| M4Singer | wav int16 | 48000 | mono | **professional (z_p)** |
| VocalVerse | wav int16 | 44100 | stereo | **amateur (z_a)** |

> ⚠ 載入時 `librosa.load(sr=22050, mono=True)` 自動 resample + stereo→mono averaging。

### 1.2 抽取的特徵（per song → 一個 .npz）

**腳本**：`nsvb/data/binarizer.py`，sequential GPU 處理。

| Feature | Shape | Dtype | 抽法 / 工具 | 為什麼 |
|---|---|---|---|---|
| `wav` | `[N_samples]` | float32 | librosa.load + loudness norm + pad align | HifiGAN 需要 |
| `mel` | `[T_mel, 80]` | float32 | librosa.stft + log10(max(1e-10, mel))，**fmin=50 fmax=11025** ⚠ | NSVB inference 實際分布 |
| `f0` | `[T_mel]` | float32 | torchcrepe (full model, fmin=50, fmax=1100, viterbi=False) | 中文聲調精度最佳 |
| `voicing` | `[T_mel]` | float32 | torchcrepe periodicity | unvoiced 判定 |
| `register_soft` | `[T_mel, 5]` | float32 | F0 → 5 個 Gaussian bucket（σ=0.3 log-Hz, centers C3/G3/D4/A4/E5） | D_z 軟條件，防 F0 shortcut |
| `register_id` | `[T_mel]` | int8 | argmax(register_soft) | JSD 統計用 |
| `ppg` | `[T_mel, 1280]` | **float16** | Whisper-large-v3 encoder layer 8 hidden state，resample 50→172 fps | language-agnostic content |
| `spk_emb` | `[256]` | float32 | Resemblyzer L2-normed | decoder 音色錨 (Risk 4 防護) |
| `phoneme_id` | `[T_mel]` | int16 | k-means(K=200) on PPG via `cluster_ppg.py` | D_z 離散音素條件 |
| `meta_*` | scalar | str/int32 | dataset / speaker_id / item_id / sr / hop | 不變數據 |

### 1.3 audio config（單一 source of truth）

`nsvb/utils/audio_config.py`：

```python
SAMPLE_RATE  = 22050
HOP_SIZE     = 128       # 對齊 NSVB 1012 vocoder ckpt
FFT_SIZE     = 512
WIN_SIZE     = 512
NUM_MELS     = 80
MEL_FMIN     = 50        # ⚠ NSVB inference 實際值（非 ckpt config 寫的 0）
MEL_FMAX     = 11025     # 同上
MEL_EPS      = 1e-10
LOUDNESS_TARGET_LUFS = -22.0

FRAME_RATE_HZ        = 22050 / 128 = 172.27 fps  (mel)
LATENT_DOWN_FACTOR   = 4
LATENT_FRAME_RATE_HZ = 172.27 / 4  = 43.07 fps  (z, latent)
```

> 為什麼這些值：見 [rebuild_checklist.md §J' 守則 2-3](../rebuild_checklist.md)。

### 1.4 PPG k-means 分群（Phase 0 第二階段）

**腳本**：`nsvb/data/cluster_ppg.py`，兩階段：
1. **Stage A**：對所有 .npz 各抽 200 frames PPG → fit `MiniBatchKMeans(K=200)` → 存 centroids
2. **Stage B**：對每個 .npz 用 centroids 算每 frame 的 phoneme_id → 寫回 .npz

| 參數 | 值 | 為什麼 |
|---|---|---|
| K | 200 | 中文音素 ~80（不分聲調），但 Whisper hidden state 含 prosody/共發音/音高效應，K=200 給更純的 cluster |
| frames per song | 200 | 10000 首 × 200 = 2M frames，足以收斂 K=200 |
| algorithm | MiniBatchKMeans | 全 frames ~250M 太大，stream batch 8192 fp32 |

### 1.5 Phase 0 監控與 gate

#### Gate ①：資料策展 JSD

**腳本**：`nsvb/utils/jsd_check.py`。

| 指標 | 閾值 | 不過怎麼辦 |
|---|---|---|
| JSD(VocalVerse phoneme dist, M4Singer phoneme dist) | < 0.05 | 重採樣 phoneme distribution |
| JSD(VocalVerse register dist, M4Singer register dist) | < 0.05 | 重採樣 register distribution |

**為什麼必要**：D_z 用 phoneme + register 當條件；若兩 dataset 的 phoneme 頻率分布有大差別（例如 M4Singers 比例壓倒性高男聲），D_z 看到「phoneme 出現比例」就能猜業餘/職業，把 M 引導到亂改。

#### Gate ②：Vocoder identity test

**腳本**：`scripts/vocoder_identity_test.py`。

| 指標 | PASS 閾值 | MARGINAL | FAIL |
|---|---|---|---|
| mel SSIM | ≥ 0.90 | 0.85–0.90 | < 0.85 |
| F0 RMSE (Hz) | ≤ 10 | 10–20 | > 20 |

**為什麼必要**：vocoder 是凍結 ckpt；若它對中文歌聲重建已經斷裂（mel→wav 失真），後續 M 任何改進都看不到。

**目前實測**（origin GT 22050 native，pyworld + interp）：
- SSIM = 0.94, F0 RMSE = 5.8 Hz → **PASS** ✅

---

## 2. Phase 1 — Stage 1 CVAE 預訓練

### 2.1 訓練目標

學會「**給定 condition（PPG + F0 + spk_emb），用 mel 重建 mel**」。輸出：訓好的 φ（encoder）、θ（decoder）、與基礎 D_mel。

> Stage 1 兩 dataset (m4singer + VocalVerse) 都當 real，**不分業餘/職業**。
> 因為這階段目標是學「自然人聲分布」；分流會破壞 latent space 結構。

### 2.2 模型 SVBVAEZh

**檔案**：`nsvb/model/svb_vae_zh.py`

```
                  ┌──────────────────┐
ppg [B,T,1280] ──▶│  PPG_proj         │ ─► [B, T, 128]
                  │  (Linear)         │
                  └──────────────────┘
                  ┌──────────────────┐
f0  [B,T]      ──▶│  LogF0Embed       │ ─► [B, T, 32]
                  │  (log2 + Linear)  │
                  └──────────────────┘                       g [B, 256, T_mel]  (channel-first)
                  ┌──────────────────┐                       │
spk [B,256]    ──▶│  spk_proj         │ ─► [B, T, 96] ┐ concat │
                  │  (Linear+expand)  │                ├──────►│
                  └──────────────────┘                ┘       ▼
                                                      ┌──────────────────────┐
mel [B,T,80] ────────────────────────────────────────▶│  FVAE encoder φ      │
                                                      │   (stride=4 down)     │
                                                      ▼
                                                z [B,128, T_z = T_mel/4]
                                                      ▼
                                                      ┌──────────────────────┐
                                                      │  FVAE decoder θ      │
                                                      │   (stride=4 up + WN) │
                                                      └──────────────────────┘
                                                      ▼
                                            mel_recon [B, T_mel, 80]
```

**參數預設（與 NSVB 1030_vae_mle ckpt 對齊）**：

| 模組 | 設定 |
|---|---|
| ppg_proj_dim | 128 |
| pitch_emb_dim | 32 |
| spk_proj_dim | 96 |
| **gin_channels (g 通道數)** | **256**（總和） |
| FVAE hidden_size | 192 |
| FVAE latent_size | 128 |
| FVAE strides | [4]（mel 172 fps × 4 = latent 43 fps）|
| FVAE kernel_size | 5 |
| FVAE enc_n_layers | 8 |
| FVAE dec_n_layers | 4 |
| 模型總 params | **7.44 M** |

> ⚠ **NSVB transfer learning**：`nsvb/utils/transfer_weights.py` 把 NSVB
> `1030_vae_mle/model_ckpt_steps_200000.ckpt` 的 `vae_model.*` 88 個 keys
> bit-exact 注入我們的 SVBVAEZh.fvae（驗證 88/88 loaded, 0 mismatch）。
> 用 `--init-from-nsvb` 啟用，預期可省 ~5x training 時間。

### 2.3 訓練流程（每 step）

**檔案**：`nsvb/task/stage1.py:Stage1Trainer.train_step`

```
batch (DataLoader 從 ConcatDataset[m4singer, VocalVerse] 抽)
   {mel, f0, voicing, register_soft, register_id, ppg, spk_emb,
    phoneme_id, mel_mask, ...}

Step 1:  Update D_mel  (if use_d_mel)
   ─ no_grad 跑 G 拿 fake_mel
   ─ real_score = D_mel(real_mel)
   ─ fake_score = D_mel(fake_mel.detach())
   ─ d_loss = hinge_d_loss(real_score, fake_score)
   ─ opt_d.step()

Step 2:  Update G (CVAE)
   ─ out = SVBVAEZh(mel, ppg, f0, spk_emb, mel_mask)
   ─ L_l1 = masked_l1(mel_recon, mel)
   ─ L_l2 = masked_l2(mel_recon, mel)
   ─ β    = kl_warmup(step, target=0.01, warmup=5000)
   ─ L_total = 1.0·L_l1 + 1.0·L_l2 + β·KL  (+ 0.1·L_adv_mel)
   ─ opt_g.zero_grad();  L_total.backward();  clip_grad(1.0);  opt_g.step()
```

### 2.4 Loss 細節

| Loss | 權重 | 公式 / 用意 |
|---|---:|---|
| `L_l1` | 1.0 | masked L1(mel_recon, mel)；對整體 dB 偏差敏感 |
| `L_l2` | 1.0 | masked L2(mel_recon, mel)；對 outlier 敏感 |
| `L_KL` | β linear warmup 0→0.01 over 5000 步 | FVAE 內部 closed-form `KL(q‖N(0,I))`；warmup 防 posterior collapse |
| `L_adv_mel` | 0.1 (optional) | hinge_g_loss(D_mel(mel_recon))；提升自然度 |
| `L_d_mel` | (D_mel 自己) | hinge_d_loss(D_mel(real), D_mel(fake.detach())) |

### 2.5 Optimizer

| optim | LR | β | 對象 |
|---|---:|---|---|
| opt_g | 2e-4 | (0.9, 0.98) | SVBVAEZh.parameters() |
| opt_d | 2e-4 | (0.5, 0.999) | D_mel.parameters() |

> 用 `--init-from-nsvb` 時 lr_g = 2e-4 × `init_lr_scale`(0.5) = **1e-4**，
> 避免大 lr 破壞預訓表徵。

### 2.6 訓練監控指標

| 指標 | 期望範圍 | 不對勁時的解讀 |
|---|---|---|
| `l_l1` | 訓練後 ~0.05–0.15 | > 0.3 持續：condition 餵錯，或 mel 公式不對齊 vocoder |
| `l_l2` | 訓練後 ~0.05 | 同上 |
| `kl` | warmup 後 ~ 1–10 / latent_dim | **kl=0 (collapse)** → β 太大；kl 超大 → β 太小，z 容量爆 |
| `kl_beta` | 線性升 0→0.01 | warmup 機制檢查 |
| `l_adv_g` | ~ -0.5 to 0 | 大幅 > 0：D 太弱；大幅 < -1：M 學會欺騙 D，G overpowers |
| `l_d` | ~ 0.5–1.5 | < 0.3：D 太弱（real/fake 都信）；> 2：D 太強 |

### 2.7 ckpt 保存格式

`checkpoints/stage1/stage1_*.pt`:

```python
{
    'step': int,
    'epoch': int,
    'model': SVBVAEZh.state_dict(),
    'd_mel': D_mel.state_dict(),
    'opt_g': optim state dict,
    'opt_d': optim state dict,
    'config': Stage1Config dict,
}
```

### 2.8 執行命令

```bash
PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 \
    --batch-size 32 \
    --max-steps 30000 \
    --num-workers 4 \
    --init-from-nsvb /path/to/1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1
```

---

## 3. Phase 2 — Stage 2 Mapping 訓練

### 3.1 訓練目標

凍結 Stage 1 訓好的 φ/θ，訓練一個 **ResidualM** 把業餘 latent z_a 推到 pro 風格。
output：訓好的 M、D_z、refined D_mel ckpt。

### 3.2 模型構成

**新增**：

#### ResidualM（`nsvb/model/m_mapping.py`）
```
z [B,128,T_z]  ─►  Δ ─► add ──► z + Δ(z) [B,128,T_z]
                    │
                    └─ Conv1d(128→256, k=1) → GroupNorm → GELU
                       Conv1d(256→256, k=1) → GroupNorm → GELU
                       Conv1d(256→256, k=1) → GroupNorm → GELU
                       Conv1d(256→128, k=1)  ← 最後 init std=1e-2 → Δ ≈ 0
```

| 參數 | 值 |
|---|---|
| latent_dim | 128 |
| hidden_dim | 256 |
| kernel_size | **1**（warp-invariant for Mode B） |
| num_layers | 4 |
| init_delta_scale | 1e-2 |
| 總 params | **0.20 M** |

#### DiscriminatorZ（`nsvb/model/d_z.py`）
```
z [B,128,T_z]              ┐
soft_register [B,T_z,5]    ├ concat ─► [B, 128+5+32, T_z]
phoneme_emb [B,T_z,32]     ┘    │
       ▲                          ▼
       │                    Conv1d(165→256, k=5) + spectral_norm + LeakyReLU(0.2)
phoneme_id +1 (pad=0)        × 4 layers
                                  │
                                  ▼
                            Conv1d(256→1, k=1) + spectral_norm
                                  │
                                  ▼
                            score [B, 1, T_z]   per-frame logit
```

| 參數 | 值 |
|---|---|
| phoneme_vocab_size | 200 (k-means cluster 數) |
| phoneme_embed_dim | 32 |
| hidden_dim | 256 |
| num_layers | 4 |
| kernel_size | 5 (43 fps × 5 ≈ 116 ms 局部上下文) |
| 總 params | **1.20 M** |

#### PatchNCELoss（`nsvb/model/losses.py`）
```
z, M(z) ─► proj head (Conv1d 128→64 → GELU → Conv1d 64→64) ─► [B, 64, T]
            ▼
         random sample 128 frames per batch item
            ▼
         L = cross_entropy(cosine_sim(q_i, k_j) / 0.07,
                          target=diagonal i=j)
```

`patchnce_proj_dim=64, num_patches=128, temperature=0.07`，與 ver1 一致。

#### D_mel（**重用 Stage 1**, real 改為只看 pro）

從 Stage 1 ckpt 載入 D_mel；real 端餵 pro mel，fake 端餵 `θ(M(z_a))` decode 出的 mel；
低 lr 1e-5 微調。

### 3.3 訓練流程（每 step）

**檔案**：`nsvb/task/stage2.py:Stage2Trainer.train_step`

```
從兩個獨立 dataloader 各抽 batch (itertools.cycle)：
  ba = next(iter_a)  # VocalVerse (amateur)
  bp = next(iter_p)  # M4Singer    (professional)

Step 1: 凍結 φ 算 z_a, z_p（用 m_q posterior mean，deterministic）
  side_a = encode_and_downsample(ba)
  side_p = encode_and_downsample(bp)
  ─ z [B,128,T_z],  register_z [B,T_z,5],  phoneme_z [B,T_z],  mask_z [B,T_z]

Step 2: Update D_z
  ─ z_a_mapped_detach = no_grad M(z_a)
  ─ real_score = D_z(z_p, side_p.register, side_p.phoneme)
  ─ fake_score = D_z(z_a_mapped.detach(), side_a.register, side_a.phoneme)
  ─ d_z_loss = hinge_d_loss(real, fake)
  ─ opt_dz.step()

Step 3: Update D_mel (real = pro mel only)
  ─ no_grad 算 fake_mel = decode(M(z_a))
  ─ real_score = D_mel(bp.mel)
  ─ fake_score = D_mel(fake_mel.detach())
  ─ d_mel_loss = hinge_d_loss(real, fake)
  ─ opt_dmel.step()

Step 4: Update M (含 PatchNCE projection head 同步更新)
  ─ z_a_mapped = M(z_a)
  ─ L_PatchNCE  (z_a vs M(z_a), batch-internal negatives)
  ─ if step >= warmup (5000):
        L_adv_z = hinge_g_loss(D_z(M(z_a), side_a.register, side_a.phoneme))
  ─ L_adv_mel = hinge_g_loss(D_mel(decode(M(z_a))))
  ─ if rand() < 0.2:
        L_id_pro = L1(M(z_p), z_p)        # 隨機 20% 抽中時加進來
  ─ L_total = 1.0·L_NCE + 1.0·L_adv_z + 0.2·L_adv_mel + 0.1·L_id_pro
  ─ opt_m.step()
```

### 3.4 Loss 權重表

```
M 的更新（每 step）：
  L_M = 1.0  · L_adv_z       (warmup 前 = 0)
      + 1.0  · L_PatchNCE
      + 0.2  · L_adv_mel
      + 0.1  · L_identity_pro    (20% batches 抽中時)
      (+ 0.0 · L_PPG, TODO 等 fast mel→PPG predictor)

D_z 的更新：
  ReLU(1 - D_z(z_p, reg_p, ph_p)).mean()
  + ReLU(1 + D_z(M(z_a).detach(), reg_a, ph_a)).mean()

D_mel 的更新（pro real only）：
  ReLU(1 - D_mel(mel_p)).mean()
  + ReLU(1 + D_mel(decode(M(z_a)).detach())).mean()
```

### 3.5 Optimizer (TTUR — Heusel 2017)

| optim | LR | β | 對象 |
|---|---:|---|---|
| opt_m | **1e-4** | (0.5, 0.999) | M.parameters() + PatchNCE.parameters() |
| opt_dz | **4e-4** (4× M) | (0.5, 0.999) | D_z.parameters() |
| opt_dmel | **1e-5** (微調) | (0.5, 0.999) | D_mel.parameters() |

> 為什麼 4× M：給 D_z 足夠 headroom 學識別，M 才有清晰梯度方向；NSVB 慣例。

### 3.6 訓練監控指標

| 指標 | 期望範圍 | 解讀 |
|---|---|---|
| `l_nce` | 收斂 ~ 1.0–3.0 | 過低（< 0.5）：M ≈ identity 沒在動；過高（> 5）：M 把內容打亂了 |
| `l_adv_z` | warmup 後 ~ -0.5 to 0 | 持續 > 0：D_z 一直贏，M 學不動；持續 < -1：M 騙過 D_z（要看 D_z accuracy） |
| `l_adv_mel` | ~ -0.3 to 0 | 同上邏輯 |
| `l_id_pro` | 20% 步抽中時 ~ 0.05–0.2 | M 在 pro 端沒亂改 |
| **`d_z`** | **~ 0.5–1.5** | < 0.3：D_z 太弱（real fake 都信，M 不需學）；> 2：D_z 太強 M 學不動 |
| `d_mel` | ~ 0.5–1.5 | 同上 |
| **`delta_over_z`** ⭐ | **0.03–0.20** | 這是「**M 漂移程度**」核心指標：< 0.03 → M 太保守，考慮切 kernel=3；> 0.30 → M 過度激進，可能破壞內容 |

#### D_z accuracy（外掛 monitor，可選）

每幾百步抽一次 batch 計算 `D_z(real) > 0` 與 `D_z(fake) < 0` 的比例：
- 0.55–0.75 = 健康（D_z 略勝但 M 仍在學）
- > 0.85 = D_z 過強（增 lr_M 或減 lr_dz）
- < 0.55 = D_z 過弱（多訓 D_z 幾步）

### 3.7 ckpt 保存格式

`checkpoints/stage2/stage2_*.pt`:

```python
{
    'step': int,
    'M': ResidualM.state_dict(),
    'D_z': DiscriminatorZ.state_dict(),
    'D_mel': D_mel.state_dict(),
    'patchnce': PatchNCELoss.state_dict(),
    'opt_m / opt_dz / opt_dmel': optim states,
    'config': Stage2Config dict,
    'stage1_ckpt': str  (Stage 1 ckpt 路徑記錄)
}
```

> 注意：CVAE φ/θ 不存（因為凍結，從 stage1_ckpt path 取得）。

### 3.8 執行命令

```bash
PYTHONPATH=. python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --amateur-dataset vocalverse \
    --pro-dataset m4singer \
    --ppg-dim 1280 \
    --phoneme-vocab-size 200 \
    --batch-size 24 \
    --max-steps 120000 \
    --num-workers 4 \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --ckpt-dir checkpoints/stage2 \
    --m-kernel-size 1
```

---

## 4. Tensor shape 速查表

### 4.1 Frame rate 三段對照

| 域 | fps | 換算 |
|---|---:|---|
| audio (wav) | 22050 | 1 sample = 1/22050 sec |
| mel | 172.27 | hop=128, 1 frame = 5.8 ms |
| latent z | 43.07 | strides=4, 1 frame = 23.2 ms |

### 4.2 訓練時 batch 內 tensor 形狀（max_frames=600）

```
T_mel (max in batch)       例：600
T_z   (= floor((T_mel-4)/4)+1)  例：150
B (batch_size)             例：16

# Mel rate 系列
mel              [B, T_mel, 80]            float32
ppg              [B, T_mel, 1280]          float32  (從 fp16 上轉)
f0               [B, T_mel]                float32  Hz
voicing          [B, T_mel]                float32
register_soft    [B, T_mel, 5]             float32
register_id      [B, T_mel]                int64    -1=padding
phoneme_id       [B, T_mel]                int64    -1=padding/未分群
mel_mask         [B, T_mel]                float32  1=valid 0=pad
spk_emb          [B, 256]                  float32  L2-normed

# Latent rate 系列（Stage 2 用）
z                [B, 128, T_z]             float32  φ(mel) m_q
M(z)             [B, 128, T_z]             float32
register_z       [B, T_z, 5]               float32  從 register_soft interpolate
phoneme_z        [B, T_z]                  int64    從 phoneme_id stride sample
mask_z           [B, T_z]                  float32

# 條件 g（輸入 FVAE 內部）
g                [B, 256, T_mel]           float32  channel-first
g_sqz (encoder 用) [B, 256, T_z]           float32  經 g_pre_net 下採

# Stage 2 Loss 中間 tensor
D_z(z, reg_z, ph_z)  [B, 1, T_z]           float32  per-frame logit
D_mel(mel)           y: [B, 1]             float32  hinge score sum
PatchNCE proj        [B, 64, T]            float32  query/key
```

---

## 5. 重要 hyperparameter 速查

### 5.1 Audio config（不能動，動了與 vocoder 不相容）
```
SR=22050, hop=128, fft=512, win=512, n_mels=80, fmin=50, fmax=11025, eps=1e-10
```

### 5.2 Stage 1（CVAE）
```
gin_channels=256 (= ppg_proj 128 + pitch_emb 32 + spk_proj 96)
hidden=192, latent=128, strides=[4], enc_layers=8, dec_layers=4, kernel=5
loss weights: l1=1 l2=1 kl_target=0.01(warmup 5k) adv_mel=0.1
optim: lr_g=2e-4 (init from NSVB → ×0.5), lr_d=2e-4
batch=16-32, max_frames=600, max_steps=30k (with NSVB init) ~ 200k (scratch)
```

### 5.3 Stage 2（Mapping）
```
M:    kernel=1, hidden=256, num_layers=4, init_delta_scale=1e-2
D_z:  hidden=256, num_layers=4, kernel=5, vocab=200, embed=32
PatchNCE: proj=64, num_patches=128, temp=0.07
loss weights: NCE=1.0, adv_z=1.0(after warmup 5k), adv_mel=0.2, id_pro=0.1@0.2_prob
optim TTUR: lr_M=1e-4, lr_Dz=4e-4, lr_Dmel=1e-5  (β=(0.5, 0.999))
batch=16-32, max_frames=600, max_steps=120k
```

---

## 6. 已知 limitation 與 TODO

| 項目 | 狀態 | 何時實作 |
|---|---|---|
| L_PPG | TODO（lambda=0 預設） | 後續加 fast mel→PPG predictor 模組後啟用 |
| Mode A / Mode B 推理 | TODO | todo #9 |
| F0 unvoiced 內插（餵 vocoder 必要） | helper 已寫於 `nsvb/utils/f0_utils.py:interp_f0_unvoiced` | 推理腳本接入 |
| Multi-GPU DDP | TODO | 視訓練時間需求加 |
| AMP / torch.compile | TODO | 訓練機 benchmark 後決定是否開 |

---

## 7. Phase 0 / 1 / 2 整體訓練命令範例（summary）

```bash
# === Phase 0：資料二進位化 ===
PYTHONPATH=. python -m nsvb.data.binarizer --dataset m4singer
PYTHONPATH=. python -m nsvb.data.binarizer --dataset vocalverse

# Phase 0 第二階段：k-means 分群 phoneme_id
PYTHONPATH=. python -m nsvb.data.cluster_ppg \
    --binarized-root data/binarized --k 200 --stage all

# Phase 0 gate：JSD 與 vocoder identity test
PYTHONPATH=. python -m nsvb.utils.jsd_check  # 檢查 phoneme/register JSD < 0.05
PYTHONPATH=. python scripts/vocoder_identity_test.py \
    --vocoder-ckpt /path/to/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 --f0-method pyworld --f0-interp \
    --out-dir outputs/phase0_vocoder

# === Phase 1：Stage 1 CVAE 預訓練 ===
PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 --batch-size 32 --max-steps 30000 --num-workers 4 \
    --init-from-nsvb /path/to/1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1

# === Phase 2：Stage 2 Mapping 訓練 ===
PYTHONPATH=. python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --amateur-dataset vocalverse --pro-dataset m4singer \
    --ppg-dim 1280 --phoneme-vocab-size 200 \
    --batch-size 24 --max-steps 120000 --num-workers 4 \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --ckpt-dir checkpoints/stage2 --m-kernel-size 1
```