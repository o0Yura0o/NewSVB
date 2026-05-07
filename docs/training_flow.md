# NSVB-ZH 訓練流程文件

本文件涵蓋從**原始 wav 資料**到**推理輸出 wav** 的完整 pipeline，
含每個處理步驟的 input/output tensor shapes、實作細節、與訓練監控指標。

> 配套文件：
> - [rebuild_checklist.md](../rebuild_checklist.md) — 架構決策
> - [risk.md](../risk.md) — 風險清單與 Monitor
> - [deployment_linux.md](deployment_linux.md) — 從零部署到 Linux GPU 機器的完整指南

---

## 0. 高層 pipeline 概覽

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 0  資料二進位化  (data/{m4singer,VocalVerse}/*.wav  →  .npz)          │
│   binarizer.py (含 dereverb) + cluster_ppg.py                                │
│                                                                              │
│ Phase 0 gate ① Vocoder identity test  SSIM ≥ 0.90, F0 RMSE ≤ 10 Hz           │
│ Phase 0 gate ② Audio quality probe    SFM/Reverb/HF/SNR JSD < 0.10           │
│ Phase 0 gate ③ JSD(register / phoneme) < 0.05                                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 1  Stage 1 CVAE 預訓練  (.npz → φ θ D_mel ckpt)                        │
│   nsvb/task/stage1.py  (含 tqdm + --resume + --init-from-nsvb)                │
│   model: SVBVAEZh (FVAE 88/88 weights from NSVB 1030_vae_mle ckpt)           │
│   loss:  L_l1 + L_l2 + β·L_KL  (+ 0.1·L_adv_mel)                              │
│                                                                              │
│ Phase 1 gate  val mel L1 < 0.15 + 聽測樣本品質                                 │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 2  Stage 2 Mapping 訓練  (φ θ frozen, train M + D_z, refine D_mel)     │
│   nsvb/task/stage2.py  (含 tqdm + --resume + --dmel-mix-amateur-real)         │
│   model: ResidualM + DiscriminatorZ + reused D_mel                           │
│   loss:  L_PatchNCE + L_adv_z (warmup 5k) + 0.2·L_adv_mel (+ 0.1·L_id_pro)    │
│                                                                              │
│ Phase 2 monitors:                                                            │
│   - D_z accuracy 0.55–0.75, ‖Δ‖/‖z‖ ratio                                    │
│   - Risk 2 L5: monitor_audio_quality 每 5000 步 (unvoiced_concentration)     │
│   - Auto-warning: kernel=1 + Δ/z<0.03 @ step 30000                           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Phase 3  推理 (已實作)                                                        │
│   nsvb/inference/ + scripts/infer.py                                          │
│   Mode A：x_a → φ → M → θ → mel → vocoder    (T_a 對齊原伴奏)                │
│   Mode B：+ DTW(mel_a, mel_p_ref) + gather warp 到 T_p (跟隨 pro 模板)        │
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

**載入與前處理 pipeline**（`audio_io.load_and_extract`）：
```
load_wav (librosa, sr=22050, mono)
  → dereverb_wav (DeepFilterNet3) ⭐ Risk 2 主防線，預設開
  → loudness_normalize (pyloudnorm BS.1770, -22 LUFS)
  → compute_mel (librosa STFT + log10-mel)
  → pad_wav_to_mel_length (對齊 wav 與 mel frame)
```

> 為什麼 dereverb 對兩個 dataset 都開（不只 amateur）：避免引入「有沒有過 dereverb」這個新差異成為 D_z 的捷徑。M4Singer 雖是錄音室，DeepFilterNet 在乾淨輸入上近乎 noop。
> **`--no-dereverb` 僅供 vocoder identity test / smoke test 使用**；production 訓練嚴禁關掉。

| Feature | Shape | Dtype | 抽法 / 工具 | 為什麼 |
|---|---|---|---|---|
| `wav` | `[N_samples]` | float32 | dereverb + loudness norm + pad align | HifiGAN 需要 |
| `mel` | `[T_mel, 80]` | float32 | librosa.stft + log10(max(1e-10, mel))，**fmin=50 fmax=11025** ⚠ | NSVB inference 實際分布 |
| `f0` | `[T_mel]` | float32 | torchcrepe (full model, **fmin=50, fmax=1400**, viterbi=False) | 覆蓋到 F6=1397 Hz（流行女聲 belt 高潮安全邊界） |
| `voicing` | `[T_mel]` | float32 | torchcrepe periodicity | unvoiced 判定 |
| `register_soft` | `[T_mel, 5]` | float32 | F0 → 5 個 Gaussian bucket（σ=0.3 log-Hz, centers C3/G3/D4/A4/E5） | D_z 軟條件，防 F0 shortcut |
| `register_id` | `[T_mel]` | int8 | argmax(register_soft) | JSD 統計用 |
| `ppg` | `[T_mel, 1280]` | **float16** | Whisper-large-v3 encoder layer 8 hidden state，resample 50→172 fps | language-agnostic content |
| `spk_emb` | `[256]` | float32 | Resemblyzer L2-normed | decoder 音色錨 (Risk 4 防護) |
| `phoneme_id` | `[T_mel]` | int16 | k-means(K=200) on PPG via `cluster_ppg.py` | D_z 離散音素條件 |
| `meta_*` | scalar | str/int32 | dataset / speaker_id / item_id / sr / hop | 不變數據 |

> **重要**：[audio_config.py](../nsvb/utils/audio_config.py) 任一參數修改後，所有已存在的 `.npz` 必須**全部重 binarize**（特徵值已 freeze 進檔，舊檔不會自動更新）。

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

F0_FMIN      = 50.0       # 男低音 C2 緩衝
F0_FMAX      = 1400.0     # ⭐ 覆蓋到 F6=1397 Hz（流行女聲 belt 高潮安全邊界）

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

> **三個 gate 都要 PASS** 才能進 Phase 1。順序建議：先 ①（vocoder 不過後續都白做），再 ②（音質基線），最後 ③（資料策展）。

#### Gate ①：Vocoder identity test

**腳本**：`scripts/vocoder_identity_test.py`。

| 指標 | PASS 閾值 | MARGINAL | FAIL |
|---|---|---|---|
| mel SSIM | ≥ 0.90 | 0.85–0.90 | < 0.85 |
| F0 RMSE (Hz) | ≤ 10 | 10–20 | > 20 |

**為什麼必要**：vocoder 是凍結 ckpt；若它對中文歌聲重建已經斷裂（mel→wav 失真），後續 M 任何改進都看不到。

**目前實測**（origin GT 22050 native，pyworld + interp）：
- SSIM = 0.94, F0 RMSE = 5.8 Hz → **PASS** ✅

#### Gate ②：Audio quality probe（Risk 2 L4）

**腳本**：`scripts/audio_quality_probe.py`。對每個 dataset 抽 N 首歌算 raw wav 的音質統計，計算跨 dataset 的 JSD。

| Metric | 含義 | PASS 閾值 |
|---|---|---|
| SFM | spectral flatness（噪音含量） | JSD < 0.10 |
| Reverb | direct-to-reverberant ratio 估計 | JSD < 0.10 |
| HF-ratio | high-frequency energy ratio | JSD < 0.10 |
| SNR | ITU-R P.56 SNR 估計 | JSD < 0.10 |

**為什麼必要**：M4Singer (錄音室) vs VocalVerse (user-generated) 的音質域差異，若不通過 dereverb 拉齊，會讓 D_mel/D_z 學到「環境音=amateur 簽名」的捷徑（Risk 2 主防線）。

**不過時的處理**（依序）：
1. 確認 binarize 用預設 dereverb=True
2. 在外部做 SNR 篩選把 VocalVerse 過糟的樣本拿掉
3. 嚴重不過則考慮換 dataset 或縮小選曲

#### Gate ③：資料策展 JSD（register / phoneme）

**腳本**：`nsvb/utils/jsd_check.py`。

| 指標 | 閾值 | 不過怎麼辦 |
|---|---|---|
| JSD(VocalVerse phoneme dist, M4Singer phoneme dist) | < 0.05 | 重採樣 phoneme distribution |
| JSD(VocalVerse register dist, M4Singer register dist) | < 0.05 | 重採樣 register distribution |

**為什麼必要**：D_z 用 phoneme + register 當條件；若兩 dataset 的 phoneme 頻率分布有大差別（例如 M4Singers 比例壓倒性高男聲），D_z 看到「phoneme 出現比例」就能猜業餘/職業，把 M 引導到亂改。

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

### 2.7 ckpt 保存格式 / 中段續訓

每 `save_interval` 步（預設 5000）寫兩份：
- `checkpoints/stage1/stage1_step{N}.pt`（永久保留，給歷史對照）
- `checkpoints/stage1/stage1_latest.pt`（每次覆蓋，方便 resume）

訓練結束時多寫 `stage1_final.pt`。

`checkpoints/stage1/stage1_*.pt` 內容：
```python
{
    'step': int,
    'epoch': int,
    'model': SVBVAEZh.state_dict(),
    'd_mel': D_mel.state_dict(),       # 若 use_d_mel=True
    'opt_g': optim state dict,
    'opt_d': optim state dict,           # 若 use_d_mel=True
    'config': Stage1Config dict,
}
```

**中段續訓**：[Stage1Trainer.load_ckpt](../nsvb/task/stage1.py) 恢復 `model + d_mel + 兩 optimizer + step + epoch`。CLI 用 `--resume`：

```bash
# 用 'latest' 簡寫自動找 {ckpt-dir}/stage1_latest.pt
python -m nsvb.task.stage1 ... --resume latest

# 或指定特定 step
python -m nsvb.task.stage1 ... --resume checkpoints/stage1/stage1_step50000.pt
```

> 為什麼要恢復 optimizer state：只恢復 model 會讓 Adam 動量重設，原 lr 下 loss 會明顯反彈幾百步。step 必須恢復以維持 KL warmup schedule、save/log interval 對齊。

### 2.8 訓中進度顯示（terminal）

[Stage1Trainer.fit](../nsvb/task/stage1.py) 用 tqdm（ASCII 模式，避免 Windows / 非 utf-8 ssh 亂碼）+ pbar.write 印 log：

```
stage1: 12%|############| 9500/80000 [3:42:11<27:30:18, 0.71it/s, l_total=0.234, kl=0.046]
[step   9500 ep14 0.71it/s] l_total=0.2336 l_l1=0.1124 l_l2=0.0143 kl=0.0462 kl_beta=0.0095 ...
[stage1] ckpt saved: checkpoints/stage1/stage1_step10000.pt
[stage1] ckpt saved: checkpoints/stage1/stage1_latest.pt
```

> 長時間訓練建議用 tmux/screen 避免 ssh 斷線中斷訓練；詳見 [deployment_linux.md §6.2](deployment_linux.md)。

### 2.9 執行命令

```bash
PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 \
    --batch-size 16 \
    --max-steps 80000 \
    --num-workers 4 \
    --init-from-nsvb /path/to/1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1
```

> `--init-from-nsvb` 啟用後 `init_lr_scale=0.5` 把 lr_g 降至 1e-4 fine-tune；80k 步即可達到從零訓 200k 的品質（省 ~5x 時間）。

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

#### D_mel（**重用 Stage 1**, real 預設只看 pro）

從 Stage 1 ckpt 載入 D_mel；real 端預設餵 pro mel，fake 端餵 `θ(M(z_a))` decode 出的 mel；低 lr 1e-5 微調。

**Risk 2 fallback（CLI flag `--dmel-mix-amateur-real`）**：
- 預設 `False`：D_mel real **只看 pro**（[rebuild_checklist §C](../rebuild_checklist.md) 設計，把 D_mel 升級為「pro 自然度」推力）
- 設 `True`：D_mel real 改餵 pro+amateur 混合（每 step `cat([pro, amateur], dim=0)`），退化成 ver1 的「自然度判別器」，犧牲 mel 層 pro-direction 訊號換取「絕不鼓勵去殘響」的安全
- 何時啟用：訓中 `monitor_audio_quality` 連續兩次顯示 `unvoiced_concentration > 0.65` 時 resume + 啟此 flag 救火

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

#### 3.6.1 Loss / 統計（每 log_interval=50 步印）

| 指標 | 期望範圍 | 解讀 |
|---|---|---|
| `l_nce` | 收斂 ~ 1.0–3.0 | 過低（< 0.5）：M ≈ identity 沒在動；過高（> 5）：M 把內容打亂了 |
| `l_adv_z` | warmup 後 ~ -0.5 to 0 | 持續 > 0：D_z 一直贏，M 學不動；持續 < -1：M 騙過 D_z（要看 D_z accuracy） |
| `l_adv_mel` | ~ -0.3 to 0 | 同上邏輯 |
| `l_id_pro` | 20% 步抽中時 ~ 0.05–0.2 | M 在 pro 端沒亂改 |
| **`d_z`** | **~ 0.5–1.5** | < 0.3：D_z 太弱（real fake 都信，M 不需學）；> 2：D_z 太強 M 學不動 |
| `d_mel` | ~ 0.5–1.5 | 同上 |
| **`delta_over_z`** ⭐ | **0.03–0.20** | 這是「**M 漂移程度**」核心指標：< 0.03 → M 太保守；> 0.30 → M 過度激進可能破壞內容 |

#### 3.6.2 Risk 2 L5：訓中音質監控

[Stage2Trainer.monitor_audio_quality](../nsvb/task/stage2.py)（每 `audio_quality_monitor_interval`=5000 步觸發一次）：
1. 抽 `audio_quality_monitor_n_samples`（=4）個 amateur 樣本
2. 算 `Δ_mel = mel_modified − mel_baseline`（modified=過 M；baseline=不過 M）
3. 比較 voiced 段 vs unvoiced 段 Δ 能量

判讀（[risk.md](../risk.md) Risk 2 補強 4）：

| `unvoiced_concentration` | 含義 | 行動 |
|---|---|---|
| < 0.55 | M 修飾集中在 voiced 段 — 真技術修正 | 繼續訓 |
| 0.55–0.65 | marginal | 留意，再觀察 5000 步 |
| > 0.65 | **Risk 2 警訊** — M 在去殘響/降噪 | 停下來，resume + `--dmel-mix-amateur-real` 救火 |

每次抽樣會把 mel spectrogram 對比存到 `checkpoints/stage2/audio_monitor/step{N}_sample0.npz`（含 `mel_gt / mel_baseline / mel_modified / delta_mel / unvoiced_concentration`），可用 matplotlib 視覺化檢查。

#### 3.6.3 Auto-warnings（一次性 print）

| 條件 | 訊息 | 處理 |
|---|---|---|
| step ≥ `delta_health_check_step`(30000) 且 `‖Δ‖/‖z‖` 移動平均 < `delta_health_check_threshold`(0.03) 且 `m_kernel_size==1` | "M 可能太保守，無法生成顫音/滑音；建議 `--m-kernel-size 3` 重訓" | 停下來換 kernel=3 重訓（M 結構改了，需從 stage1 ckpt 重啟）|
| `T_z < 4`（PatchNCE）| RuntimeWarning：建議 max_frames ≥ 64 | 調大 `--max-frames` |

#### 3.6.4 D_z accuracy（外掛 monitor，可選）

每幾百步抽一次 batch 計算 `D_z(real) > 0` 與 `D_z(fake) < 0` 的比例：
- 0.55–0.75 = 健康（D_z 略勝但 M 仍在學）
- > 0.85 = D_z 過強（增 lr_M 或減 lr_dz）
- < 0.55 = D_z 過弱（多訓 D_z 幾步）

### 3.7 ckpt 保存格式 / 中段續訓

每 `save_interval` 步（預設 5000）寫兩份：`stage2_step{N}.pt`、`stage2_latest.pt`；訓練結束時多寫 `stage2_final.pt`。

`checkpoints/stage2/stage2_*.pt` 內容：
```python
{
    'step': int,
    'M': ResidualM.state_dict(),
    'D_z': DiscriminatorZ.state_dict(),
    'D_mel': D_mel.state_dict(),
    'patchnce': PatchNCELoss.state_dict(),       # projection head weights
    'opt_m / opt_dz / opt_dmel': optim states,
    'config': Stage2Config dict,
    'stage1_ckpt': str  (Stage 1 ckpt 路徑記錄)
}
```

> 注意：CVAE φ/θ 不存（因為凍結，從 `stage1_ckpt` 路徑取得）；每次啟動 trainer 都會從該路徑重新載入並凍結。

**中段續訓**：[Stage2Trainer.load_ckpt](../nsvb/task/stage2.py) 恢復 `M / D_z / D_mel / PatchNCE proj head + 三個 optimizer + step`。CLI 用 `--resume`：

```bash
python -m nsvb.task.stage2 ... --resume latest         # 自動找 latest
python -m nsvb.task.stage2 ... --resume checkpoints/stage2/stage2_step60000.pt
```

> PatchNCE 也存 ckpt 是因為它有 learnable projection head（2-layer MLP）；不恢復會讓 contrastive loss 一夕重置，M 收到大跳變的梯度信號。

### 3.8 執行命令

```bash
# 預設配方（kernel=1, D_mel pro-only）
PYTHONPATH=. python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --amateur-dataset vocalverse \
    --pro-dataset m4singer \
    --ppg-dim 1280 \
    --phoneme-vocab-size 200 \
    --batch-size 16 \
    --max-steps 120000 \
    --num-workers 4 \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --ckpt-dir checkpoints/stage2

# 保守配方（kernel=3 from start，避免 30k 步後才發現 M 太保守）
PYTHONPATH=. python -m nsvb.task.stage2 ... --m-kernel-size 3

# Risk 2 救火：unvoiced_concentration > 0.65 連兩次後 resume
PYTHONPATH=. python -m nsvb.task.stage2 ... \
    --resume latest \
    --dmel-mix-amateur-real
```

---

## 4. Phase 3 — 推理（已實作）

**模組**：[`nsvb/inference/`](../nsvb/inference/)
**CLI**：[`scripts/infer.py`](../scripts/infer.py)

### 4.1 子模組組織

| 檔案 | 職責 |
|---|---|
| [`feature_pipeline.py`](../nsvb/inference/feature_pipeline.py) | 單檔特徵抽取（與 binarize 對齊：dereverb + loudness + mel + F0 + PPG + spk_emb），回傳 `InferenceFeatures` dataclass |
| [`model_loader.py`](../nsvb/inference/model_loader.py) | 從 Stage 2 ckpt（含對 Stage 1 ckpt 的引用）+ vocoder ckpt 載入 frozen `InferenceModels`（cvae / M / vocoder） |
| [`dtw_warp.py`](../nsvb/inference/dtw_warp.py) | Mode B 的 mel-rate DTW（librosa.sequence.dtw）+ `torch.gather` warp 工具 |
| [`pipeline.py`](../nsvb/inference/pipeline.py) | `run_mode_a` / `run_mode_b` 主入口；自動 pad 到 `LATENT_DOWN_FACTOR=4` 倍數，輸出 trim 回原長度 |
| [`__init__.py`](../nsvb/inference/__init__.py) | 統一 re-export：`run_mode_a / run_mode_b / InferenceFeatureExtractor / load_inference_models` |

### 4.2 Mode A — 純自動推理（預設）

```
x_a (wav)
  → feature_pipeline.extract  →  mel_a, f0_a, ppg_a, spk_emb_a
  → φ(mel_a, ppg_a, f0_a, spk_emb_a)  →  m_q (posterior mean, deterministic, T_z_a)
  → M(m_q)                              →  z_a' (T_z_a)
  → θ(z_a', ppg_a, f0_a, spk_emb_a)    →  mel_recon (T_mel_a)
  → vocoder(mel_recon, f0_a)            →  wav (T_a samples)
```

**重點**：
- F0 來源 = x_a 自身 → 輸出長度 = T_a → 可直接配原伴奏
- z 用 `m_q` 而非 sampled（與 stage2 訓練端 `_encode_and_downsample` 一致；確定性 reproducibility）
- 餵 vocoder 前對 unvoiced=0 的 frame 在 log-Hz space 線性內插（避免 SineGen 邊界電音）

### 4.3 Mode B — 完全參考（節奏+音高拉到 pro 模板）

```
x_a (wav)         x_p_ref (wav, 同首歌專業參考)
  ↓                  ↓
features_a       features_p
  ↓
φ(mel_a, ...)        ↓
  ↓                  ↓
M(m_q)               ↓
  ↓                  ↓
z_a' (T_z_a)         ↓
                  DTW(mel_a, mel_p_ref) → path_to_a (T_mel_p)
  ↓
warp_latent(z_a', path)  →  z_a'_warped (T_z_p)
  ↓
θ(z_a'_warped, ppg_p_ref, f0_p_ref, spk_emb_a)  →  mel(T_mel_p)
  ↓
vocoder(mel, f0_p_ref)  →  wav (T_p samples)
```

**條件選擇**（rebuild_checklist §H 定案）：
- F0：`f0_p_ref`（pro 模板的音高）
- PPG：`ppg_p_ref`（同詞同旋律，pro 端在 T_p 軸上自然對齊；warp amateur PPG 反而會引入咬字模糊）
- spk_emb：`spk_emb_a`（**保留業餘音色**，Risk 4 主防護；不換成 pro）

**輸出長度 = T_p**（不等於 T_a）→ 必須配 pro reference 的伴奏，不能配原伴奏。

### 4.4 Padding 處理（FVAE stride 對齊）

FVAE encoder/decoder 的 stride conv 要求 `T_mel % LATENT_DOWN_FACTOR == 0`。訓練時 `max_frames=600` 剛好是 4 倍數繞過；推理時 user 輸入長度任意，必須在 pipeline 內補齊：

```python
# pipeline._pad_batch_to_multiple(batch, factor=4)
T_mel = batch["mel"].shape[1]
if T_mel % factor != 0:
    pad_t = factor - (T_mel % factor)
    # mel/ppg/f0 都 pad 0；mel_mask 也 pad 0（標記 padding 段）
```

最後在 wav/mel/f0 同步 trim 回 `T_mel_orig × HOP_SIZE` samples（vocoder 在 padding 段是垃圾輸出，必須切掉）。

### 4.5 推理 CLI

```bash
# Mode A（預設）：業餘 → 修飾後 wav，配原伴奏
python -m scripts.infer \
    --stage2-ckpt checkpoints/stage2/stage2_latest.pt \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --input-a path/to/amateur.wav \
    --output outputs/mode_a_result.wav

# Mode B：加 --pro-ref 自動切換
python -m scripts.infer ... \
    --input-a path/to/amateur.wav \
    --pro-ref path/to/pro_reference.wav \
    --output outputs/mode_b_result.wav

# 跨機器：Stage 2 ckpt 內紀錄的 stage1_ckpt 路徑可能找不到，明確覆寫
python -m scripts.infer ... \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt
```

主要旗標：
- `--no-dereverb` / `--no-loudness`：關閉前處理（若輸入已是乾淨 studio 才用）
- `--no-f0-interp`：debug 用，會產生電音
- `--dtw-metric euclidean|cosine`：Mode B DTW 距離
- `--save-mel`：把 decoder 出的 mel 存成 .npy 供視覺化

### 4.6 推理 model 載入

```python
from nsvb.inference import load_inference_models, run_mode_a

models = load_inference_models(
    stage2_ckpt="checkpoints/stage2/stage2_latest.pt",   # 含 M weights + 對 stage1 路徑引用
    vocoder_ckpt="checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt",
    stage1_ckpt=None,    # None = 用 stage2 ckpt 內紀錄的路徑；跨機器時覆寫
    device="cuda",
)
```

`InferenceModels` dataclass: `cvae` (frozen) + `M` (frozen) + `vocoder` callable + `device` + `config` dict。

---

## 5. Tensor shape 速查表

### 5.1 Frame rate 三段對照

| 域 | fps | 換算 |
|---|---:|---|
| audio (wav) | 22050 | 1 sample = 1/22050 sec |
| mel | 172.27 | hop=128, 1 frame = 5.8 ms |
| latent z | 43.07 | strides=4, 1 frame = 23.2 ms |

### 5.2 訓練時 batch 內 tensor 形狀（max_frames=600）

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

## 6. 重要 hyperparameter 速查

### 6.1 Audio config（不能動，動了要重 binarize 全部）
```
SR=22050, hop=128, fft=512, win=512, n_mels=80,
mel_fmin=50, mel_fmax=11025,                    # mel 對齊 NSVB inference 實際值
F0_fmin=50, F0_fmax=1400,                       # ⭐ 1400 覆蓋到 F6=1397 Hz
eps=1e-10, loudness_target=-22 LUFS
```

### 6.2 Stage 1（CVAE）
```
gin_channels=256 (= ppg_proj 128 + pitch_emb 32 + spk_proj 96)
hidden=192, latent=128, strides=[4], enc_layers=8, dec_layers=4, kernel=5
loss weights: l1=1 l2=1 kl_target=0.01(warmup 5k) adv_mel=0.1
optim: lr_g=2e-4 (init from NSVB → ×0.5 = 1e-4), lr_d=2e-4
batch=16, max_frames=600, max_steps=80k (with NSVB init) ~ 200k (scratch)
```

### 6.3 Stage 2（Mapping）
```
M:    kernel=1 (or 3), hidden=256, num_layers=4, init_delta_scale=1e-2
D_z:  hidden=256, num_layers=4, kernel=5, vocab=200, embed=32
PatchNCE: proj=64, num_patches=128, temp=0.07
loss weights: NCE=1.0, adv_z=1.0(after warmup 5k), adv_mel=0.2, id_pro=0.1@0.2_prob
optim TTUR: lr_M=1e-4, lr_Dz=4e-4, lr_Dmel=1e-5  (β=(0.5, 0.999))
batch=16, max_frames=600, max_steps=120k

D_mel real source: pro-only (default) | pro+amateur if --dmel-mix-amateur-real
auto-warning: kernel=1 + Δ/z<0.03 @ step 30000
audio quality monitor every 5000 steps (Risk 2 L5)
```

---

## 7. 已知 limitation 與 TODO

| 項目 | 狀態 | 備註 |
|---|---|---|
| Mode A / Mode B 推理 | ✅ 完成 | [`nsvb/inference/`](../nsvb/inference/) + [`scripts/infer.py`](../scripts/infer.py) |
| `--resume` 中段續訓 | ✅ 完成 | Stage 1 / 2 都有 |
| tqdm 訓中進度條 | ✅ 完成 | ASCII 模式 + pbar.write 印 log |
| Risk 2 L5 訓中音質監控 | ✅ 完成 | `Stage2Trainer.monitor_audio_quality` |
| Risk 2 D_mel fallback | ✅ 完成 | `--dmel-mix-amateur-real` |
| L_PPG | TODO（λ=0 預設） | 後續加 fast mel→PPG predictor 模組後啟用 |
| F0 unvoiced 內插（餵 vocoder） | ✅ 完成 | 推理 pipeline 預設啟用 (`--no-f0-interp` 可關) |
| Multi-GPU DDP | TODO | 視訓練時間需求加 |
| AMP / torch.compile | TODO | 訓練機 benchmark 後決定是否開 |

---

## 8. Phase 0 / 1 / 2 / 3 整體命令範例

```bash
# === Phase 0：資料二進位化 ===
# Gate ①：vocoder identity test（先確認 vocoder 不是 bottleneck）
PYTHONPATH=. python -m scripts.vocoder_identity_test \
    --vocoder-ckpt /path/to/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 --save-wavs \
    --out-dir outputs/phase0_vocoder

# Gate ②：audio quality probe（Risk 2 L4 — 兩 dataset 音質統計差距）
PYTHONPATH=. python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --out-dir outputs/phase0_audio_quality

# Binarize（Risk 2 L1+L2 — dereverb + 響度正規化都自動跑）
PYTHONPATH=. python -m nsvb.data.binarizer --dataset m4singer
PYTHONPATH=. python -m nsvb.data.binarizer --dataset vocalverse

# Phase 0 第二階段：k-means 分群 phoneme_id
PYTHONPATH=. python -m nsvb.data.cluster_ppg \
    --binarized-root data/binarized --k 200 --stage all

# Gate ③：JSD（register / phoneme）檢查
PYTHONPATH=. python -m nsvb.utils.jsd_check  # 檢查 phoneme/register JSD < 0.05

# === Phase 1：Stage 1 CVAE 預訓練 ===
PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 --batch-size 16 --max-steps 80000 --num-workers 4 \
    --init-from-nsvb /path/to/1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1
# 中段續訓：加 --resume latest

# === Phase 2：Stage 2 Mapping 訓練 ===
PYTHONPATH=. python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --amateur-dataset vocalverse --pro-dataset m4singer \
    --ppg-dim 1280 --phoneme-vocab-size 200 \
    --batch-size 16 --max-steps 120000 --num-workers 4 \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --ckpt-dir checkpoints/stage2
# 中段續訓：加 --resume latest
# Risk 2 救火：加 --dmel-mix-amateur-real

# === Phase 3：推理 ===
# Mode A（預設）：業餘 → 修飾後 wav，配原伴奏
PYTHONPATH=. python -m scripts.infer \
    --stage2-ckpt checkpoints/stage2/stage2_latest.pt \
    --vocoder-ckpt /path/to/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --input-a path/to/amateur.wav \
    --output outputs/mode_a_result.wav

# Mode B：加 --pro-ref（同首歌專業參考），輸出長度 = T_p
PYTHONPATH=. python -m scripts.infer ... \
    --input-a path/to/amateur.wav \
    --pro-ref path/to/pro_reference.wav \
    --output outputs/mode_b_result.wav
```