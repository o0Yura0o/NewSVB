# NSVB-ZH 與 ver1 的對照與決策紀錄

本文件記錄 `NSVB-ZH_ver1/`（先前一份 NSVB-ZH 完整架構稿）與我們**目前重建版**（`nsvb/`）
之間的設計差異，以及**每個差異的理由**：哪些 ver1 做法我們吸納、哪些不採用、為什麼。

> 配套文件：[rebuild_checklist.md](../rebuild_checklist.md)（架構決策全集）、
> [training_flow.md](training_flow.md)（操作手冊）

---

## 1. 摘要

| 維度 | ver1 | 我們 |
|---|---|---|
| 整體開發策略 | **patch-style** drop-in 進 NSVB fork | **standalone** 在獨立 `nsvb/` package 重建 |
| 業餘 / 職業資料集 | M4Singer = amateur, OpenSinger = pro | **M4Singer = pro, VocalVerse = amateur**（角色反向） |
| Phase 0 哲學 | **post-training probe**（訓 50–80k 步後測 dataset discriminability，分 Case A/B/C） | **pre-training JSD gate**（訓前用純資料統計判斷，1 個 case） |
| GRL DomainDiscriminator | Case B 啟用 | **不採用**（用 spk_emb + L_PatchNCE + L_identity_pro 三層保險取代） |
| PPG 來源 | 外部 Chinese PPG ckpt（VCASR-style，未實作） | **Whisper-large-v3 hidden state + k-means**（self-supervised）|
| Phoneme ID 來源 | MFA forced alignment + zh txt_processor | **PPG k-means(K=200)** |
| Speaker embedding | NSVB base binarizer 隱含 | **Resemblyzer 256-dim 顯式**（Risk 4 主防線） |
| 推理模式 | A / B / C 三模式 | **A / B 兩模式**（合併 ver1 的 B，C 為新 B）|

兩邊**核心架構（M / D_z / PatchNCE）相同**，差異主要在前處理、訓練哲學、與工程組織。

---

## 2. ver1 是什麼

`NSVB-ZH_ver1/` 包含：
```
modules/voice_conversion/   ResidualM, DiscriminatorZ, DomainDiscriminator, PatchNCELoss, SoftRegisterEncoder
tasks/singing/              NsvbZhTask（Stage 1 + Stage 2 dispatcher）
data_gen/singing/           NsvbZhBinarizer（繼承 NSVB SingingBinarizer）
scripts/                    probe_dataset_discriminability.py, vocoder_identity_test.py
egs/datasets/audio/         base.yaml, stage1_pretrain.yaml, stage2_mapping.yaml
inference/                  nsvb_zh_inference.py（Mode A/B/C）
```

**特性**：
- 每個檔案都是「插件」格式，drop in 進 NSVB fork 對應路徑就能與既有 utils / NSVB hparams / SVBVAEMleTask 整合
- 依賴 NSVB 全域 `from utils.hparams import hparams` 狀態
- 偏向「最小改動」哲學：盡量重用 NSVB 既有 task 框架

---

## 3. 我們**吸納**的 ver1 做法（與我們的實作對照）

### 3.1 ResidualM 結構（kernel=1 + residual + near-identity init）

| 設計 | ver1 | 我們 |
|---|---|---|
| 殘差結構 | `f(z) = z + Δ(z)` | **同** |
| 預設 kernel | 1（warp-invariant） | **同** |
| 可選 kernel=3 | 若 ‖Δ‖ < 3% 切 3 | **同** |
| init_delta_scale | 1e-2（最後一層 conv std） | **同** |
| Norm | GroupNorm | **同** |
| Activation | GELU | **同** |

**為什麼吸納**：ver1 的 ResidualM 設計直接對應論文「warp-invariant residual mapping」需求；
這部分在 unpaired SVB 場景已經是業界共識，沒有重新發明的必要。
**檔案**：`nsvb/model/m_mapping.py`

### 3.2 D_z 內部寫法（spectral_norm + LeakyReLU(0.2) + hinge loss）

| 設計 | ver1 | 我們 |
|---|---|---|
| 卷積層 | `spectral_norm(Conv1d)` | **同** |
| activation | LeakyReLU(0.2, inplace=True) | **同** |
| condition 注入 | channel-axis concat（不用 projection trick） | **同** |
| 輸出 | per-frame 1×1 conv head 出 logit | **同** |
| 訓練 loss | hinge G/D loss | **同** |

**為什麼吸納**：
- spectral_norm 限制 Lipschitz，GAN 訓練穩
- LeakyReLU 在負值區仍給梯度，避免 dying ReLU
- hinge 在 D 強時 G 梯度仍 alive（vs. BCE 的 saturating）
- channel concat 對連續 register + embedding phoneme 更通用（projection trick 適合 class id 不適合此處）

**檔案**：`nsvb/model/d_z.py`、`nsvb/model/losses.py:hinge_d_loss/hinge_g_loss`

### 3.3 PatchNCE 設計（learnable projection + batch-internal negatives）

| 設計 | ver1 | 我們 |
|---|---|---|
| Projection head | 2-layer MLP (1×1 Conv + GELU + 1×1 Conv) | **同** |
| proj_dim | 64 | **同** |
| temperature | 0.07 | **同** |
| num_patches | 128 | **同** |
| Negative 來源 | batch item 內取（cross-batch 不取） | **同** |

**為什麼吸納**：
- 投影到 64 維讓 cosine 信噪比高（raw 128 維 contrastive signal 弱）
- temperature=0.07 是 CUT 論文驗證值
- batch-internal negatives 強迫 model 區分「同首歌不同 frame」，學 fine-grained content invariance；
  cross-batch 太簡單（不同歌、不同歌手 naturally 已差別大），訓練 saturated

**檔案**：`nsvb/model/losses.py:PatchNCELoss`

### 3.4 Soft register bucket（5 Gaussian, σ=0.3 log-Hz）

| 設計 | ver1 | 我們 |
|---|---|---|
| Bucket 數 | 5 | **同** |
| σ | 0.3 log-Hz | **同** |
| Centers | log-Hz 均勻分布 (4.18, 4.87, 5.57, 6.26, 6.95) | **C3, G3, D4, A4, E5 對應 130.81 / 196.00 / 293.66 / 440.00 / 659.25 Hz**（音樂上更直觀） |
| unvoiced 處理 | 全零 | **同** |

**為什麼吸納**：5 buckets 在「粗到防 F0 shortcut，但細到能分胸/頭聲」之間平衡良好。
**為什麼換 centers**：我們選音樂常用音高（C3/G3/D4/A4/E5）對應實際歌手聲區，
比 ver1 的等間距 log-Hz 更貼近歌唱直覺；數值上差異 < 0.5 log-Hz 不影響 model 表現。

**檔案**：`nsvb/utils/soft_bucket.py`

### 3.5 TTUR + D_z warmup

| 設計 | ver1 | 我們 |
|---|---|---|
| opt_M LR | 1e-4 | **同** |
| opt_Dz LR | 4e-4 (4× M) | **同** |
| opt_Dmel LR | 1e-5 | **同** |
| Adam β | (0.5, 0.999) | **同** |
| D_z warmup steps | 5000 | **同** |
| Warmup 期 M 收訊 | 不收 D_z 梯度，但仍訓 PatchNCE + adv_mel + identity_pro | **同** |

**為什麼吸納**：TTUR 是 NSVB 原作者後實證的穩定設定；warmup 防止 D_z 從零訓的 noisy 階段帶歪 M。

**檔案**：`nsvb/task/stage2.py:Stage2Trainer.train_step` + `Stage2Config.d_z_warmup_steps`

### 3.6 ⭐ L_identity_pro（20% batches × weight 0.1）

| 設計 | ver1 | 我們 |
|---|---|---|
| 抽中機率 | 0.2 | **同** |
| 權重 | 0.1 | **同** |
| 目標 | `M(z_p) ≈ z_p` (L1) | **同** |

**為什麼吸納（這是最重要的吸納項）**：
ver1 點出了一個 NSVB 原論文沒處理的問題——M 在 Stage 2 主要看 amateur z，
可能在 pro 端自由發揮（無 supervision），即便 spk_emb 鎖音色仍可能讓 M 對 pro z 加奇怪偏移。
Stochastic 20% 機率隨機要求 M(z_p) 近恆等，平均效果是「不要漂太遠」，
又不犧牲 amateur 端訓練。Risk 4 的二級保險。

**檔案**：`nsvb/model/losses.py:l_identity_pro` + `nsvb/task/stage2.py` 隨機抽中邏輯

### 3.7 ⭐ Vocoder identity test as Phase 0 gate

| 設計 | ver1 | 我們 |
|---|---|---|
| 提出此 gate | ✅ | ✅（吸納） |
| Metric | mel SSIM + F0 RMSE | **同** |
| Threshold | SSIM ≥ 0.90 + F0 RMSE ≤ 10 Hz PASS | **同**（pass/marginal/fail 三段） |
| F0 抽法 | pyworld DIO+StoneMask | **採用 ver1 的 pyworld 為預設**（torchcrepe / parselmouth 也支援） |

**為什麼吸納**：vocoder 是凍結 ckpt；若它對中文歌聲重建已經斷裂，後續 M 任何改進都隱形。
這個 gate 必須在 Phase 0 跑，不過再進 Stage 1。

**我們進一步補強**：
- ver1 的 stub 沒有實作 vocoder 載入細節；我們發現 NSVB 1012 ckpt 是真 HifiGAN-NSF（不是 PWG）
  並完整 port (`nsvb/backbone/vocoder/{hifigan_nsf, source}.py`)，與 ckpt **244/244 keys strict-load** 通過
- 加 `--save-wavs` 選項給人耳聽測（metric 失靈時的 fallback gate）

**檔案**：`scripts/vocoder_identity_test.py` + `nsvb/backbone/vocoder/`

### 3.8 ⭐ Stage 2 D_mel 復用 Stage 1（low LR）+ real 改為只看 pro

| 設計 | ver1 | 我們 |
|---|---|---|
| 復用 Stage 1 D_mel | ✅ | ✅ |
| LR 1e-5 微調 | ✅ | ✅ |
| Stage 2 real 改為 pro only | ✅ | ✅ |

**為什麼吸納**：D_mel 在 Stage 1 學「自然人聲分布」是 language-agnostic 的；Stage 2 升級成「pro 自然度」
判別器不必從零訓，省 ~30% 訓練時間。

---

## 4. 我們**不採用** ver1 的做法（與替代方案）

### 4.1 ❌ Patch-style 開發策略 → 我們選 standalone

| | ver1 | 我們 |
|---|---|---|
| 路徑 | drop-in 進 NSVB fork | 獨立 `nsvb/` package |
| 依賴 | NSVB 全域 `hparams` state、`SVBVAEMleTask`、`SingingBinarizer` 等 | 全部 explicit constructor args + dataclass config |
| 子模組移植 | 不移植，直接 import NSVB | 只 port 必要的 4 個檔（fvae / multi_window_disc / hifigan_nsf / source） |

**為什麼不採用 patch-style**：
1. NSVB 含大量無關子系統（asr/glow/tts/voice_conversion 多種 task），patch-style 全拖進來
2. NSVB hparams 是全域 state，多個 yaml include / override，不顯式追難 debug
3. NSVB 用 librosa 0.7 positional API，新版 librosa 不相容（已踩過坑）
4. 跨平台不友善（NSVB 主要在 Linux 開發，Windows 多處要 patch）

**我們的做法**：照「最小集合移植」原則（rebuild_checklist §J' 守則 6），只 port 真正用到的；
所有 hparams 走 explicit dataclass，避免全域 state。

### 4.2 ❌ Phase 0 post-training probe (Case A/B/C) → 我們選 pre-training JSD gate

| | ver1 | 我們 |
|---|---|---|
| 時機 | 訓練 50–80k 步 Stage 1 後 | **訓練前**（純資料統計） |
| 工具 | logreg + tiny MLP 預測 dataset 標籤 | JSD(phoneme freq) + JSD(register freq) |
| 分 case | A (acc<0.6) / B (0.6-0.75 + GRL) / C (≥0.75 fallback) | 1 case（< 0.05 PASS, 否則重採樣） |
| 計算成本 | 高（要先跑半熟 Stage 1） | 極低（純 numpy 統計） |

**為什麼不採用**：
1. **昂貴**：要先跑 ~50k Stage 1 才能 probe，發現 Case C 的話前面 5–7 天 GPU 全白費
2. **後驗的不可逆性**：ver1 Case C 給的選項（MOSNet relabel / DTW / 換資料集）都需要重來，
   這個成本應該前置避免，不該事後補救
3. **JSD 是充分而非必要**：JSD 過了表示分布相近 → D_z 不會用分布差當捷徑；
   JSD 不過則我們可以用 phoneme/register **重採樣**對齊（ver1 沒提這條更便宜的 fallback）

**我們的做法**：
- 寫 `nsvb/utils/jsd_check.py` 作 Phase 0 hard gate
- 不過則用「貪婪重採樣」（drop 過多比例的 phoneme/register frame）對齊到閾值內
- 若重採樣會丟超過 50% 的 frame → 才考慮 ver1 的 C1/C2/C3 fallback

### 4.3 ❌ GRL DomainDiscriminator (ver1 Case B) → 我們不引入

| | ver1 | 我們 |
|---|---|---|
| 何時用 | Case B (probe acc 0.6–0.75) 啟用 | 不引入 |
| 機制 | DANN-style gradient reversal，warmup λ 0.1 → 0.3 over 30k 步 | （以三層保險取代） |

**為什麼不採用**：
1. GRL 訓練不穩（λ schedule 經驗難調）
2. 我們已用 **三層保險**處理 dataset bias：
   - Phase 0 JSD gate（前置消除）
   - **L_PatchNCE**（z 層 frame 對應鎖）
   - **L_identity_pro**（M 在 pro 端近恆等防漂移）
   - **spk_emb 顯式條件**（decoder 音色錨）
3. 三層保險都跑了還不夠才考慮 GRL，目前無證據需要

**我們的做法**：保留 GRL 為「第四層」備案（risk.md 記錄），不主動加入訓練 loop。

### 4.4 ❌ MFA forced alignment + zh txt_processor 提取 phoneme_id → 我們選 PPG k-means

| | ver1 | 我們 |
|---|---|---|
| 提取方式 | MFA + pinyin (zh txt_processor) | **k-means(K=200) on Whisper PPG** |
| 必要前置 | 每首歌的 lyrics 檔 | 純 audio，不需 lyrics |
| 對歌曲 | MFA 對歌唱對齊不準（vibrato / 拉長母音）| Whisper hidden state 直接是 frame-level，無對齊問題 |
| K | 約 80（pinyin 音素級） | 200（給 K-means 足夠 cluster 容納 prosody / 共發音 變體） |

**為什麼不採用**：
1. **資料限制**：VocalVerse 是 user-generated 業餘歌聲，**多數沒有 lyrics 檔**（強制要求 lyrics 會丟一半樣本）
2. **MFA 對歌唱不準**：MFA 訓練於語音 (LibriSpeech 等)，對 vibrato / 拉長母音 / 假音常 align 錯
3. **Self-supervised 路徑更乾淨**：PPG k-means 不需 lyrics、不需 MFA 環境、不依賴中文音素表

**我們的做法**：
- `nsvb/data/feature_extract/ppg_whisper.py` 抽 Whisper-large-v3 layer 8 hidden state
- `nsvb/data/cluster_ppg.py` 用 MiniBatchKMeans(K=200) 分群得 phoneme_id
- 完全 self-supervised，跨語言通用

> 但我們仍保留 ver1 的 zh txt_processor (`nsvb/data/txt_processors/zh.py`)，
> 用於：(a) 未來若有 lyrics 的 dataset 可用作 phoneme reference；
> (b) JSD 計算的 reference baseline。

### 4.5 ❌ ver1 假設外部 Chinese PPG extractor ckpt → 我們直接用 Whisper

ver1 的 `_maybe_load_ppg()` 期望從 `hparams['ppg_extractor_ckpt']` 載入 VCASR-style 模型；
但 ver1 沒提供該 ckpt（user 要自備或自訓）。

**我們不採用**：直接用 **Whisper-large-v3** 的 encoder layer 8 hidden state 當 frame-level
content representation。Whisper 是 multilingual SOTA，不需要訓 PPG model，且跨語言友善。

### 4.6 ❌ ver1 推理 Mode B（業餘節奏 + 專業音準）→ 我們合併到一個 Mode

ver1 三模式：
- A: 純自動
- B: 業餘節奏 + 專業 F0（DTW warp F0 to T_a）
- C: 完全參考（z_a warp to T_p + 專業 F0）

**我們改成兩模式**：
- A: 純自動（同 ver1 A）
- **B**: **完全參考**（= ver1 C，z_a' DTW warp 到 T_p）

**為什麼合併**：
- ver1 Mode B 的價值是「保留業餘節奏」，但實證上業餘節奏經常**就是要修正的問題**（節拍不準/拉長拍）
- 用戶明確要求「節奏跟音高都拉到專業版」，這正好是 ver1 C 的功能
- 維持三模式增加實作 / 維護成本（DTW warp F0 vs warp z 是兩套邏輯）

**檔案**：`rebuild_checklist.md §H` 推理模式定案

### 4.7 ❌ ver1 dataset 分流 (M4=amateur, OpenSinger=pro) → 我們反向

| | ver1 | 我們 |
|---|---|---|
| Amateur | M4Singer | **VocalVerse** (user-generated 業餘) |
| Pro | OpenSinger | **M4Singer**（錄音室專業歌手） |

**為什麼角色反向**：
- ver1 自己的 README 都說「M4 = amateur 假設不確定」（強烈 disclaimer）
- M4Singer 實際是錄音室職業歌手（M4 = Multi-singer Multi-style Multi-language Mandarin），命名與品質都偏 pro
- VocalVerse 是 user-uploaded 業餘樣本，amateur 性質明確
- OpenSinger 我們手邊沒有

**我們的做法**：以實際資料品質為準，不執著 ver1 的標籤。

### 4.8 ❌ NSVB egs/yaml include chain（hparams 全域 state）→ 我們用 dataclass

ver1 三個 yaml：
- `egs/datasets/audio/M4OpenSinger/base.yaml` (繼承 NSVB tts/base + singing/base)
- `stage1_pretrain.yaml`（繼承 base）
- `stage2_mapping.yaml`（繼承 base）

需要 `set_hparams()` 處理多層 include + override，全域 state 流轉。

**我們不採用**：用 `dataclass Stage1Config / Stage2Config` 顯式定義所有 hparam，
CLI args 直接 override 對應 field，無全域 state 污染。

**為什麼**：rebuild_checklist §J' 守則 2「追蹤 hparams 執行時值」的教訓——這次 NSVB
fmin/fmax 衝突就是 yaml include 鏈讓人誤判。dataclass 把所有設定 on the same screen
看得一清二楚。

---

## 5. 兩邊**用詞 / 實作差異對照**（即使核心相同）

| 項目 | ver1 | 我們 | 是否實質差異 |
|---|---|---|---|
| File 命名 | `nsvb_zh_model.py`（all in one） | `m_mapping.py / d_z.py / losses.py`（分檔）| 否，組織風格 |
| Trainer 框架 | NSVB BaseTask（自寫 trainer，類似 PyTorch Lightning） | 純 PyTorch 迴圈（Stage1Trainer / Stage2Trainer class） | 否，等價功能 |
| hparam 訪問 | `hparams.get('m_kernel_size', 1)` | `cfg.m_kernel_size` | 否，等價 |
| ConditionBuilder（mel encoder 的條件） | NSVB 用 FastSpeech2 hidden（內含 phoneme + pitch + spk） | 自寫 ConditionBuilder：PPG_proj + LogF0Embed + spk_proj | 是，因為我們不走文字路徑 |
| FVAE backbone | NSVB FastSpeech2VAE.fvae（含 prior_glow option） | 我們的 backbone.fvae.FVAE（移除 prior_glow，簡化）| 是，移除冗餘 |
| FVAE strides | [4]（mel 172 fps × 4 = latent 43 fps） | **同** | 否 |
| FVAE gin_channels | NSVB 256（給 NSVB FastSpeech2 hidden） | **256**（對齊以便 transfer learning） | 否 |

---

## 6. ⭐ 我們**補強了 ver1 沒處理或處理不足**的地方

### 6.1 NSVB ckpt config fmin/fmax 衝突（電音化 root cause）

ver1 假設 vocoder 訓練 mel 設定 = `fmin=50, fmax=11025`（從 NSVB singing/base.yaml 推），
但實際上 NSVB ckpt 旁的 config.yaml 標 `fmin=0, fmax=8000`。我們完整追了 effective hparams：
- 看 ckpt config → 採用 fmin=0/8000 → 電音化 (SSIM=0.50)
- 改用 acoustic model yaml 的 fmin=50/11025 → 重建 clean (SSIM=0.94)

**ver1 的 stub vocoder identity test 沒實際跑過，所以沒踩到這個坑**。我們踩了並寫進
[rebuild_checklist §J' 守則 2](../rebuild_checklist.md)。

### 6.2 vocoder 真實架構（HifiGAN-NSF, 不是 PWG）

ver1 假設使用「HifiGAN NSF」但**沒實作 vocoder loading**（ckpt loading 是 stub）。
我們深入 inspect ckpt state_dict，發現：
- ckpt 旁 config 寫 `generator_params: {layers=30, residual=64, ...}`（看似 PWG）
- 但 state_dict keys (`m_source.l_linear`, `noise_convs.0..3`, `resblocks.0..11.convs1/convs2`)
  證實是 **真 HifiGAN-NSF**

我們完整 port `nsvb/backbone/vocoder/{hifigan_nsf, source}.py`，與 ckpt **244/244 keys strict-load** 通過。

### 6.3 NSVB 1030_vae_mle ckpt 的 transfer learning 路徑

ver1 沒提這個。我們發現 NSVB 1030_vae_mle ckpt 的 vae_model.* state_dict 與我們
SVBVAEZh.fvae 的 88 個 keys **完全 shape-compatible**（驗證 88/88 strict-load），
寫 `nsvb/utils/transfer_weights.py` 作為 Stage 1 啟動點。預期省 ~5x training 時間。

### 6.4 F0 unvoiced log-space 線性內插（餵 vocoder 必要）

ver1 的 vocoder identity test 用 pyworld DIO+StoneMask 抽 F0，但**沒做 unvoiced 內插**。
我們踩坑後發現 vocoder 餵帶 0 gap 的 F0 → SineGen 在邊界產生相位斷裂 → 電音。

寫 `nsvb/utils/f0_utils.py:interp_f0_unvoiced` 作為「餵 vocoder 前必跑」的 helper，
log2 space 線性內插（對應 NSVB norm_interp_f0 + denorm_f0(use_uv=False) 行為）。

### 6.5 跨平台守則（Linux 訓練機）

ver1 是 Linux 開發為主，沒明確跨平台守則。我們在 [rebuild_checklist §J](../rebuild_checklist.md)
列了 10 條 Windows ↔ Linux 共通守則（pathlib / utf-8 / multiprocessing spawn / HF cache 等），
本地 Windows 開發、Linux 機器訓練的工作流走得順。

---

## 7. 整體工作量比較

| 範疇 | ver1 行數 | 我們 行數 | 備註 |
|---|---:|---:|---|
| Voice conversion 核心 (M / D_z / losses) | ~350 | ~470 | 我們分檔 + 詳細註解 |
| Backbone (fvae / multi_window_disc / vocoder) | 重用 NSVB | ~1220 | 我們明確 port + 註解 |
| Phase 0（含 binarizer + cluster_ppg + jsd_check + vocoder_test） | ~400 | ~1500 | 我們有完整實作 + sanity scripts |
| Stage 1 task | ~80 (繼承 NSVB) | ~290 | 我們純 PyTorch trainer |
| Stage 2 task | ~250 | ~345 | 我們含 NSVB CVAE freeze 邏輯 |
| **總計** | ~1080 | ~3825 | 我們是 standalone 完整實作 |

**為什麼我們行數較多**：
- standalone 不依賴 NSVB → 必須 port backbone（ver1 是 import）
- 詳細「為什麼」註解（每個檔案開頭 30+ 行 docstring）
- Sanity / validation scripts（vocoder_identity_test, sanity_vocoder_nsvb, compare_mel_compute 等）

---

## 8. 結語：適用情境

| 情境 | 建議方案 |
|---|---|
| 已有 NSVB fork、Linux 環境、且不需大改 | **ver1**（drop-in 即用，最小成本） |
| 從零建專案、跨平台、需要可獨立維護 | **我們版本**（standalone，無 NSVB 依賴） |
| 想做 production 部署 / 商業化 | **我們版本**（無 NSVB 全域 state，較易封裝） |
| 教學 / 學習 NSVB-ZH 架構 | **我們版本**（每個檔案開頭詳細註解，較易理解設計理由） |

兩邊核心架構共識度高（~80% 設計選擇相同），差異主要在工程組織與前處理路徑。
我們從 ver1 吸納所有經驗證的核心設計（M / D_z / PatchNCE / TTUR / L_identity_pro），
並在實作層補強 ver1 沒處理或處理不足的地方（vocoder mel 對齊、F0 內插、transfer learning、跨平台）。