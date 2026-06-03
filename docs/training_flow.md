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
│   binarizer.py (含 dereverb + VV chunk 5s) + cluster_ppg.py                  │
│   → M4 ~21K snippets + VV ~21K chunks (5s each), 樣本數 1:1 平衡             │
│                                                                              │
│ Phase 0 gate ① Vocoder identity test  SSIM ≥ 0.90, F0 RMSE ≤ 10 Hz           │
│ Phase 0 gate ② Audio quality probe    SFM/Reverb/HF/SNR JSD < 0.10           │
│ Phase 0 gate ③ JSD(register / phoneme) < 0.05                                │
│ Phase 0 gate ④ PPG cluster: MI(phoneme;register) < 0.3, dwell ≥ 8 frames     │
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
│   - D_z accuracy 0.55–0.75, ‖Δ‖/‖z‖ (magnitude), tdr (trajectory)            │
│   - Risk 2 L5: monitor_audio_quality 每 5000 步                              │
│     (unvoiced_concentration + voiced_spectral_ratio)                         │
│   - Auto-warning: failure mode A (Δ/z<0.03) / B (tdr>1.0) @ step 30000       │
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
data/m4singer/{歌手#歌名}/{idx}.wav                                    # 業餘? 不，職業
data/VocalVerse/{user_id}/{wav_id}.wav                                 # 業餘
data/VocalVerse/VocalVerse_Datasets-human_labels/Amateur_overall_mos_avg5.xlsx
                                                                        # 5 位評審 1-5 分平均 MOS
```

| 資料集 | 檔案數 | mean duration | 總時長 | sample rate | 角色 |
|---|---:|---:|---:|---:|---|
| M4Singer | 20,896（5-sec snippets，699 個 `{歌手}#{歌名}` 目錄） | 5.4 s | ~31 h | 48000 | **professional (z_p)** |
| VocalVerse | 929（full-length recordings，33 個歌手目錄） | 203 s | ~52 h | 44100 | **amateur (z_a)** |

> ⚠ 載入時 `librosa.load(sr=22050, mono=True)` 自動 resample + stereo→mono averaging。

#### 1.1.1 VocalVerse 多維過濾（推薦）⭐

VocalVerse 929 筆中含「near-pro 業餘」樣本（pro 4-dim 評分接近 pro 水準）；若直接全用，
D_z 會在 amateur batch 內看到「z 分布近 pro」的矛盾訊號 → 梯度方向不穩、M 修飾不足。

VocalVerse 同時提供兩份標記（[作者論文 arXiv:2512.06999](https://arxiv.org/abs/2512.06999) §3.1）：

| xlsx | 標記者 | 內容 | 對 NSVB-ZH 的價值 |
|---|---|---|---|
| `Amateur_overall_mos_avg5.xlsx` | 5 位業餘評審/筆 | 整體好聽度 1-5 平均 | 次要 corroborator（與 pro 標記只 0.38 spearman 相關） |
| `Professional_multidim_..._.xlsx` | 1 位 pro 教練/筆 | 4 個維度 1-5：音色/情感/技巧/氣息控制 | **主要 signal**（M 訓練的維度都在裡面） |

##### 主推薦過濾分數：`amateur_score = (技巧 + 氣息控制) / 2`

四個 pro 維度對 NSVB-ZH 各自的關聯性：

| 維度 | 在 NSVB-ZH 的角色 | 用作過濾？ |
|---|---|---|
| **技巧** (Vocal Technique) | M 直接訓練的目標（vibrato/portamento/phrasing）。論文 §3.1.5 標「improvable through training」 | ✅ 主要 |
| **氣息控制** (Breath Control) | M 訓練的目標（與技巧 0.69 spearman 強相關） | ✅ 主要 |
| **音色** (Timbre Quality) | 由 spk_emb 鎖定（Risk 4 防護），論文 §3.1.5 標「largely related to physiological characteristics」**不可改也不該改** | ❌ 不要過濾 |
| **情感** (Emotional Expression) | M 不直接訓練，間接由 breath/dynamics 帶出 | △ 次要 |

##### 閾值對應筆數 / 時長（909 筆有完整 pro 標記）

| `amateur_score ≤ X` | 筆數 | singers | 總時長 | per-singer 中位 | 評估 |
|---:|---:|---:|---:|---:|---|
| 2.0 | 202 | 32/33 | 11.1 h | 6 | 太少，過擬合 |
| 2.5 | 371 | 33/33 | 20.6 h | 11 | 強過濾，可用 |
| **3.0** | **536** | **33/33** | **29.8 h** | **17** | ⭐ **預設**（與 M4Singer 30h 對齊） |
| 3.5 | 676 | 33/33 | 37.8 h | 21 | 含 average，不夠 amateur |

> 929 筆中 20 筆 pro 標記不全（NaN），過濾時統一 `drop`（保守處理）。
> 33 singer 全保留，per-singer 中位 17 樣本（min ~12），spk_emb 訓練不過擬合。
> kept amateur_score 範圍 [1.0, 3.0]，mean 2.40，明確「技術偏弱」區間。

##### 多維 AND 組合（可選）

若想加嚴可同時設多個 max（全部 AND）：

| 指令 | 結果 |
|---|---|
| `--vocalverse-amateur-score-max 3.0` | 536 / 29.8 h（**主推**） |
| `--vocalverse-amateur-score-max 3.0 --vocalverse-mos-max 3.5` | 440 / 24.6 h（去掉「pro 看差但群眾覺得好聽」雜訊） |
| `--vocalverse-amateur-score-max 2.5` | 371 / 20.6 h（強過濾） |
| `--vocalverse-technique-max 2.5 --vocalverse-breath-max 3.0` | 個別維度單獨 cap |

`--vocalverse-timbre-max` flag 雖存在但**不建議用**（音色屬 physiological 特性、由 spk_emb 鎖；過濾音色等於拒絕「音色不夠 pop」的業餘者）。

##### 過濾對訓練架構的影響（量化）

| 項目 | 全 929 筆 | amateur_score≤3.0 (536 筆) | 影響 |
|---|---|---|---|
| Phase 0 binarize 時間 | ~1.5 h | ~0.85 h | -43% |
| Phase 1 dataloader chunk 數 | ~28K（pro 21K + amateur 7K） | ~16K（pro 21K + amateur 4K） | epoch -40%；max_steps=80k 仍綽綽 |
| Phase 2 amateur 訓練多樣性 | 高但混雜 | 中而乾淨 | M 修飾訊號乾淨；過擬合風險微升 |
| Phase 0 JSD gate ③ | 已驗 | **必須重跑**（過濾後分布變動） | gate 條件不變 |
| Phase 1/2 程式架構 | — | — | **完全不變** |

**指令**：

```bash
# 推薦
python -m nsvb.data.binarizer --dataset vocalverse \
    --vocalverse-amateur-score-max 3.0

# Sanity check 標籤分布（不會跑 binarize）
python -m nsvb.data.vocalverse_mos --vocalverse-root data/VocalVerse
```

> `--vocalverse-mos-threshold` 為舊 flag、deprecated 但仍相容；推薦改用
> `--vocalverse-amateur-score-max` 走 pro 4-dim 主信號。

#### 1.1.2 VocalVerse 切 chunk（推薦）⭐

**為什麼必要**：VocalVerse 平均 200s × 536 首（過濾後）vs M4Singer 5.4s × 20,896 個 snippets → **樣本數 1:39 嚴重失衡**。若不切：

| 問題 | 細節 |
|---|---|
| **Stage 1 dataset 嚴重 imbalance** | ConcatDataset 均勻抽樣 → VV 被抽到只 2.5%，CVAE 幾乎只看 M4 分布 |
| **Stage 2 per-sample 過度訪問** | VV 536 sample → 每 sample 訪問 ~3500 次 vs M4 ~92 次，VV 端 overfitting 風險高 |
| **IO 載入浪費** | VV .npz ~120 MB 每首，random crop 只用 ~5s ≈ **95% IO 浪費** |
| **偏離 NSVB 設計 envelope** | NSVB 假設「每樣本=一個 phrase 訓練單位」，PopBuTFy 3-9s 切片完全符合；VV 200s 完全跳出 |

**做法**：binarize 時加 `--vocalverse-chunk-sec 5.0`，每首 200s 切成 ~40 個 5s chunks。

**chunk 邏輯**（[binarizer.chunk_sample](../nsvb/data/binarizer.py)）：
- 固定 stride 5s 切（不做 silence-aware，簡單可預測）
- 每 chunk 各自存 .npz，item_id 加 `__c{NNN}` 後綴
- spk_emb（per-song）每 chunk 重複存一份 → 同首所有 chunks 維持同一歌手
- 最後 < 3s 尾段丟棄（避免訓練不穩；損失 < 0.5% 資料）

**結果對比**：

| Dataset | 改動前 | chunk_sec=5 後 |
|---|---:|---:|
| M4Singer | 20,896 個 5-sec snippets | 20,896（不變） |
| VocalVerse | 536 個 200-sec songs | **~21,440 個 5-sec chunks** |
| 樣本數比 | 1:39 | **~1:1** ⭐ |
| 每樣本 .npz 大小 | M4 ~3 MB / VV ~120 MB | 兩者皆 ~3 MB |
| 訓練 IO 浪費 | VV 95% | **VV ~0%** |

**對齊 max_frames=1500**（[audio_config.DEFAULT_MAX_FRAMES](../nsvb/utils/audio_config.py)）：5s chunk ~860 frames + M4 9s outlier ~1500 frames 都涵蓋；random crop 不再頻繁觸發，每樣本完整作為訓練單位。詳見 [§5.2 tensor shape](#52-訓練時-batch-內-tensor-形狀max_frames1500) 與 [§6.2 / §6.3 配方表](#62-stage-1cvae)。

**指令**（與 §1.1.1 過濾組合使用）：

```bash
python -m nsvb.data.binarizer --dataset vocalverse \
    --vocalverse-amateur-score-max 3.0 \
    --vocalverse-chunk-sec 5.0
```

##### 已知限制：固定 stride 會切穿少數連音 ⚠

固定 5s stride **不做 silence-aware / phrase-aware 對齊**，會切穿連音、vibrato、melisma。粗估每首 200s 歌 39 個內部邊界，約 **~6.5% 的音符在邊界被切**（200s × ~3 notes/s ≈ 600 notes，39 個被切）。

**為什麼仍可接受**（決定不在 Phase 0 修）：
1. **切點特徵乾淨**：mel / F0 / PPG 是在連續整首音訊上算完才 `chunk_sample` 切片（純 array slicing），chunk 邊界**無 STFT edge artifact**
2. **架構對 phrase 完整性不敏感**：Stage 1 CVAE 是 mel→mel 重建、Stage 2 M 是 kernel=1 pointwise + PatchNCE frame-wise——都逐 frame 運作，不依賴「sample 是否完整 phrase」
3. **chunk 內 ~97% 動態完整**：5s chunk 含 ~25-35 個 vibrato cycle，只有邊界 1 個被切；模型也看到大量完整 phrase 的 chunk（~93.5%）

**未來若聽測發現 phrase 級 choppy artifact 才回頭做**（backlog，二選一）：
- **silence-aware snap**（~20 行）：用已存的 `voicing` array，在每個 target 邊界 ±0.5s 內 snap 到最近 unvoiced frame；連續演唱無空隙才 fallback 硬切
- **overlap chunking**（~5 行）：5s chunk + 1s overlap（hop=4s），每個跨邊界音符至少在某 chunk 完整出現一次；代價是 chunks +25%

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

**腳本**：`nsvb/data/cluster_ppg.py`,兩階段:
1. **Stage A**:對所有 .npz 各抽 `frames_per_song` frames PPG → fit `MiniBatchKMeans` → 存 centroids
2. **Stage B**:對每個 .npz 用 centroids 算每 frame 的 phoneme_id → 寫回 .npz

**最終 accept 的設定**(經 Gate ③/④ + 重 cluster 實驗驗證,詳見 [phase0_log.md](phase0_log.md) §3-§5):

| 參數 | accepted | (initial baseline) | 為什麼 |
|---|---|---|---|
| K | **100** | (原 200) | K=200 + unpaired Mandarin → 嚴重 dataset-shortcut(M4 MI=0.862 / phoneme JSD=0.43);降到 100 + DC removal 後 MI=0.216 healthy / JSD=0.16 可接受 |
| `--per-utt-mean-norm` | **on** | (原 off) | 每首歌 PPG 沿時間軸取均值並減掉,移除 Whisper layer 8 的 pitch DC 痕跡(Risk 2b 主防線);fit + assign 必須同 flag |
| frames_per_song | **50** | (原 200) | 42K 檔 × 200 = 8.5M frames → np.concatenate 峰值 ~86 GB OOM;降到 50 ~2.1M frames、峰值 ~21 GB 安全;對 K=100 已遠超收斂需求 |
| max_total_frames | 4,000,000 | (新增 cap) | 防 檔案數 × frames_per_song 失控,硬上限兜底 |
| algorithm | MiniBatchKMeans | 同 | 全 frames ~250M 太大,stream batch 8192 fp32 |

**accept 的 CLI 指令**:
```bash
python -m nsvb.data.cluster_ppg \
    --binarized-root data/binarized \
    --centroids-out data/binarized/ppg_kmeans_centroids.npy \
    --k 100 --frames-per-song 50 --per-utt-mean-norm \
    --stage all
```

⚠ **Stage 2 的 `--phoneme-vocab-size` 必須跟 K 一致**(現為 100;預設 200 會浪費 embedding 容量但仍可運作)。

### 1.5 Phase 0 監控與 gate

> **三個 gate 都要 PASS** 才能進 Phase 1。順序建議：先 ①（vocoder 不過後續都白做），再 ②（音質基線），最後 ③（資料策展）。

#### Gate ①：Vocoder identity test（建議跑兩次）

**腳本**：`scripts/vocoder_identity_test.py`。

| 指標 | PASS 閾值 | MARGINAL | FAIL |
|---|---|---|---|
| mel SSIM | ≥ 0.90 | 0.85–0.90 | < 0.85 |
| F0 RMSE (Hz) | ≤ 10 | 10–20 | > 20 |

**為什麼必要**：vocoder 是凍結 ckpt；若它對中文歌聲重建已經斷裂（mel→wav 失真），後續 M 任何改進都看不到。

**跑兩次**：
- **A. Raw wav**（無 `--apply-loudness-norm`）：vocoder ckpt 訓練分布的對照
- **B. Loud-normed**（加 `--apply-loudness-norm`）：NSVB-ZH binarize 端實際 mel 分布的兼容性 gate

> NSVB 原版 vocoder 訓練時 `loud_norm: false`，但 NSVB-ZH binarize 為 Risk 2（響度對齊）預設啟用 BS.1770 -22 LUFS；兩跑都要 PASS 才能保證訓練 / 推理 mel 與 vocoder 兼容。

**目前實測**（A. raw wav，pyworld + interp）：
- M4Singer:SSIM = 0.94, F0 RMSE = 5.8 Hz → **PASS** ✅
- VocalVerse:**Phase 0 沒單獨跑**,後來 Phase 2 補測發現 SSIM = 0.65, F0 RMSE = 53 Hz → **FAIL** ❌(見下方 caveat)
- B（loud-normed）：尚未跑 — 部署到 Linux GPU 機器後第一件事

> ⚠ **Phase 0 Gate ① 覆蓋不足（2026-05 post-Phase-2 發現)**:當初 Gate ① 只對 M4 跑,
> 沒對 VV 單獨檢測。Phase 2 聽測時發現 amateur 端 wav 有電音,用
> [`scripts/diagnose_stage1_vocoder_path.py`](../scripts/diagnose_stage1_vocoder_path.py)
> 補測得到 VV 在 vocoder(GT mel) 路徑就已 SSIM = 0.65 / F0 RMSE = 53 Hz **大幅 fail**。
> 這不影響 Stage 1/2 訓練本身(mel-domain),但 Phase 3 部署前必須先 vocoder
> fine-tune on Chinese amateur。詳見 [risk.md Risk 10](../risk.md#risk-10vocoder-對-amateur-中文歌聲分布不熟新發現)
> 與 [phase2_outcome.md](phase2_outcome.md)。
>
> **未來重做 Phase 0 的人**:Gate ① 一定要 **對每個 dataset 個別跑**,不能只用平均
> verdict。`scripts/vocoder_identity_test.py --wav-dirs` 已支援多 dir,但 Phase 0
> 沒用全。

#### Gate ②：Audio quality probe（Risk 2 L4）

**腳本**：`scripts/audio_quality_probe.py`。對每個 dataset 抽 N 首歌算 wav 的音質統計，計算跨 dataset 的 JSD。可選 `--apply-dereverb` 跑 dereverb 後再算（模擬 binarize 端進訓練的實際分布）。

**4 個 metric 但可信度分兩級**（實測 M4 vs VV 跑出來的 mitigation response 後分類）：

| Metric | 含義 | 可信度 | PASS 閾值 |
|---|---|---|---|
| **`sfm`** | Spectral flatness（頻譜均勻度） | ⭐ **Reliable**：直接量頻譜，DF3 影響可預測 | JSD < 0.10 |
| **`hf_ratio`** | High-frequency energy ratio | ⭐ **Reliable**：直接量頻譜 | JSD < 0.10 |
| `snr_db` | voiced_E / unvoiced_E ratio | ⚠ **Heuristic**（不是真 SNR）| 對 VV 持續背景噪音 saturate；DF3 同比例壓 voiced+unvoiced 後比例不變 |
| `reverb_sec` | 能量包絡衰減估算 RT60 | ⚠ **Heuristic**（不是真 reverb）| DF3 改 transient 形狀後 heuristic 失準 |

**為什麼必要**：M4Singer (錄音室) vs VocalVerse (user-generated) 的音質域差異，若不通過 dereverb 拉齊，會讓 D_mel/D_z 學到「環境音=amateur 簽名」的捷徑（Risk 2 主防線）。

**Verdict 判讀**（重要：FAIL 不一定等於 mitigation 失效）：

| 狀況 | 判讀 | 行動 |
|---|---|---|
| 全部 PASS | 理想，極少見（M4 vs VV 本質不同） | 直接進 Stage 1 |
| **僅 heuristic FAIL（snr / reverb），reliable PASS（sfm / hf_ratio）** | **形式 FAIL 但實質 mitigation 生效**（典型結果，Colab 實測 dereverb 後 hf_ratio 從 0.26 → 0.05 PASS、sfm 從 0.64 → 0.39 大幅改善） | 視為通過進 Stage 1；Stage 2 訓練時嚴密看 L5 monitor |
| Reliable metric 也 FAIL | mitigation 真的不足 | (a) 確認 binarize 用 dereverb=True；(b) 加 SNR 篩選砍極端 VV 樣本；(c) 換 dataset |

**真正的 Risk 2 ground truth 是 Stage 2 訓中 L5 monitor**（`unvoiced_concentration < 0.55`），它直接量 M 是否在去殘響——比 raw-wav heuristic 公允得多。

**Colab A100 實測值**（apply_dereverb=True，n_per_dir=100）：

| Metric | Raw JSD | Dereverb JSD | Δ | Verdict |
|---|---:|---:|---|---|
| sfm | 0.644 | **0.390** | -39% ↘ | FAIL 但方向對 |
| hf_ratio | 0.258 | **0.048** ✅ | -81% ↘↘ | PASS |
| reverb_sec | 0.113 | 0.118 | +5% → | FAIL（heuristic 限制）|
| snr_db | 0.659 | 0.624 | -5% → | FAIL（heuristic 限制）|

判讀：**Reliable metric 改善 / PASS → mitigation 生效，可進 Stage 1**。

#### Gate ③：資料策展 JSD（register / phoneme）

**腳本**：`nsvb/utils/jsd_check.py`。

| 指標 | 閾值 | 不過怎麼辦 |
|---|---|---|
| JSD(VocalVerse phoneme dist, M4Singer phoneme dist) | < 0.05 | 重採樣 phoneme distribution |
| JSD(VocalVerse register dist, M4Singer register dist) | < 0.05 | 重採樣 register distribution |

**為什麼必要**：D_z 用 phoneme + register 當條件；若兩 dataset 的 phoneme 頻率分布有大差別（例如 M4Singers 比例壓倒性高男聲），D_z 看到「phoneme 出現比例」就能猜業餘/職業，把 M 引導到亂改。

#### Gate ④：PPG cluster 品質 / pitch 污染檢查

**腳本**：`scripts/cluster_ppg_inspect.py`。

**為什麼必要**：Whisper layer 8 是 phonetic 層，但**沒完全把 prosody 抽掉**——對 speech 是小量、對 singing（pitch 是主要變異）會被 k-means 放大成「同一母音不同 pitch 切到不同 cluster」。結果：`phoneme_id` 序列與 `register_id` 序列高度相關 → D_z 的 phoneme + register 兩個條件實質塌成一個 → **隱形版 F0 shortcut**（Risk 6 死灰復燃）。Gate ③ JSD 只看跨 dataset 分布差異，看不到同 dataset 內部的 cluster-register 相關性，所以需要這個 Gate。

| 指標 | HEALTHY | MARGINAL | WARNING |
|---|---:|---|---|
| `MI(phoneme_id; register_id)` (bit) | < 0.3 | 0.3–0.6 | ≥ 0.6 (cluster 被 pitch 污染) |
| `mean_dwell_frames_voiced` | ≥ 8 (~46 ms) | 3–8 | < 3 (cluster 切太細，frame-by-frame 跟 pitch 跳) |

**輸出**：
- `outputs/phase0_cluster_inspect/{dataset}/timeseries_NN_*.png`：抽 N 首歌畫 `phoneme_id` + F0 + voicing 三層疊圖，眼睛交叉檢查 sustained 音/vibrato/換氣段 cluster 是否穩定
- `outputs/phase0_cluster_inspect/{dataset}/dwell_histogram.png`：voiced 段內 cluster dwell length 分布 + 健康閾值線
- `report.json`：所有量化指標

**指令**：
```bash
python -m scripts.cluster_ppg_inspect \
    --binarized-root data/binarized \
    --datasets m4singer vocalverse \
    --phoneme-vocab-size 200
```

**WARNING 補救**（依優先級）：
1. **降 K**（200 → 100）：粗化 cluster，避免 pitch 微差分到不同桶
2. **換 Whisper layer**（試 layer 6 或 10）：layer 6 更近聲學表徵（pitch 訊號可能少）、layer 10 更近語意（音色 invariance 更強）
3. **PPG per-utterance 去 DC**：binarize 時對每首歌 `ppg -= ppg.mean(axis=0)`，去除說話人/錄音常數偏移，再 k-means
4. **phoneme_id mode filter**：對輸出 `phoneme_id` 跑長度 5 的 mode filter 平滑掉 1-2 frame 的瞬時跳變

### 1.6 Train / Val / Test 切割（Phase 1/2 前置步驟）

`cluster_ppg` 跑完、`phoneme_id` 寫回所有 .npz 之後、開始訓練之前，跑一次 [`scripts/make_splits.py`](../scripts/make_splits.py) 切出三份 item_id 列表：

```bash
python scripts/make_splits.py \
    --binarized-root /path/to/binarized \
    --m4-test-singers Alto-2 Tenor-3 \
    --m4-val-songs-per-singer 2 \
    --vv-test-singer-frac 0.10 \
    --vv-val-utterance-frac 0.05 \
    --seed 42
# → /path/to/binarized/splits/{train,val,test}.txt  +  report.json
```

#### 切割原則（SVC/SVB 慣例）

| split | 來源 | 用途 |
|---|---|---|
| **train** ~90% | M4 訓練歌手的非 val 歌曲 + VV 訓練 user 的非 val 來源錄音 | Stage 1 / Stage 2 訓練 |
| **val** ~5% | M4 訓練歌手裡 hold out 2 首歌/人;VV 訓練 user 裡 hold out 5% 來源錄音 | 訓中監控 loss、挑 best ckpt |
| **test** ~5% | M4 整位 hold-out 歌手(2 位);VV 整位 hold-out user(~10%) | Phase 3 推理評估,**訓練永不碰** |

兩條重要原則:
1. **以「歌手 / user_id」為單位 hold out test** —— 同人不可同時出現在 train / test,否則模型可能記 spk_emb 而非學泛化
2. **val 用「整首歌 / 整個來源錄音」為單位** —— 同首歌不同 phrase 不可分到 train / val,否則 random crop 後等於洩漏

#### 為什麼 seed 固定

`--seed 42`(預設)→ 同種子在同 dataset 上**永遠產生相同切割**。`splits/report.json` 內含實際被 hold out 的歌手名單供 audit。

#### dataset.py 如何使用 split 檔

[`BinarizedNSVBDataset`](../nsvb/data/dataset.py) 接 `split_file` 參數,讀檔內 item_id list 把 `self.npz_paths` 過濾到只剩 list 內檔。`stage1.py` / `stage2.py` 預設找 `{binarized-root}/splits/{train,val}.txt`,有就用、無就 fallback 用全 dataset(印 warning)。

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
# infinite_loader 是 `while True: yield from dl` 非快取 generator;
# ⚠ 不用 itertools.cycle:cycle 會把 yield 過的 batch 全 cache 供 replay,
# 而 _to_device 就地 mutate dict 把 CPU tensor 換成 GPU tensor →
# cycle cache 變成「持續累積的 GPU tensor refs」→ ~200 步把 40GB A100 吃光 OOM。
從兩個獨立 dataloader 各抽 batch:
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
| **`delta_over_z`** ⭐ | **0.03–0.30** | M **動多少**（magnitude）：< 0.03 → 太保守；> 0.30 → 過度激進可能破壞內容 |
| **`temporal_diff_ratio`** ⭐ | **< 0.3** | M **動的方向對不對**（trajectory preservation）：mean\|Δ_t M(z) − Δ_t z\| / mean\|Δ_t z\|；< 0.3 = M 只動 spectral 方向不抹平時間軌跡（健康，顫音/滑音保留）；> 1.0 = M 改變的時間導數量級超過 z 自身 → trajectory 被抹平或重寫（警訊）|

> **`delta_over_z` 與 `temporal_diff_ratio` 共同判讀**：
> | Δ/z | tdr | 含義 |
> |---|---|---|
> | 低 (<0.03) | 低 | M ≈ identity，沒在學（**Failure mode A**） |
> | 高 (>0.05) | 低 (<0.3) | ⭐ **健康**：M 動 spectral envelope，trajectory 不變 |
> | 高 | 高 (>1.0) | M 改動把時間結構抹平或重寫（**Failure mode B**） |
> | 低 | 高 | 罕見；M 幾乎不動但偶爾大幅震盪，檢查 dataset 是否異常 |

#### 3.6.2 Risk 2 L5：訓中音質監控

[Stage2Trainer.monitor_audio_quality](../nsvb/task/stage2.py)（每 `audio_quality_monitor_interval`=5000 步觸發一次）：
1. 抽 `audio_quality_monitor_n_samples`（=4）個 amateur 樣本
2. 算 `Δ_mel = mel_modified − mel_baseline`（modified=過 M；baseline=不過 M）
3. 同時報告兩個 ratio：

**(a) `unvoiced_concentration` — 跨 voicing 切分**

| 值 | 含義 | 行動 |
|---|---|---|
| < 0.55 | M 修飾集中在 voiced 段 — 真技術修正 | 繼續訓 |
| 0.55–0.65 | marginal | 留意，再觀察 5000 步 |
| > 0.65 | **Risk 2 警訊** — M 在去殘響/降噪 | resume + `--dmel-mix-amateur-real` 救火 |

**(b) `voiced_spectral_ratio` — voiced 段內部的時間頻譜拆分**

把 voiced 段的 `Δ_mel` 沿時間軸拆成「低時間頻率」(envelope shift) 與「高時間頻率」(F0 trajectory 動態) 兩個分量：

| 值 | 含義 | 行動 |
|---|---|---|
| ≥ 0.7 | ⭐ M 改動以 envelope shift 為主（健康；統一加強共鳴/亮度） | 繼續訓 |
| 0.4–0.7 | marginal | 觀察 |
| < 0.4 | M 改動以高頻時間振盪為主 → 可能在動 F0 trajectory（抹平顫音或加抖動）| 配合 tdr 與聽測診斷 |

每次抽樣會把 mel spectrogram 對比存到 `checkpoints/stage2/audio_monitor/step{N}_sample0.npz`，含：
- `mel_gt / mel_baseline / mel_modified / delta_mel`（視覺化）
- `f0 / voiced`（標 voicing）
- `unvoiced_concentration / voiced_spectral_ratio`（標量指標）

可用 matplotlib 視覺化檢查。

#### 3.6.3 Auto-warnings（一次性 print）

兩種 M failure mode，分別偵測：

| 條件 | Failure mode | 訊息 | 處理 |
|---|---|---|---|
| step ≥ `health_check_step`(30000) 且 `‖Δ‖/‖z‖` 移動平均 < `delta_low_threshold`(0.03) 且 `m_kernel_size==1` | **A：M 太保守** | "M 動的太少（pointwise 表達力不足），建議 `--m-kernel-size 3` 重訓" | 從 stage1 ckpt 接續（M 結構變了，stage2 ckpt 不能直接 resume） |
| step ≥ `health_check_step`(30000) 且 `temporal_diff_ratio` 移動平均 > `temporal_diff_high_threshold`(1.0) | **B：M 抹平 trajectory** | "顫音/滑音等時間結構可能被抹平或重寫" | (1) 用 monitor npz 視覺化 delta_mel 確認；(2) 降 `lambda_adv_z` 或提早停訓；(3) 跑 F0 trajectory 比對 |
| `T_z < 4`（PatchNCE）| 訓練配置錯 | RuntimeWarning：建議 max_frames ≥ 64 | 調大 `--max-frames` |

> Failure mode A 與 B 互斥但可同時觸發：
> - 只 A：M 幾乎沒動（kernel=1 表達力問題，切 kernel=3）
> - 只 B：M 動得太多且方向錯（過度激進）
> - A+B：罕見，通常是訓練不穩定，檢查 loss 與 dataset 完整性

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

FVAE encoder/decoder 的 stride conv 要求 `T_mel % LATENT_DOWN_FACTOR == 0`。訓練時 `max_frames=1500` 剛好是 4 倍數繞過；推理時 user 輸入長度任意，必須在 pipeline 內補齊：

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
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt \
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
    vocoder_ckpt="checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt",
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

### 5.2 訓練時 batch 內 tensor 形狀（max_frames=1500）

```
T_mel (max in batch)       例：1500   (~ 8.7 sec @ hop=128)
T_z   (= floor((T_mel-4)/4)+1)  例：375   (= 1500/4)
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

### 5.3 Condition input ↔ Phase 0 feature 對照表

每個有 condition 輸入的模型 / loss 對應的 Phase 0 特徵來源。**業餘端來自 VocalVerse .npz、職業端來自 M4Singer .npz**;Phase 3 推理時來自使用者輸入 wav 走同 Phase 0 binarize pipeline 即時抽出。

下標慣例:`_a` = amateur(VocalVerse),`_p` = professional(M4Singer),`_p_ref` = Mode B 的 pro 參考。

#### 5.3.1 Stage 1(CVAE pretrain,m4 + vv 兩邊都當 real)

| 模組 | Input(被建模)| Condition(Phase 0 feature)| Mask |
|---|---|---|---|
| FVAE encoder φ | `mel` | `ppg` + `f0` + `spk_emb` ← 三者由 `SVBVAEZh.condition` 拼成 g `[B,256,T]`(ppg→128 + log2 f0→32 + spk_emb→96 expanded)| `mel_mask` |
| FVAE prior p(z) | - | N(0, I) 標準正態 | - |
| FVAE decoder θ | `z`(來自 φ) | 同 encoder 的 g(`ppg` + `f0` + `spk_emb`)| `mel_mask` |
| D_mel | `mel`(real)/ `mel_recon`(fake)| 無 | - |

**Stage 1 不用 `register_*` 跟 `phoneme_id`**(這兩個是 Stage 2 D_z 才用);所以 Stage 1 啟動可以**不等 cluster_ppg 跑完**(dataset.py 對 phoneme_id 缺失回 -1 sentinel)。

#### 5.3.2 Stage 2(Mapping training,φ + θ 從 Stage 1 ckpt 載入並凍結)

| 模組 | Input | Condition(Phase 0 feature)| Mask |
|---|---|---|---|
| 凍結 φ(`_encode_and_downsample`,with no_grad,取 deterministic m_q)| `mel_a` 或 `mel_p` | `ppg_a/p` + `f0_a/p` + `spk_emb_a/p`(各自的)| `mel_mask` → 下採成 `mask_z` |
| **ResidualM** | z (latent) | **無**(kernel=1 pointwise MLP,純動 z 不看任何條件) | - |
| **D_z** | `z_a` / `z_p` 或 `M(z_a)` | **`register_soft` + `phoneme_id`** ← 下採到 z rate:`register_soft → linear interp [B,T_z,5]`,`phoneme_id → stride sample [B,T_z]` 再 embed 32 維 | `mask_z` |
| 凍結 θ(`_decode_with_mapped_z`,用於 L_adv_mel)| `M(z_a)` | **`ppg_a` + `f0_a` + `spk_emb_a`(業餘端)** ⚠ | `mel_mask` |
| D_mel | `mel_p`(real,只看 pro)/ `decode(M(z_a))`(fake)| 無 | - |
| L_PatchNCE | `z_a` ↔ `M(z_a)`(proj head 128→64)| 無(batch-internal negatives)| `mask_z` |
| L_id_pro(20% prob)| `z_p` ↔ `M(z_p)` | 無 | `mask_z` |

⚠ **凍結 θ 在 `_decode_with_mapped_z` 用的是業餘端 condition**(不是 pro 端),理由是 Mode A 推理本就用業餘 F0;訓練端條件對齊推理 → train/test 一致。但**副作用**:D_mel real = pro mel(pro F0 生),`mel_g` 永遠帶業餘 F0 痕跡 → D_mel 可用「F0 不夠 pro」當捷徑判別 → `l_adv_mel` 有不可消地板。實測 step ~35K 看到 `l_adv_mel` 漸升 + `tdr` 漸升,正是這條 confound 在發生。詳 [risk.md §二.3](../risk.md)。

#### 5.3.3 Phase 3 推理

**Mode A**(預設,純自動,只吃使用者業餘音檔):

| 子步驟 | Input | Condition | 備註 |
|---|---|---|---|
| 對使用者 wav 跑 Phase 0 pipeline | wav | - | 即時抽 mel_a / ppg_a / f0_a / spk_emb_a |
| φ 編碼 → z_a → M(z_a) | mel_a | ppg_a + f0_a + spk_emb_a | encoder 走 cvae.condition |
| θ 解碼 → mel_out | M(z_a) | **ppg_a + f0_a + spk_emb_a**(全業餘端,**包括 F0**) | F0 用業餘 = Mode A 設計 |
| vocoder → wav_out | mel_out | f0_a(連續、unvoiced log-space 內插) | HifiGAN-NSF SineGen 需連續 F0 |

**Mode B**(完全參考,需同首歌 pro 參考):

| 子步驟 | Input | Condition | 備註 |
|---|---|---|---|
| 對 x_a 跑 Phase 0 pipeline | wav_a | - | 抽 mel_a / ppg_a / f0_a / spk_emb_a |
| 對 x_p_ref 跑 Phase 0 pipeline | wav_p_ref | - | 抽 mel_p / ppg_p / f0_p_ref |
| φ 編碼 → z_a → M(z_a) | mel_a | ppg_a + f0_a + spk_emb_a | 同 Mode A 前段 |
| DTW alignment(a vs p_ref)| 兩首 PPG / mel | - | 算出 warp index `[T_p]` |
| `torch.gather` warp z 到 p_ref 時間軸 | M(z_a) | warp index | M(z_a) 長度從 T_z_a 變 T_z_p |
| θ 解碼 → mel_out | warped M(z_a) | **ppg_a/warped + f0_p_ref + spk_emb_a** ⚠ | F0 用 pro 參考、音色用業餘 |
| vocoder → wav_out | mel_out | f0_p_ref(log-space 內插)| 輸出長度 = T_p,**無法配原伴奏** |

⚠ Mode B 的 `spk_emb` 仍用業餘端(不是 pro 端歌手)→ **Risk 4 防護**(防音色被推到 pro 歌手身上)。Mode A 全部用業餘端是因為 unpaired 場景不可能有 pro 參考。

#### 5.3.4 速查總表

| Phase 0 feature | Stage 1 用? | Stage 2 用? | Phase 3 推理用? |
|---|---|---|---|
| `mel` | ✓ encoder input + reconstruct target | ✓ encoder input(zh) + D_mel real(pro)| ✓ encoder input |
| `wav` | (vocoder Gate ① 用) | (Risk 2 monitor 拿來重抽 mel)| ✓ vocoder 端輸入也是;但訓練不用 |
| `f0` | ✓ encoder + decoder condition | ✓ encoder + decoder(`_decode_with_mapped_z`)condition | ✓ encoder + decoder + vocoder |
| `voicing` | (binarize 時用來決定 F0=0)| 同 | (用於 vocoder F0 內插) |
| `ppg` | ✓ encoder + decoder condition | ✓ encoder + decoder condition | ✓ encoder + decoder condition |
| `spk_emb` | ✓ encoder + decoder condition | ✓ encoder + decoder condition | ✓ encoder + decoder condition |
| `register_soft` | ✗ 不用 | ✓ **D_z condition**(下採到 z rate)| ✗ 推理不用(D_z 訓練端才用) |
| `register_id` | ✗ 不用 | (僅 JSD 統計用,訓練不用) | ✗ |
| `phoneme_id` | ✗ 不用 | ✓ **D_z condition**(下採到 z rate,embed 32 維)| ✗ |
| `meta_*` | (索引用) | (索引用) | (索引用) |

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
batch=16, max_frames=1500, max_steps=80k (with NSVB init) ~ 200k (scratch)
```

### 6.3 Stage 2（Mapping）
```
M:    kernel=1 (or 3), hidden=256, num_layers=4, init_delta_scale=1e-2
D_z:  hidden=256, num_layers=4, kernel=5, vocab=200, embed=32
PatchNCE: proj=64, num_patches=128, temp=0.07
loss weights: NCE=1.0, adv_z=1.0(after warmup 5k), adv_mel=0.2, id_pro=0.1@0.2_prob
optim TTUR: lr_M=1e-4, lr_Dz=4e-4, lr_Dmel=1e-5  (β=(0.5, 0.999))
batch=16, max_frames=1500, max_steps=120k

D_mel real source: pro-only (default) | pro+amateur if --dmel-mix-amateur-real

monitors (per log_interval=50):
  - delta_over_z          (M 動多少, healthy 0.03–0.30)
  - temporal_diff_ratio   (M 是否抹平 trajectory, healthy < 0.3)
audio quality monitor (per audio_quality_monitor_interval=5000):
  - unvoiced_concentration   (Risk 2 L5, < 0.55 healthy)
  - voiced_spectral_ratio    (M 改 envelope vs F0 traj, ≥ 0.7 healthy)
auto-warnings @ step 30000:
  - failure mode A: Δ/z < 0.03 + kernel_size=1  → 切 kernel=3
  - failure mode B: tdr   > 1.0                  → M 抹平 trajectory
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
    --vocoder-ckpt /path/to/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt \
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

# Gate ④：PPG cluster 品質（MI / dwell 檢查）
PYTHONPATH=. python -m scripts.cluster_ppg_inspect \
    --datasets m4singer vocalverse --phoneme-vocab-size 200

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
    --vocoder-ckpt /path/to/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt \
    --input-a path/to/amateur.wav \
    --output outputs/mode_a_result.wav

# Mode B：加 --pro-ref（同首歌專業參考），輸出長度 = T_p
PYTHONPATH=. python -m scripts.infer ... \
    --input-a path/to/amateur.wav \
    --pro-ref path/to/pro_reference.wav \
    --output outputs/mode_b_result.wav
```

---

## 附錄 A：PopBuTFy 跨語言驗證流程（thesis §4.6）

NSVB-ZH 架構設計為 language-agnostic。為驗證跨語言通用性，我們在 NSVB 原版所附的
英文 **paired** SVB 資料集 PopBuTFy 上重跑相同 Phase 0 + Phase 1/2 pipeline。
PopBuTFy 提供同歌手同首歌的 amateur/professional 配對 → 補上中文 unpaired 流程
缺的 **paired metrics**（MCD/SSIM/F0 RMSE vs paired pro）與 **NSVB 原版 ckpt baseline 對照**。

### A.1 資料位置與配對 dump

PopBuTFy 來源（本機）：`<NSVB workspace>/data/processed/PopBuTFy_new/data/`
- 結構：`{Singer}#singing#{Song}_{Amateur|Professional}/*_{idx}.mp3`
- 共 ~904 folders = ~452 paired (singer, song) pairs、24 speakers、~14K chunks/side

先用 adapter dump amateur→pro 配對 dict（§4.6 paired eval 直接讀）：
```bash
python -m nsvb.data.popbutfy_adapter \
    --root <NSVB_workspace>/data/processed/PopBuTFy_new/data \
    --pairing-json outputs/popbutfy_pairing.json
```

### A.2 Binarize（兩 side 分別跑）

`--data-root` 直接指向 PopBuTFy_new/data 本身（不像 m4/vv 加 subdir）：
```bash
PYTHONPATH=. python -m nsvb.data.binarizer --dataset popbutfy_pro \
    --data-root <NSVB_workspace>/data/processed/PopBuTFy_new/data \
    --out-root data/binarized

PYTHONPATH=. python -m nsvb.data.binarizer --dataset popbutfy_amateur \
    --data-root <NSVB_workspace>/data/processed/PopBuTFy_new/data \
    --out-root data/binarized
```
備註：
- `load_wav` 走 librosa 原生支援 mp3，pipeline 內部處理跟 wav 完全等價
- Phase 0 gates 仍然要過（SSIM、JSD、cluster MI 等），門檻同中文流程
- 配對假設 chunk N(amateur) ↔ chunk N(pro)，下游 eval 階段若需 fine-grained 對齊
  再走 DTW（NSVB 原版做法）

### A.3 訓練與評估

訓練：與中文流程同 `nsvb.task.stage1` / `nsvb.task.stage2` 配方，
僅 `--amateur-dataset popbutfy_amateur --pro-dataset popbutfy_pro`。

評估：除了既有 unpaired metrics，加跑 paired eval（讀 `outputs/popbutfy_pairing.json`）
量化 MCD / SSIM / F0 RMSE vs paired pro；同 ckpt 也跑 NSVB 原版作為 baseline。