# 評估指標解讀指南

這份文件給「不熟聲音生成模型、但想看懂 NSVB-ZH 訓練與評估結果」的人。
讀完應該能：
- 在每份報告（log / .npz / .json / .csv / .md）中**找到對應指標**
- 知道每個指標**為什麼存在**、**健康範圍**、**反映什麼性質**
- 想做某種修正時，**該盯哪幾個指標**判斷是否生效

## 0. 先理解三個術語

| 術語 | 含意 |
|---|---|
| **mel** | 把音檔頻譜壓成 80 個頻率帶的「時間 × 80」矩陣。視覺化就是聲紋圖。M 的修飾最終都呈現在這個矩陣上。 |
| **latent z** | 把 mel 壓縮過後的低維表示（VAE encoder φ 出來的東西）。M 在這層動手腳，再讓 decoder θ 把改過的 z 解回 mel。 |
| **voiced / unvoiced** | voiced = 有發聲的 frame（聲帶振動，F0>0）；unvoiced = 無聲段、子音、呼吸聲、停頓（F0=0）。一首歌約 50% voiced、50% unvoiced。 |

> 這個系統是「業餘 → 修飾後的業餘」風格轉換，**M 不該換歌手、不該換歌詞、不該換旋律**。它應該只動「音色技巧細節」（共鳴、咬字穩定度、氣息控制）。

---

## 1. 指標來源全表

下表是「每個指標來自哪裡、什麼步驟產生」的完整對照。表格中：

- **訓中** = Stage 2 訓練時即時計算/落地
- **訓後** = 訓練結束後跑分析腳本得到

| 指標 | 出現在 | 何時/由誰產生 |
|---|---|---|
| `m_total` / `d_z` / `d_mel` / `l_nce` / `l_adv_z` / `l_adv_mel` / `l_id_pro` | 訓練 log（`logs_v2/stage2_v2_*.log`）每 50 步一筆 | **訓中**：`nsvb/task/stage2.py` `train_step()` 算 |
| `delta_over_z` / `temporal_diff_ratio`（**latent 空間版**）| 同上 | 訓中：`stage2.py train_step()` 末段算 |
| `unvoiced_concentration` / `voiced_spectral_ratio` / `Δ_voiced_E` / `Δ_unvoiced_E` | log 內 `[stage2-monitor step N]` 行，每 5000 步一筆 | 訓中：`stage2.py monitor_audio_quality()` |
| 訓中 mel 對比 npz：`mel_gt` / `mel_baseline` / `mel_modified` / `delta_mel` | `checkpoints_v2/stage2_v2/audio_monitor/step{N}_sample0.npz` | 訓中：`stage2.py monitor_audio_quality()` 同上,但 dump 一個 sample 的 mel 陣列供後續視覺化 |
| Training log 整體摘要：milestone trajectory / first-crossing / 健康判定 | `runs/stage2_v2_xxx.summary.md` | **訓後**：`scripts/summarize_stage2_log.py runs/xxx.log` |
| **mel 域** `mel_l1_vs_orig` / `mel_l1_vs_recon` / `pro_dist_*` / `pro_direction_alignment` / mel 版 `unvoiced_concentration` / `voiced_spectral_ratio` / `hf_energy_increase` / `temporal_diff_ratio_mel` | `outputs/stage2_v2_eval/report.md` + 各 sample 內 `metrics.json` | **訓後**：`scripts/stage2_mel_eval.py`（重用 listening 跑出來的 mel 或 fresh inference） |
| mel 視覺化（每 sample 一張 stacked spectrogram + F0 overlay）| `outputs/stage2_v2_eval/{sample}/mel_grid.png` | 同上,`stage2_mel_eval.py render_mel_grid()` |
| 全 sample × 全 step 的指標矩陣 | `outputs/stage2_v2_eval/metrics_aggregate.csv` | 同上 |
| M4 對照 vs amateur 比 | `outputs/stage2_v2_eval/report.md` 末段 | 同上 |
| Phase 0 vocoder identity（mel SSIM / F0 RMSE） | `outputs/phase0_vocoder_*/` | **訓前**：`scripts/vocoder_identity_test.py`，做為 Phase 0 gate |
| Phase 0 audio quality probe（sfm / hf_ratio / snr_db / reverb_sec 的 JSD）| `outputs/phase0_audio_quality/` | **訓前**：`scripts/audio_quality_probe.py` |

> **想看訓中即時行為**：讀 `summarize_stage2_log.py` 出的 summary。
> **想看訓後對 val 樣本的模型表現**：讀 `stage2_mel_eval.py` 出的 report。
> 兩份不衝突，互補。Summary 看訓練動態，eval 看實際輸出品質。

---

## 2. 指標分類解釋

按「想看 M 的什麼性質」分類。每組指標只解釋設計動機跟健康判讀，公式詳細請看 [scripts/stage2_mel_eval.py](../scripts/stage2_mel_eval.py) `compute_metrics()` 與 [nsvb/task/stage2.py](../nsvb/task/stage2.py) `train_step` / `monitor_audio_quality`。

### 2.A 修飾幅度 — M 實際改了多少

> ⚠ **名稱注意**：這組指標**反映「M 改了多少」**,但**不能直接證明 content 被保留**。
> 真正的 content preservation 需要額外指標(PPG similarity / ASR-based CER /
> phoneme posterior distance / duration alignment 一致性),目前 backlog,尚未實作。
> 如果未來想驗收「M 沒亂改歌詞」,要新增這組指標,不要把 `mel_l1_vs_recon` 當代理。

| 指標 | 直觀含意 | 健康範圍 | 在哪看 |
|---|---|---|---|
| `mel_l1_vs_orig` | M+decoder 輸出 vs 原 amateur mel,全部差異總和 | 含 VAE 重建底噪,單看意義有限 | eval `report.md` |
| `mel_l1_vs_orig_extra` | 上面那項扣掉 baseline,**只剩 M 在 baseline 之上的修飾量** | 0 ~ 0.3 健康(M 在 baseline 之上多動了一些) | eval `report.md` |
| `mel_l1_vs_recon` | M+decoder 輸出 vs Stage 1 only baseline,**M 的純貢獻** | 訓練開始 ~0,訓中增大但 < 1.0 | eval `report.md` |
| `delta_over_z`(**latent 版本**) | ‖M(z) − z‖ / ‖z‖;M 在 latent 空間動了 z 多少 | 0.03 ~ 0.30 「健康」(文件 §3.6.1) | 訓中 log + summary |

**為什麼有兩個版本（mel 跟 latent）**：
- latent 版（`delta_over_z`）便宜：訓中順手算。**但它的數值跟「實際聽起來改變多少」對應很糟** — 0.87 看起來「過度修飾」，但在 mel 域可能是 envelope shift 0.5（健康範圍）。
- mel 版（`mel_l1_vs_recon`）是 ground truth：mel 是真正餵給 vocoder 的東西。

**v2 經驗教訓**：latent 版 0.87 讓我們以為訓壞了，mel 版顯示其實是健康狀態。**判訓練好壞優先看 mel 版**。

**`_extra` 是什麼**:很多 mel 指標需要跟 GT mel 比(`mel_l1_vs_orig`、`temporal_diff_ratio_mel`、
`hf_energy_increase`),但 Stage 1 VAE 重建本身就會跟 GT 差(tdr_mel ~1.0、hf 略偏)。
直接看絕對值會把 VAE 噪音底歸咎給 M。**`_extra` 版 = current step − baseline(`_1_stage1_recon`)**,
扣掉 VAE 重建底噪後 M 的真實貢獻。建議優先看 `_extra` 版。

### 2.B 修飾方向 — M 改的「方向」對嗎

| 指標 | 直觀含意 | 健康範圍 | 在哪看 |
|---|---|---|---|
| `pro_dist_orig` | 原 amateur 樣本 envelope 到 pro 平均的距離 | reference value | eval `report.md`，metrics.json |
| `pro_dist_out` | M 修飾後 envelope 到 pro 平均的距離 | < `pro_dist_orig` | 同上 |
| `pro_dist_delta` | = `pro_dist_orig − pro_dist_out`，**修飾把樣本拉近 pro 多少** | **> 0**（正號=往 pro 靠）| 同上 |
| `pro_direction_alignment` | cos(修飾向量, 從 amateur 走到 pro 的向量)；**1=方向完全對，0=正交，-1=反向** | **> 0.3 健康**, > 0.5 很好, < 0 反向 | 同上 |

**為什麼這組指標關鍵**：
SVB（singing voice beautification）是「unpaired」任務 — 同一句業餘錄音沒有對應的 pro 版可比。我們不能說「M 應該生成什麼」。**唯一可量化的「對的方向」就是「pro 整體分布的方向」**。
具體做法：取 N 個 M4Singer pro 樣本，對每個算 mean envelope（沿時間軸取平均得到 80 維向量），再平均成「pro 平均長相」`pro_mean_env`。然後比較任何輸出 envelope 跟它的距離 + 方向。

**怎麼讀數字**（v2 step 30000 為例）：
- `pro_dist_delta = +0.23`：修飾後比原 amateur 接近 pro 平均 0.23 個距離單位 → 確實往 pro 走
- `pro_direction_alignment = +0.74`：修飾向量跟「往 pro 的方向」cosine = 0.74 → 七成走對方向，剩下三成是 noise 或副作用

> ⚠ **重要 caveat — pro_direction_alignment 能/不能證明什麼**:
> 這個指標**只看 time-averaged envelope**(沿時間軸平均後的 80 維向量)到 pro 平均的方向。
> 它能說的:「輸出的頻譜包絡**平均形狀**有沒有往 pro 靠」。
> 它**不能證明**:
> - phrase 表現更專業(沒看時間動態)
> - vibrato 更自然(被時間平均掉了)
> - pitch 更準(F0 不在 envelope 內)
> - 咬字更好(phoneme 細節被平均)
> - 聽感一定更好(envelope 對但顫音被改寫,聽起來可能變糟)
>
> 適合當 **model selection proxy**(在多個 step 中挑相對最 pro 的),**不能取代主觀聽測**。
> 數字高也要看 §2.C 健康指標跟 §2.D artifact 都通過,才算「真的更 pro」。

### 2.C 健康指標 — M 的修飾「集中在哪」

這組指標檢查「M 改的東西是不是該改的東西」。

| 指標 | 直觀含意 | 健康範圍 | 在哪看 |
|---|---|---|---|
| `unvoiced_concentration` | M 的修飾能量在 unvoiced（無聲段、呼吸聲、子音）佔的比例 | **< 0.55 健康**, > 0.65 警訊（Risk 2） | 訓中 log + eval `report.md` |
| `voiced_spectral_ratio` | voiced 段裡，M 的修飾**慢時間變化成分**（envelope shift）佔的比例 | **≥ 0.70 健康**, < 0.40 警訊 | 同上 |

**為什麼這兩個重要**：

「M 該改什麼」的設計意圖是：
- ✅ **voiced 段做 envelope shift**（共鳴強化、formant 微調）→ 聽起來更 pro
- ❌ **不該大幅改 unvoiced**（去殘響 / 去呼吸聲 / 抹掉子音細節）→ 那叫做「去環境噪音」，不是「美化歌聲」

`unvoiced_concentration` 抓「修飾能量是否異常集中在無聲段」。如果 M 學會用「去掉業餘錄音的環境音」假裝 pro 化（Risk 2 的擔憂），這個數字會飆過 0.65。

`voiced_spectral_ratio` 抓「voiced 段裡的修飾是慢還是快」。慢時間變化 = envelope 平移（音色微調，自然）。快時間變化 = 高頻 jitter（顫音/滑音被改寫，破壞時間細節）。健康值 ≥ 0.7 表示主要在做 envelope shift。

**v2 經驗**：
- `voiced_spectral_ratio 0.96` ✅ — voiced 段修飾是 envelope shift,健康
- `unvoiced_concentration 0.59` ⚠️ — **不是完全 healthy 而是 marginal**(< 0.65 警訊線但已偏向 unvoiced)。
  Risk 2 沒爆,但 M 的修飾有點偏到 unvoiced 段。**未來若要再優化,降低 unvoiced delta 是優先目標**
  (例如加 voiced/unvoiced 修飾比例 loss)。
- 不要把 v2 的 uv_conc 0.59 當「完全沒問題」 — 它是「可接受但有改善空間」。

### 2.D Artifact 檢查 — M 是不是在加噪

| 指標 | 直觀含意 | 健康範圍 | 在哪看 |
|---|---|---|---|
| `hf_energy_increase` | 高頻(mel bin ≥ 50)能量相對 orig 的變化比例 | 含 VAE 噪音底,看 `_extra` 比較準 | eval `report.md` |
| **`hf_energy_increase_extra`** | 上面那項扣 baseline,**M 的純貢獻** | **±0.20 healthy / −0.35 ~ −0.20 marginal / < −0.35 ❌(削掉氣聲)/ > +0.50 ❌(hiss)** | eval `report.md` |
| `temporal_diff_ratio_mel` | mel 域時間導數差異 / 原 mel 自身時間導數 | 含 VAE 噪音底(baseline 就 ~1.0!),看 `_extra` 比較準 | eval `report.md` |
| **`temporal_diff_ratio_mel_extra`** | 上面那項扣 baseline,**M 在 baseline 之上多加的時間導數噪音** | **< 0.3 健康, > 1.0 嚴重** | eval `report.md` |
| `temporal_diff_ratio`(**latent 版**)| 同上但在 latent 算 | 0.3 健康, 1.0 嚴重 | 訓中 log |

**`hf_energy_increase_extra`**:漂得太正 → M 加了高頻 hiss / 金屬感。漂得太負 → M 把高頻細節抹掉(齒音 s/sh/ts、氣聲、亮度)。
**注意非對稱**:正向 +0.50 才警訊,負向 -0.35 就警訊 — 因為高頻 detail 一旦抹掉很難 vocoder 端補回來。

**`temporal_diff_ratio_mel_extra`**:大 → M 在 baseline 之上又改寫了時間結構(顫音 / 滑音的細節)。
若 `_extra` 接近 0(像 v2 step120K 0.04)代表 M 動的時間導數跟 VAE 本身差不多,**沒額外破壞時間結構**。

**v2 經驗教訓**:`temporal_diff_ratio_mel` 原始值 1.22 看似 ❌,但 baseline 就 1.18 — `_extra` 只 0.04,**M 在時間導數上其實沒做大事**。
原本看絕對值會誤判「M 過度破壞時間結構」,扣掉 baseline 後才看到 M 是清白的。

**為什麼 baseline tdr_mel 就 ~1.0**:VAE 重建有固有的 frame-to-frame 微小 jitter,而 mel 自身的時間導數很慢(voiced 段平滑),分母小 → ratio 容易飆到 1+。
這不代表 Stage 1 訓壞,是 metric 對 small denominator 敏感 + VAE 的 KL bottleneck 必然帶來的精度損失。詳見 [phase2_outcome.md §3](phase2_outcome.md) 的 latent vs mel 衝突討論。

### 2.E 訓中 GAN 動態指標

這些跟「訓練是否平衡」有關，不直接反映「輸出品質」，但失衡會直接導致 M 學壞。

| 指標 | 直觀含意 | 健康範圍 |
|---|---|---|
| `d_z` | D_z 判別 amateur z 跟 pro z 的 loss | ~0.15–0.30 平衡；< 0.05 太強（M 跟不上）；> 0.5 太弱 |
| `l_adv_z` | M 騙 D_z 的 loss | ~1.5–2.5 通常；> 3 表示 D_z 把 M 壓死 |
| `l_adv_mel` | M 騙 D_mel 的 loss | v1 ~5（高 floor，D_mel confound）；v2 freeze 後 ~0.8 |
| `l_nce` | PatchNCE 對比學習 loss | 緩慢下降到 ~0.4 穩態 |
| `l_id_pro` | M 對 pro 輸入做 identity 的 loss | < 0.05 ✅（很小代表 M 沒亂改 pro）|

### 2.F Phase 0 gate 指標（vocoder 與資料的健康檢查）

這些**在訓練之前**就該過。如果不過，後面任何指標都不可信。

| 指標 | 用途 | 健康閾值 |
|---|---|---|
| mel SSIM | vocoder identity test — pretrained vocoder 對中文歌聲 mel 是否還能 reconstruct 出乾淨 wav | ≥ 0.90 |
| F0 RMSE | 同上，但看 F0 精度 | ≤ 10 Hz |
| `sfm` JSD | M4 跟 VV 兩 dataset 的 spectral flatness 分布差 | < 0.10 |
| `hf_ratio` JSD | 高頻能量比例分布差 | < 0.10 |
| `snr_db` / `reverb_sec` JSD | 訊噪比 / 殘響估計 | < 0.10（heuristic，可放寬）|

詳見 [training_flow.md §1.4](training_flow.md)。

---

## 3. 怎麼讀三種主要報告

### 3.1 訓中 log summary（`runs/stage2_v2_xxx.summary.md`）

讀法順序：
1. **整體判定**段 — 一行說「Δ/z 跟 tdr 落在哪」
2. **末段 10% 穩態**表 — 訓練收斂在哪
3. **Milestone trajectory** — 看 metric 走勢，找「壞起來的轉折點」
4. **First-crossing 時間軸** — 哪一步出第一個警訊
5. **Warmup 期狀況** — D_z 啟用前 M 的「自由失控」程度

**注意**：這份 summary 全部基於 **latent 空間**指標。如本指南 §2.A 所述,latent metric 數值對「實際聽起來」對應差,不要單獨依賴它判訓壞。

### 3.2 mel-domain eval（`outputs/stage2_v2_eval/report.md`）

讀法順序：
1. **Per-step aggregate** 表 — 一覽各 step 的平均健康狀態，看每個指標的 ✅/⚠️/❌
2. **M4 vs VV ratio** — M 是否 amateur-specific（健康 > 2×）
3. **找 best step**：
   - `pro_direction_alignment` 越高越好
   - `mel_l1_vs_recon` 不要太大（修飾不該爆炸）
   - 各健康指標都 ✅
4. **per-sample mel_grid.png** — 視覺確認「修飾長相」是否合理

### 3.3 mel_grid.png 視覺判讀

每張圖是同一個 sample 在不同 step 的 mel 疊圖，紅線是 F0：
- 排版：`_0_orig`（原 amateur）→ `_1_stage1_recon`（VAE 重建 baseline）→ `step5K` → ... → `step120K`
- 共同 vmin/vmax，色階一致可橫比
- 看「voiced 段 envelope」是否漸漸往 pro 風（共鳴帶集中、frequency lobes 清晰）
- 看「unvoiced 段」是否被 M 抹掉（深色變淺色 = 能量被刪）
- 看 F0 紅線軌跡是否一致 — 我們**不該動 F0**，所有列的紅線應該長一樣

---

## 4. 未來修正方向 ↔ 指標對照

針對「想做某種改動，要怎麼判斷有沒有效」整理。

### 4.1 如果想 **加強 pro 化效果**

目標：讓 M 更明顯地把 amateur 變 pro。

- 動作：增 `lambda_adv_z`、`lambda_patchnce`、或加 `lambda_adv_mel`
- 主指標：
  - `pro_direction_alignment` ↑（想拉到 > 0.85）
  - `pro_dist_delta` ↑
  - M4 vs VV ratio：應**維持 > 2×**，避免 over-fit 把 pro 也亂改
- 副指標（不要爆）：
  - `unvoiced_concentration` 不要爆 > 0.65（不能用「去業餘環境音」假 pro 化）
  - `voiced_spectral_ratio` 不要掉 < 0.40（不能去動顫音時間結構）
  - `hf_energy_increase` 不要爆 > 0.50（不能加金屬感 hiss）

### 4.2 如果想 **壓 M 過度修飾**（怕後期 over-fit）

目標：讓 M 修飾更克制，保護 amateur 自身音色。

- 動作：增 `lambda_id_pro`、增 PatchNCE 權重、降 D_z lr、提早停訓
- 主指標：
  - `mel_l1_vs_recon` ↓（修飾量小一點）
  - `delta_over_z` ↓（latent 修飾少一點）
  - `temporal_diff_ratio_mel` ↓
- 副指標（不要垮）：
  - `pro_direction_alignment` 不要掉 < 0.3（不要修到零）
  - `pro_dist_delta` 不要負

### 4.3 如果想 **救「呼吸/子音被殺」問題**（Risk 2）

目標：避免 M 走「去 amateur 環境音」捷徑。

- 動作：[risk.md §二.3](../risk.md) f0_support 想法；或加 voiced/unvoiced 修飾比例 loss
- 主指標：
  - `unvoiced_concentration` ↓（要 < 0.55）
  - `hf_energy_increase` 趨近 0（不要把高頻細節挖掉）
- 副指標：
  - 看 `mel_grid.png` 中 unvoiced 段（沒紅線的地方）的深淺變化

### 4.4 如果想 **驗證泛化性**（不只訓練集）

- 動作：把 eval 從 val.txt 改成 test.txt（從未看過的 hold-out 歌手）
- 命令：
  ```bash
  python scripts/stage2_mel_eval.py --val-split .../splits/test.txt ...
  ```
- 主指標：所有指標跟 val 結果**接近** → 泛化良好；若 test 顯著糟 → over-fit on train

### 4.5 如果想 **修 Stage 1 端電音**（vocoder/F0 path bug）

目標：amateur 端 `_1_stage1_recon` 已有電音（即使 M4 端沒有）。M 之上的問題基本卡這個。

- 動作：先跑 vocoder identity test on amateur mel
- 主指標：
  - 對 amateur 樣本算 mel SSIM、F0 RMSE → 看是否跌出 Phase 0 gate
  - 對 amateur f0_a 跟 GT F0 的差距
- 工具：`scripts/vocoder_identity_test.py`、`scripts/sanity_vocoder_nsvb.py`
- 預期：若 SSIM 對 amateur 顯著低於對 pro（< 0.80），代表 Stage 1 FVAE 對 amateur 編解碼不穩定，可能要 fine-tune vocoder 於 dereverb'd amateur 或調 Stage 1 設定

### 4.6 如果想 **判斷哪個 step 是 best ckpt**

GAN 訓練沒有單一 val loss 可挑 best，必須綜合：

- `pro_direction_alignment` ↑ 最高的 step（拉 pro 最有效率）
- `unvoiced_concentration` 跟 `voiced_spectral_ratio` 仍 ✅
- `mel_l1_vs_recon` 不要過大
- 加上人耳聽測（用 `stage2_ckpts_listening.py` 產的 wav）

v2 經驗：mel 域指標顯示 **step 30000** 是 alignment 最高（+0.74）且其他指標仍健康，比末期 120000 更值得用。

---

## 5. 易混淆點 / 重要 caveat

### 5.1 latent 指標 vs mel 指標常衝突

- `delta_over_z` 0.87（latent）跟 `mel_l1_vs_recon` 0.46（mel） 不可直接比。
- latent metric 受 latent 維度 scale 影響大，**單獨用它判好壞會誤判**。
- 經驗：以 mel 指標為主、latent 指標只當訓中即時警報。

### 5.2 baseline（`_1_stage1_recon`）本身有 VAE 重建誤差

- `_1_stage1_recon` 不是「乾淨 GT」，它是「VAE 對 GT 的重建」，本身有微小誤差。
- 訓中 `unvoiced_concentration` 跟 mel 域 eval 的 `unvoiced_concentration` 數字可能略不同：訓中比 M 對 z_a 直接 decode 出的 baseline，eval 同樣公式但對 listening 時實際得到的 baseline 算。差異反映 baseline 採樣不同。

### 5.3 vocoder 電音 ≠ M 的問題

- amateur 端 `_1_stage1_recon` 就有電音 → 來自 Stage 1 encoder/decoder/F0 對 amateur 的處理，**M 沒參與**。
- M 之上的聽感判斷會被這個底噪遮蔽 → 我們改用 **mel-domain eval** 跳過 vocoder 看 M 的行為。
- 修這個 bug 是獨立議題（見 §4.5），不影響 Stage 2 訓練本身的判定。

### 5.4 `pro_dist_*` 只看 envelope（時間平均後）

- `pro_mean_env` 是把 pro 樣本沿時間軸取平均得到的 80 維向量。失去時間細節。
- 所以 `pro_dist_delta` 高 = envelope（音色重心）接近 pro。**不代表時間結構也像 pro**。
- 時間結構由 `voiced_spectral_ratio` + `temporal_diff_ratio_mel` 補。

### 5.5 M4 vs VV ratio 只在「val 含 M4」時有意義

- `stage2_mel_eval.py` 依 sample 選擇邏輯，會抽 1 個 M4 + 5 個 VV（共 6）。
- 如果未來改成只跑 amateur，ratio 段會消失。

### 5.6 數字本身的「正常範圍」是經驗值

- 健康閾值（uv_conc 0.55、vsr 0.70 等）是基於 v1 + v2 兩次訓練 + synthetic test 校準的經驗值。
- 不是論文驗證過的 absolute truth，未來訓不同 dataset 可能要重新校準。
- 質性比定量重要：「相對方向對嗎」比「數字落不落在 0.55」重要。

### 5.7 mel_l1_vs_recon ≠ content preservation

`mel_l1_vs_recon` 只代表「M 在 mel domain 改了多少」,**不能直接證明歌詞/咬字/旋律被保留**。
真正驗證 content 保留需要的指標(目前 backlog,未實作):

- **PPG similarity**: 對 GT mel 跟 output mel 各抽 PPG(Whisper hidden state),算 cosine sim;低 → 咬字 / 音素變了
- **ASR / lyrics CER**: 對 wav 跑 Whisper ASR,跟原始歌詞算 CER;高 → 歌詞被改寫
- **Phoneme posterior distance**: 對 PPG 序列做 phoneme-level alignment 差異
- **Duration / alignment consistency**: 用 DTW 對齊 GT / output,看 phoneme 邊界偏移

未來若要正式驗收「M 沒亂改歌詞」,該補這組指標,**不要把 `mel_l1_vs_recon` 低就當 content 保留**。
(目前 v2 的 `mel_l1_vs_recon ~0.5` 直觀上「M 修飾不大」,但邏輯上不能保證歌詞沒變。我們靠耳朵聽測補這個盲點。)

---

## 6. 想再深入的話

| 想了解 | 看哪份文件 |
|---|---|
| 訓練架構整體（Stage 1 / Stage 2 / Phase 3 推理） | [training_flow.md](training_flow.md) |
| 我們擔心的失敗模式 + 防線 | [../risk.md](../risk.md) |
| 訓中各 loss 的設計動機 | [training_flow.md §3.6](training_flow.md) + [../nsvb/task/stage2.py](../nsvb/task/stage2.py) 內 docstring |
| 修指標 / 健康閾值 | [scripts/stage2_mel_eval.py](../scripts/stage2_mel_eval.py) 內 `HEALTHY` dict 跟 `compute_metrics()` |