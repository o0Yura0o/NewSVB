# NSVB-ZH 重建改動清單（方案 A' 定案版）

本文件彙整 NSVB 重建為 NSVB-ZH（中文 unpaired 版本）過程中的全部架構決策。
配套文件：[risk.md](risk.md)、[scripts/draw_model_diagram.py](scripts/draw_model_diagram.py)、[figures/nsvb_zh_flow.png](figures/nsvb_zh_flow.png)。

---

## A. 從原論文**直接移除**的機制

| 項目 | 為什麼移除 |
|---|---|
| **a2p_f0_alignment**（EHSADTW 配對預處理） | Unpaired 資料下兩側沒有同首歌，DTW 無意義 |
| **L_map1**（paired KL on z） | 失去配對 → 沒有 z_p 可以逐 frame 比對 |
| **torch.gather 把 A warp 到 P 長度** | 訓練時 unpaired，根本沒有 P 長度可以 warp 過去 |
| **MFA forced alignment**（用於配對） | 訓練階段不需要；只在模式 C 推理選用 |
| **a2p mode 的 cross reconstruction path** | 原論文 a2p 路徑依賴配對，unpaired 無法走 |

## B. 從原論文**保留且不改**的機制

| 項目 | 為什麼保留 |
|---|---|
| **CVAE 架構**（encoder φ + decoder θ） | 核心骨幹，mel 處理能力與論文一致 |
| **Stage 1 CVAE 預訓練** | 論文的預訓練策略有效，直接復用 |
| **FastSpeech2 / fVAE backbone** | 語言無關、已驗證 |
| **HifiGAN vocoder ckpt（直接沿用 NSVB 作者提供的 1012_hifigan_all_songs_nsf）** | 訓練自 100+ 小時中英文混合歌聲，對中文歌已見過；省 ~2 週自訓 vocoder 時間。**注意：mel 參數必須對齊 ckpt config 的 hop=128 / fft=512 / win=512 / fmin=0 / fmax=8000**（不是 NSVB egs yaml 預設值；以 ckpt config 為準）。介面為 `(mel, F0) → wav`，因 generator config `use_pitch_embed: true` |
| **D_mel 的結構**（multi-window discriminator） | 結構本身沒問題，只改訓練資料組成 |

## C. **修改**原論文機制

| 項目 | 修改內容 | 為什麼 |
|---|---|---|
| **M（mapping 網路）結構** | 改為 residual kernel=1 conv，`f(z) = z + Δ(z)` | 確保 warp-invariant（`M(warp(z)) = warp(M(z))`），讓推理模式 B/C 可用；residual 降低 M 漂離初始 identity 的風險 |
| **D_mel 在 Stage 2 的 real 來源** | 只餵 x_p 當 real（原論文兩側都餵） | 原本 D_mel 是「自然度」判別器，改造後升級為「pro 自然度」，提供 mel 層向 pro 靠攏的推力 |
| **訓練抽樣方式** | x_a 與 x_p 完全獨立 batch，不配對 | Unpaired 資料的必然改動 |
| **資料集** | 英文 PopBuTFy → 中文 M4Singer + VocalVerse | 中文目標 |
| **PPG 提取器** | 英文 ASR → 中文 Whisper / WeNet | 中文目標 |
| **文字處理器 / 音素集** | CMU dict → 中文聲母韻母，`use_tone: true` | 中文目標 |

## D. **新增**（原論文沒有）

| 項目 | 作用 | 為什麼需要 |
|---|---|---|
| **D_z**（z 層判別器） | adversarial 監督 M 的 z 輸出往 pro 分布靠 | 失去 L_map1 後，這是唯一能在 z 層施加「向 pro 方向」的機制——D_mel 只管自然度，不管方向 |
| **Soft register bucketing**（5 buckets, Gaussian σ=0.3 log-Hz） | D_z 的 F0 條件 | 避免聲區錯置攻擊（低音被套頭腔共鳴），但又不引入 F0 shortcut |
| **Discrete phoneme ID**（argmax PPG） | D_z 的 content 條件 | 提供 D_z 判別 context，又不洩漏 PPG 信心度差異 |
| **L_PatchNCE** | M 的 content 保留主力 | 阻止 M 全域濾波（PatchNCE 鎖住 frame 對應），替代 L_identity |
| **L_PPG**（輸出端） | M 的 content 保留輔助 | 確保最終 mel 的 phoneme 語意不變 |
| **L_identity_pro**（20% batches，weight 0.1） | 防 M 在 pro 端飄移 | 從 ver1 借鑑：每 5 個 batch 隨機 1 個額外要求 `M(z_p) ≈ z_p`；保證 M 是「修不夠好的」而非「無腦套 pro 風格」，是 Risk 4 的二級保險 |
| **兩模式推理介面** | 滿足不同使用情境 | A=零門檻自動修飾；B=z_a' 透過 DTW warp 到 pro 模板的 T_p 時間軸（節奏+音高都拉過去） |
| **資料策展 JSD 檢查** | Phase 0 硬性前置 | 沒有此步驟，phoneme/register 分布差直接成為 D_z 捷徑 |
| **Soft bucketing 工具** | `nsvb/utils/soft_bucket.py` | 提供 F0 → 5-dim soft register 向量 |

## E. **資料前處理**（Phase 0 必做）

| 工作 | 為什麼 |
|---|---|
| 統一採樣率 / 響度正規化 / 去殘響 | 降低錄音環境 artifact 被 M 誤學為「pro 特徵」 |
| 提取管線：mel / F0(CREPE) / PPG(Whisper or WeNet) / spk_emb | 這些是訓練與推理共用的基本特徵 |
| 計算 **Phoneme 頻率 JSD**（M4Singer vs VocalVerse） | 驗證 < 0.05，不達標需重採樣 |
| 計算 **Register 頻率 JSD** | 同上，避免 D_z 用分布差當捷徑 |
| 歌曲策展（避免選曲品類過度差異） | 降低 phoneme / register 分布差的根因 |

## F. **Loss 組成**（Stage 2 訓練）

```
M 的更新（每 step）：
  L_M = 1.0 · L_adv_z       (D_z 的 adv 訊號 — hinge G loss)
      + 1.0 · L_PatchNCE    (z_a 與 f(z_a) 的 frame 對應鎖)
      + 0.5 · L_PPG         (輸出 mel 的 phoneme 一致)
      + 0.2 · L_adv_mel     (D_mel 的 adv 訊號，輔助)

M 的隨機附加項（identity_pro_prob=0.2）：
  + 0.1 · L_identity_pro    (M(z_p) ≈ z_p，L1 distance；20% batches 抽中)
        → 為什麼 stochastic 而非 always-on：always-on 會限制 M 在 pro 端的微調空間；
          隨機抽中讓 M 大部分時候只看 amateur，少部分時候被 pro 端「拉回 identity」

D_z 的更新（hinge D loss）：
  ReLU(1 - D_z(z_p, reg_p, ph_p)).mean() + ReLU(1 + D_z(f(z_a).detach(), reg_a, ph_a)).mean()

D_mel 的更新（hinge D loss，pro-only real）：
  ReLU(1 - D_mel(x_p)).mean() + ReLU(1 + D_mel(θ(f(z_a), ...).detach())).mean()

D_z warmup：
  前 5000 步 D_z 不傳梯度給 M（但 D_z 自己仍訓練），讓 M 在「弱 D_z」期間先學近恆等

學習率（TTUR, Heusel et al. 2017）：
  opt_M     = 1e-4
  opt_D_z   = 4e-4   (4x M LR；給 D_z headroom 學識別)
  opt_D_mel = 1e-5   (僅微調，從 Stage 1 沿用)
  Adam β = (0.5, 0.999)
```

## G. **訓練流程**（三階段，對應原論文兩階段 + 資料前置）

### Phase 0 — 資料前置
- 統一前處理 + 特徵提取 + JSD 驗證
- 產出 binarized dataset，D_z 需要的 register/phoneme 條件預先算好存進 sample

### Phase 1 — Stage 1 CVAE 預訓練
- 跟原論文 Stage 1 **幾乎一致**
- 同時用 M4Singer + VocalVerse 訓練 CVAE（兩邊都當重建目標）
- D_mel 此階段看兩邊都是 real（就是學「自然人聲」）
- 輸出：φ、θ、D_mel_stage1 三個權重

### Phase 2 — Stage 2 Mapping 訓練（unpaired，**主要改動集中在此**）
- 凍結 φ、θ
- 初始化 M（identity：`Δ(z)=0`）
- 初始化 D_z（from scratch）
- 復用 D_mel_stage1 繼續訓練（但 real 改為只有 x_p）
- 每個 step：
  1. 獨立抽樣 x_a (batch_a) 和 x_p (batch_p)
  2. 前向：z_a=φ(x_a), z_p=φ(x_p), f(z_a)=M(z_a), x̂=θ(f(z_a), c_a)
  3. 更新 D_z、D_mel（用 detached fakes）
  4. 重新前向 + 更新 M（L_M 四項組合）

### Phase 3 — 推理腳本 + DSP 整合
- 實作模式 A / B / C 三個進入點
- 整合既有 `singing_voice_beautifier_pipeline.py` 當模式 B 的 F0 前處理
- 搬原論文的 EHSADTW + gather 邏輯（僅模式 C）

## H. **兩個推理模式**（定案）

### 模式 A（預設）：純自動推理
- **輸入**：`x_a` 一個音檔
- **輸出長度**：T_a（與輸入業餘演唱同長度，可直接配原伴奏）
- **管線**：`x_a → φ → z_a → M → z_a' → θ(z_a', f0_a, spk_emb_a) → mel → vocoder`
- **F0 來源**：直接用 `x_a` 的測量 F0 作為 decoder 條件
- **用途**：零門檻、最常見使用情境,音色/技術/音準都交給 M 自動處理
- **代價**：無

### 模式 B（完全參考）：節奏與音高都拉到專業版
- **輸入**：`x_a`（業餘演唱主體）+ `x_p_ref`（同首歌的職業版參考，提供節奏 + 音高模板）
- **輸出長度**：**T_p**（跟隨專業參考的時長與節奏）
- **管線**：
  ```
  x_a       → φ → z_a → M → z_a'  (T_a)
  x_p_ref   → 抽 F0 → f0_p_ref     (T_p)
  z_a'      → DTW + gather warp 到 T_p 時間軸 → z_a'_warped (T_p)
  z_a'_warped + f0_p_ref + spk_emb_a → θ → mel(T_p) → vocoder
  ```
  - encoder 仍餵 `x_a` → 過 M 得到 `z_a'`（與 Mode A 共用前段，**不會切到 pro encoder**）
  - **核心處理**：在進 decoder 前，把 `z_a'` 用 DTW + `torch.gather`
                  warp 到 `f0_p_ref` 的 T_p 時間軸（沿用原論文 EHSADTW 機制），
                  讓 z 的 frame 結構與 pro F0 模板對齊
  - decoder 條件 `spk_emb_a` 來自 `x_a`：鎖定業餘歌手音色（防 timbre 外洩, Risk 4）
- **F0 來源**：直接使用 `x_p_ref` 的 F0（T_p）
- **用途**：使用者有同首歌專業參考時，套用 pro 的節奏 + 音高模板，
            但 z_a' 仍是 M 從業餘修出的、保留自己音色
- **代價**：
  - 推理延遲增加（DTW + gather + decoder 在 T_p 運作）
  - 輸出長度 ≠ 業餘演唱長度，**無法直接配原伴奏**——使用者必須改用 x_p_ref 的伴奏或對齊處理
  - 要求 `x_p_ref` 與 `x_a` 唱同首歌

### 模式建議與實作優先級
- **Phase 3 主力**：模式 A（所有使用者的預設體驗）
- **Phase 3 次要**：模式 B（搬原論文 EHSADTW + `torch.gather` warp 邏輯，約 80 行；
                              純推理路徑，不影響訓練）

## I. **資料策展硬性要求**（Phase 0 gate）

```
  M4Singer 與 VocalVerse:
  ├── Phoneme 頻率分布  JSD  <  0.05  (✅ 必達)
  ├── Register 頻率分布 JSD  <  0.05  (✅ 必達)
  ├── 採樣率統一        (✅ 必達)
  ├── 響度正規化        (✅ 必達)
  └── 殘響預處理        (🟡 建議)

  Vocoder identity test (HifiGAN 餵 GT mel 重建)：
  ├── mel SSIM        ≥  0.90  (✅ 必達)
  ├── F0 RMSE (Hz)    ≤  10    (✅ 必達)
  └── 任一 fail → 在 Phase 0 fine-tune vocoder 於中文歌聲後再進 Stage 1
```

> **為什麼 vocoder identity test 必須做**：
> NSVB 的 pretrained HifiGAN 是在英文歌聲（PopBuTFy）訓練的；若餵中文歌聲 mel
> 重建品質就已經斷裂，後續 M 的任何改進在聽測上都看不出來（因為 vocoder 本身
> 已成 bottleneck）。這個測試由 [`scripts/vocoder_identity_test.py`](scripts/vocoder_identity_test.py)
> 完成，不需要任何訓練即可跑。

## J'. **重建除錯守則**（從 NSVB vocoder bring-up 經驗淬鍊）

> 痛點來源：花了多輪迭代追 vocoder 電音問題，root cause 最後是 NSVB ckpt 旁的
> config.yaml 標 `fmin=0/fmax=8000`，但 inference 實際用的是 acoustic model
> yaml 的 `fmin=50/fmax=11025`。中間還誤判過架構（PWG vs HifiGAN-NSF）、
> mel log base、F0 抽法等。下面 6 條是事後總結，下次重建類似專案先讀。

### 守則 1：以 state_dict keys 為架構 ground truth，**不信檔名、不信 config**
- 範例陷阱：`1012_hifigan_all_songs_nsf` 裡寫 `generator_params: {layers: 30, residual_channels: 64, ...}`（PWG 招牌），但 state_dict 實際是 `m_source / noise_convs / resblocks / ups`（真 HifiGAN-NSF）
- 動作：**第一步先 `torch.load(ckpt, map_location='cpu')` 印 state_dict keys**，再判斷該移植哪個架構檔
- 副作用：strict load 是最強的相容性測試，pass 表示 forward pass 至少能跑

### 守則 2：追蹤 hparams **執行時實際值**，不是 config 檔上寫的
- 範例陷阱：vocoder ckpt 旁 config.yaml 標 fmin=0/fmax=8000，但 inference 在
  `set_hparams(acoustic_yaml)` 時被 acoustic model yaml 的 fmin=50/fmax=11025 覆蓋
- 動作：在 inference entrypoint 執行後 `print(hparams['fmin'], hparams['fmax'], ...)`，
  以這份「effective hparams」對齊，不要從 config 檔靜態讀
- 額外動作：寫 inline script 把 NSVB 的 base_config include 鏈完整跑一次再 dump，
  直接拿到對應 entrypoint 的最終 hparams

### 守則 3：**比數值，不比公式**
- 範例陷阱：我們驗證 `compute_mel` 公式 bit-exact 對齊 NSVB process_utterance，
  數值 max diff = 0.0；但因為 fmin/fmax 設錯，mel basis 矩陣本身就不同 →
  「同公式不同 fmin」 的 mel 統計分布天差地遠（mean -3.55 vs -3.55 看起來相同，
  但 frequency band 分配完全不同）
- 動作：對齊時除了公式，**還要 dump mel/F0 的 statistics**（min/max/mean/std/百分位）
  與 NSVB inference 真實值對比；不一致就找原因

### 守則 4：**早期跑 end-to-end sanity，不依賴單一指標**
- 範例陷阱：我們先 port 完 backbone、寫 vocoder identity test，跑出 SSIM=0.5 才發現問題
- 動作：在 port backbone 前先寫 `sanity_*.py`：用 NSVB 原 repo 自己的 class
  (`from modules.X import Y`) 跑一遍重建 → 確定基準 → 再對比
- 我們補寫的 `sanity_vocoder_nsvb.py` 才是真正釐清問題的關鍵 script

### 守則 5：**對音訊任務，盡早人耳聽**
- 範例陷阱：SSIM=0.5、F0 RMSE=13Hz 的數字看起來「marginal」，但實際聽是電音化，
  人耳跟 metric 完全脫節
- 動作：vocoder identity test / 任何重建腳本，**從第一版就加 `--save-wavs`**，
  metric 通過再聽 1 對是基本要求；metric 數字看起來「還可以」更應該聽

### 守則 6：**隔離冗餘子模組，只 port 真正用到的**
- 範例陷阱：NSVB 含 asr / glow / tts / 多個 task 變種；許多檔案命名誤導
  （vocoder 子目錄叫 parallel_wavegan，但 ckpt 是真 HifiGAN-NSF）
- 動作：以 **ckpt 載入路徑** 為起點往上追，按需 port；最後我們只 port 4 個檔
  （fvae / multi_window_disc / hifigan_nsf / source），其餘 NSVB 模組全沒碰
- 額外建議：每個 port 過來的檔案在開頭註解寫清「**為什麼移植 / 移除了什麼 / 為什麼不依賴 hparams 全域**」，避免將來自己也忘記

### 適用範圍
這 6 條對任何「重建他人 SVS / TTS 專案」流程都適用。下次遇到類似專案：
1. **先 ckpt → 再 config**（守則 1）
2. **印 effective hparams**（守則 2）
3. **dump 統計值對齊**（守則 3）
4. **end-to-end sanity 先行**（守則 4）
5. **每輪都聽**（守則 5）
6. **最小集合移植**（守則 6）

預估能把這次 ~6 hr 的 vocoder debug 縮短到 ~1 hr。

---

## J. **Linux 訓練機注意事項**（程式碼跨平台守則）

訓練最終會在 Linux GPU 機器跑；本地 Windows 開發時就遵守這些守則，避免移植時踩坑：

| 守則 | 為什麼 / 怎麼做 |
|---|---|
| **路徑全用 `pathlib.Path`** | 自動處理 `/` vs `\\`；不要手寫 `os.path.join` 或字串 `+ "/"` |
| **檔案 I/O 一律 `encoding="utf-8"`** | Windows 預設 cp950，Linux 預設 utf-8；明寫避免 unicode 錯誤 |
| **不要用 `os.system()` / `subprocess` 跑 bash 內建命令** | `ls`、`cp`、`rm` 跨平台不一致；用 `pathlib` / `shutil` 取代 |
| **multiprocessing 用 `spawn` 而非預設 fork** | Linux 預設 fork（CUDA fork-safety 差），Windows 只支援 spawn；`mp.set_start_method("spawn", force=True)` 兩邊一致 |
| **HF 模型 cache 路徑用環境變數** | Linux: `~/.cache/huggingface`; 不要 hardcode；用 `HF_HOME=/path/to/cache` 控制 |
| **CUDA device 字串只用 `"cuda"` 不用 `"cuda:0"`** | 多卡訓練時讓 framework 自動分配；單卡也照樣可跑 |
| **`torch.set_num_threads(1)` 只在 multiprocessing worker 設** | Linux fork 後 BLAS 的 thread 會炸開；單一進程則不要設 |
| **Whisper 載入時警告 symlink 是 Windows 限定** | Linux 不會出現，可忽略；不需要寫 try/except 處理 |
| **不要 hardcode `C:\` / `/c/Users/...`** | 全部從 CLI args 或環境變數取；測試命令使用相對路徑 |
| **不依賴 Windows-specific PowerShell 工具** | smoke test、benchmark 都寫成 Python，能在兩邊跑 |

---

## 接下來要做的事（依序）

### 1. Phase 0 前置（估計 1~2 週）
- 下載 M4Singer + VocalVerse，純人聲版本
- 實作提取管線（CREPE、Whisper/WeNet、speaker encoder）
- 實作 `nsvb/utils/soft_bucket.py`
- 跑 JSD 檢查 script，不達標則做 phoneme/register 重採樣
- Binarize 資料集，register/phoneme_id 預先存入每個 sample
- **跑 vocoder identity test**（`scripts/vocoder_identity_test.py`）
  - 檢查 pretrained HifiGAN 餵中文歌聲 mel 重建是否 ≥ 0.90 SSIM, ≤ 10 Hz F0 RMSE
  - 不過則 fine-tune vocoder 於中文歌聲後再進 Stage 1

### 2. Phase 1 Stage 1 預訓練（估計 2~3 週 on GPU）
- 設定 `configs/singing_zh_stage1.yaml`
- 訓練 CVAE + D_mel，直到 a2a/p2p 重建品質穩定
- 檢查 Monitor 2（z_p / z_a 分布是否重疊）

### 3. Phase 2 Stage 2 訓練（估計 2~3 週 on GPU，**最關鍵階段**）
- 設定 `configs/singing_zh_stage2.yaml`
- 凍結 φ/θ，初始化 M 為 identity
- 持續監控 D_z accuracy（Monitor 3）
- 不穩定時調 LR 比例或權重

### 4. Phase 3 推理整合（估計 1 週）
- 實作模式 A 腳本
- 加模式 B（DTW warp F0 到 T_a）
- 加模式 C（搬 EHSADTW + gather）
- 整合 `singing_voice_beautifier_pipeline.py` 的 DSP 音準修正

### 5. 聽測驗證（Monitor 4、5）
- 抽樣人工聽測，確認無 Mode Collapse、無殘響 artifact
