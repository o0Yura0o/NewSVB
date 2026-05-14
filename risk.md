# NSVB-ZH 風險清單（方案 A' 定案後）

## 已透過架構消除的風險

### Risk 1：條件分桶的顆粒度陷阱（已消除）
- **原描述**：hard quintile bucket 會把 vibrato/portamento 等微音高動態平均掉
- **消除方式**：改用 soft Gaussian bucketing（σ=0.3 log-Hz）+ bucket 數降至 5（register 級，非 semitone 級）
- 微音高動態完全保留在 z 裡，bucket 只提供粗粒度聲區導引

### Risk 2：音質域與技術域混淆（已加 3 層補強）
- **原描述**：M4Singer (錄音室) vs VocalVerse (user-generated) 的錄音環境差異可能讓 M
  學成「降噪/去殘響濾波器」而非「修技術」
- **緩解方式（5 層）**：
  1. **採樣率統一 + 響度正規化（必開）**：`librosa.load(sr=22050)` + pyloudnorm BS.1770
     -22 LUFS（`audio_io.loudness_normalize`）
  2. **DeepFilterNet3 dereverb + denoise（P0，必開）** ⭐：
     對兩 dataset 都做（不只 amateur 端，避免引入新差異）；
     `audio_io.dereverb_wav()` + `binarizer.py --no-dereverb` 預設 OFF（即啟用 dereverb）。
     Risk 2 主防線——把音質基線拉齊到「乾淨」狀態，逼 M 學真正的技術差
  3. **D_z 在 z 層工作**：錄音環境差異在 Stage 1 CVAE 時已被 mel 層吸收
  4. **訓中 audio quality monitor（P2，建議）**：
     `Stage2Trainer.monitor_audio_quality` 每 5000 步抽 N 個 amateur 樣本，計算
     `Δ_unvoiced_E / (Δ_voiced_E + Δ_unvoiced_E)` ratio：
       - < 0.55 良好（M 修飾集中在 voiced 段，是真技術修正）
       - 0.55–0.65 marginal（觀察）
       - > 0.65 **Risk 2 警訊**（M 修飾集中在 unvoiced/silent 段，可能在去殘響）
  5. **Phase 0 audio quality probe（P1，必跑）** ⭐：
     `scripts/audio_quality_probe.py` 計算兩 dataset 的 4 個 metric 分布 + JSD。
     **4 metric 可信度不等**（實測 dereverb response 後分類）：
       - **Reliable**（直接量頻譜，DF3 影響可預測）：`sfm`、`hf_ratio`
       - **Heuristic**（量測方法限制）：`snr_db`（voiced_E/unvoiced_E 比，VV 背景噪音
         saturate）、`reverb_sec`（能量衰減估算，DF3 改 transient 形狀後失準）
     形式 FAIL 不等於 mitigation 失效——**只看 reliable metric**：sfm + hf_ratio
     改善 / PASS 即視為實質通過。真正 ground truth 是 L5 monitor
     `unvoiced_concentration`，比 raw-wav heuristic 公允得多。
     用 `--apply-dereverb` 可跑 dereverb 後變體，直接驗證 mitigation 縮小頻譜差距。

### Risk 3：Anchor MSE 過於剛性（已消除）
- **原描述**：frame-by-frame MSE 會強烈懲罰專業歌手的時間軸微調
- **消除方式**：放棄 MSE-based anchor，改用 L_PatchNCE（frame-wise contrastive）
  - PatchNCE 只要求「對應 frame 更像自己、不像別 frame」，不要求絕對幅值
  - 允許 vibrato、attack/release、共鳴變化等技術層改動

### Risk 4：Encoder 解耦不完美導致音色外洩（已消除）
- **原描述**：z_a 殘留業餘音色，M 推向 pro 分布時可能改變音色
- **消除方式（三層保險）**：
  1. **結構性錨定**：Decoder 有明確 `spk_emb` 條件分支；推理時餵 `spk_emb = spk_a`，
     即便 z 中音色資訊不完全解耦，decoder 會以 spk_a 為準
  2. **Speaker embedding 來源獨立於 z**：用 Resemblyzer 對 raw audio 抽 256-dim
     L2-normed 向量，不經 encoder φ；保證 z 走多遠都不影響 spk_emb 條件
  3. **L_identity_pro 防 M 漂移（從 ver1 借鑑）**：每 5 個 batch 隨機 1 個額外
     要求 `M(z_p) ≈ z_p`（L1 distance, weight 0.1）；保證 M 在 pro 端不會自由發揮，
     輸入 pro z 時近恆等。即便 amateur z 中殘留音色被 M 推到 pro 區，pro 區的 M
     已被訓練成「不會主動加東西」

### Risk 5：條件對齊 Mode Collapse（已消除）
- **原描述**：D_z 用 conditional MMD 時，所有同 bucket 樣本被映射到「平均 pro 狀態」，失去多樣性
- **消除方式**：
  - 放棄 MMD，改用 adversarial D_z（學高階分布而非均值變異數）
  - PatchNCE 強制保留個體 frame 指紋
  - Decoder 的 spk_emb 條件額外保留說話者個性

### Risk 6：F0 shortcut（已消除）
- **原描述**：若 D_z 直接接收連續 F0 條件，amateur F0（走音）與 pro F0（準）分布不同，D_z 用 F0 品質當捷徑判別，M 收不到有效梯度
- **消除方式**：
  - D_z 的 F0 條件改為 **5 個 register bucket**（資訊量 ~2.3 bit）
  - bucket 邊界遠大於半音間距，走音半音不跨 bucket
  - Soft bucketing 讓邊界處 smooth 過渡

### Risk 7：跨歌條件洩漏（已消除）
- **原描述**：若用 reference-based 訓練（F0 from 不同首歌），條件內部不一致會被 D_z 抓到
- **消除方式**：完全放棄 reference-based 訓練，改用純 unpaired 抽樣

### Risk 8：全域共鳴濾波攻擊（已消除）
- **原描述**：純無條件 D_z 下，M 可能把所有業餘樣本套上「最高分的 pro 技術指紋」（例如把低音也加頭腔共鳴）
- **消除方式**：D_z 的 register bucket 條件強制聲區對齊——低音區必須展現胸聲特徵、高音區必須展現頭聲特徵

### Risk 9：Amateur dataset 含「near-pro」雜訊樣本（已消除）
- **原描述**：VocalVerse 929 筆中含「near-pro」樣本（pro 4-dim 評分接近 pro 水準）。若全收當 amateur side：
  - D_z 看 amateur batch 內混有 z 分布近 pro 的樣本 → 矛盾訊號 → 梯度方向不穩
  - PatchNCE / L_id_pro 對 high-score amateur 樣本平均效應 → M 「需要修飾」的訓練壓力降低 → ‖Δ‖/‖z‖ 偏低、修飾不足
  - 推理時 user 拿真正差的業餘輸入，M 訓練分布偏中等 → 修不到位
- **消除方式**：[`nsvb/data/vocalverse_mos.py`](nsvb/data/vocalverse_mos.py) + binarizer `--vocalverse-amateur-score-max 3.0`
  - 利用 [VocalVerse 作者論文 arXiv:2512.06999](https://arxiv.org/abs/2512.06999) 提供的 **pro 4-dim 標記**（音色/情感/技巧/氣息控制）
  - 主推 `amateur_score = (技巧 + 氣息控制) / 2 ≤ 3.0`：
    - 只取與 vocal mechanics 相關的兩個維度（M 訓練的目標）
    - **不**用音色（physiological 特性，由 spk_emb 鎖；論文 §3.1.5 與 Sundberg 1974 文獻）
    - **不**單純依賴 amateur MOS（與 pro 標記只 0.38 spearman 相關，可作次要 corroborator）
  - 結果：留 536/929 筆 (~29.8 h)，與 M4Singer pro side (~30 h) 對齊
  - 33 個 singer 全保留，per-singer 中位 17 樣本（min ~12），speaker diversity 不受損
  - kept amateur_score 範圍 [1.0, 3.0]，mean 2.40，明確「技術偏弱」區間

## 持續監控項

### Monitor 1：資料策展 JSD 指標
- Phoneme 頻率 JSD(M4Singer, VocalVerse) < 0.05
- Register 頻率 JSD(M4Singer, VocalVerse) < 0.05
- **未達標後果**：D_z 會用 phoneme/register 分布差當捷徑
- **偵測方式**：Phase 0 前處理 script 自動計算並報告

### Monitor 1b：Vocoder identity test（Phase 0 必跑）
- pretrained HifiGAN 餵中文歌聲 GT mel 重建
- mel SSIM ≥ 0.90、F0 RMSE ≤ 10 Hz
- **未達標後果**：vocoder 本身已是 bottleneck，後續 M 任何改進在聽測上都隱形
- **偵測方式**：[`scripts/vocoder_identity_test.py`](scripts/vocoder_identity_test.py)
  在 Phase 0 自動跑，不過則 fine-tune vocoder 於中文歌聲後再進 Stage 1

### Monitor 2：Stage 1 CVAE 解耦品質
- z_p 和 z_a 的分布在 Stage 1 訓練完後應該**高度重疊**（因為 F0/PPG/spk 都已條件化出去）
- **偵測方式**：t-SNE z_p vs z_a；若明顯分離代表 decoder 條件化不足，需調 KL 係數

### Monitor 3：D_z / M 對抗平衡
- D_z accuracy 應穩定在 0.55~0.75（太高代表 M 學不贏、太低代表 M 在走捷徑）
- **偵測方式**：訓練 log 即時監控，不平衡時調整 LR 比例

### Monitor 4：Mode Collapse（降級監控）
- 即便有 PatchNCE + spk_emb，仍需定期抽樣驗證輸出多樣性
- **偵測方式**：隨機抽 10 個 amateur 樣本，用同一 spk_emb 推理，人耳聽是否有「同一個 pro 配音」感

### Monitor 5：錄音環境 artifact
- M4Singer 可能有錄音室殘響，VocalVerse 可能較乾
- 推理時 M 可能意外引入殘響當「pro 特徵」
- **偵測方式**：推理 + 聽測，有殘響異常則考慮在 Stage 1 加做 dereverb

## 已知取捨項（設計決策，非風險）

### Tradeoff 1：放棄 paired 帶來的強監督
- L_map1（paired KL on z）是最強的 z 層監督，我們用 L_adv_z 替代
- **代價**：adversarial 訓練比 KL 不穩、收斂慢
- **補償**：TTUR + spectral normalization + PatchNCE 作為正則

### Tradeoff 2：兩種推理模式的複雜度差異

本專案在推理端支援兩種模式，使用者依情境選擇（詳細管線見
[rebuild_checklist.md §H](rebuild_checklist.md)）：

#### 模式 A（預設）：**純自動推理**
- **輸入**：`x_a` 一個音檔
- **輸出長度**：T_a（與輸入業餘演唱同長度，可直接配原伴奏）
- **F0 來源**：直接用 `x_a` 的測量 F0 作為 decoder 條件
- **用途**：零門檻、最常見使用情境，音色/技術/音準都交給 M 自動處理
- **代價**：無

#### 模式 B（完全參考）：**節奏與音高都拉到專業版**
- **輸入**：`x_a`（業餘演唱主體）+ `x_p_ref`（同首歌專業版，提供節奏 + 音高模板）
- **輸出長度**：**T_p**（跟隨專業參考的時長與節奏）
- **管線重點**：
  - encoder 仍餵 `x_a` → 過 M 得到 `z_a'`（與 Mode A 共用前段）
  - 進 decoder 前對 `z_a'` 做 DTW + `torch.gather` warp 到 T_p 時間軸
  - decoder 條件：`f0_p_ref` (T_p) + `spk_emb_a`（保留業餘音色，Risk 4 防護）
- **代價**：
  - 推理延遲增加（DTW + gather + decoder 在 T_p 運作）
  - 輸出長度 ≠ 業餘演唱長度，**無法直接配原伴奏**——使用者需改用 x_p_ref 伴奏
  - 要求 `x_p_ref` 與 `x_a` 唱同首歌

#### 模式建議與實作優先級
- **Phase 3 主力**：模式 A（所有使用者的預設體驗）
- **Phase 3 次要**：模式 B（搬原論文 EHSADTW + `torch.gather` 邏輯約 80 行；
                              純推理路徑，不影響訓練）

---

# Potential risks of Implemented code

> **狀態圖例**：
> - ✅ 已修正（程式碼有對應改動）
> - ⚠️ 部分緩解（加 fallback / 警告，非預設改動）
> - ℹ️ 評估後不需改（風險評估誤判或現有架構已防）

---

### 一、 核心架構的物理衝突（高風險）

#### 1. M 網路 kernel_size 與時間軌跡保存 — ⚠️ 部分緩解
- **原描述**：kernel_size=1 是 pointwise MLP，無時間感受野；amateur z 若為 flat，無法生成顫音/滑音
- **判讀重點轉移**：真正的風險不是「kernel=1 能不能表示顫音」（z_a 已含顫音 trajectory），而是「**M 是否動到正確的地方**」——即保留 trajectory + 只改 spectral envelope。kernel 大小只是手段，不是終點。
- **兩種真實 failure mode**（已實作分別偵測）：
  - **Mode A：M 太保守**（pointwise 表達力不足）
    - 偵測：`delta_over_z` 移動平均 < 0.03 持續到 step 30000
    - 行動：切 `--m-kernel-size 3` 重訓
  - **Mode B：M 抹平既有 trajectory**（高 Δ/z 但破壞時間結構）
    - 偵測：`temporal_diff_ratio = mean|Δ_t M(z) − Δ_t z| / mean|Δ_t z|` 移動平均 > 1.0
    - 解釋：z 自身的時間導數有特定 magnitude；M(z) − z 的時間導數差距若超過 z 自身 → M 把時間結構整個重寫（顫音/滑音被殺或加上不自然抖動）
    - 行動：降 `lambda_adv_z` 或提早停訓；用 monitor npz 視覺化 delta_mel 確認
- **已實作緩解**（[Stage2Trainer](nsvb/task/stage2.py)）：
  - 每 step 計算 `delta_over_z` + `temporal_diff_ratio`，pbar 即時顯示
  - 30000 步後分別檢查兩條件 → 一次性警告（互斥但可同時觸發）
  - 兩個閾值都暴露在 [Stage2Config](nsvb/task/stage2.py)：`delta_low_threshold=0.03` / `temporal_diff_high_threshold=1.0`
  - 每 5000 步 `monitor_audio_quality` 補 **`voiced_spectral_ratio`**：把 voiced 段 Δ_mel 拆成「envelope shift」與「F0 trajectory」兩個時間頻譜分量，≥ 0.7 envelope-dominated 健康；< 0.4 警訊（M 改到 F0 trajectory）
- **保守選擇**：直接用 `--m-kernel-size 3`，warp-invariance 在 3-frame context (~70 ms) 下只是輕微違反，遠小於音素時長

#### 2. PPG 強制 Resample 50 → 172 fps 子音邊界模糊 — ℹ️ 評估後不需改
- **原描述**：50 fps 線性插值放大 3.45× 到 172.27 fps，子音 (~20 ms) 邊界被塗抹
- **判讀錯誤點**：Whisper hidden state 不是 one-hot phoneme，是已被 attention 平滑過的高維連續表徵；20ms 子音早已被 Whisper 內部攤平到周圍幾個 50fps frame 中。在已平滑的連續向量空間做線性 interp 幾乎等於 ground truth
- **架構保護層已存在**：
  - D_z 條件用 [phoneme_id](nsvb/data/cluster_ppg.py)（k-means argmax discrete，**不**經 interp）
  - Continuous PPG 只進 decoder（WN kernel=5 大感受野，吸收 interp 平滑）
  - Nearest 反而會引入 step-function 跳變對下游梯度更不友善
- **行動**：不改程式碼。若 Phase 2 試聽**確認**有咬字模糊，再做 ablation（`mode="nearest"` + 1D smoothing conv）

#### 2b. Whisper PPG 對 singing 的 pitch 污染 — ⚠️ 部分緩解（Phase 0 gate 偵測）
- **原描述**：Whisper 是 speech-heavy 模型；唱歌的延音、vibrato、melisma、非語音段可能讓 hidden state 混入 pitch / prosody 資訊
- **判讀**：layer 8 雖近 phonetic 但沒完全把 prosody 抽掉。對 speech 是小量、對 singing（pitch 是主要變異）會被 k-means 放大成「同一母音不同 pitch → 不同 cluster」。最終 failure mode：`phoneme_id` 與 `register_id` 強相關 → D_z 的兩個 condition 實質塌成一個 → **隱形版 Risk 6 F0 shortcut**
- **架構部分保護**：
  - Decoder 取顯式 F0 + spk_emb（dominant pitch signal 不靠 PPG）
  - D_z 用 discrete cluster id 而非連續 PPG（有限分桶）
  - Gate ③ JSD 已檢查跨 dataset 分布，但**看不到同 dataset 內部** cluster ↔ register 相關性
- **已實作緩解**：
  - 新增 **Gate ④** [`scripts/cluster_ppg_inspect.py`](scripts/cluster_ppg_inspect.py)：抽樣計算 `MI(phoneme_id; register_id)` 與 voiced 段 cluster dwell length；同時畫時間序列疊圖供眼睛交叉檢查 sustained / vibrato 段 cluster 是否穩定
  - 健康閾值：`MI < 0.3 bit`、`mean dwell ≥ 8 frames`
- **WARNING 補救順序**：降 K → 換 Whisper layer (6 或 10) → PPG per-utterance 去 DC → phoneme_id mode filter

---

### 二、 訓練動態與 Loss 設計（中風險）

#### 1. 隨機觸發的 L_id_pro 摧毀優化器動量 — ℹ️ 評估後不需改
- **原描述**：每 5 步丟進一次量級不同的 L_id_pro 打亂 Adam 動量
- **判讀**：理論方向對但量級錯：
  - `lambda × E[l_id_pro] ≈ 0.1 × 0.01·||z|| ≈ 0.001·||z||`，比主 loss (l_nce/l_adv_z ~0.5–2) 小 1–2 數量級
  - opt_m 用 Adam β1=0.5（GAN 慣例），動量半衰期 ~1 步即輕量 — β1=0.9 才是用戶原描述適用的「重動量」場景
- **設計意圖**：「100% × 0.02」不等價，固定觸發會把 M 訓成「總是試圖 identity-on-pro」（pro 聲也有些瑕疵應允許微調）
- **狀態**：尚未經實機驗證（NSVB-ZH ver1 配方繼承自另一 AI 草稿，未跑過實際訓練）
- **已實作緩解**：[Stage2Trainer.train_step](nsvb/task/stage2.py) `# 3d. L_identity_pro` 區塊加說明 comment + 失敗時的 fallback 提示（改 100% × 0.02）
- **觀察條件**：訓中若 m_total 出現「每 5 步一個 spike」週期性震盪 → 改為固定權重

#### 2. D_mel 領域偏移（Stage 2 real = pro only）→ 災難性遺忘 → 鼓勵去殘響 — ⚠️ 部分緩解
- **原描述**：Stage 2 D_mel 只看 pro，會把「乾淨錄音室聲」當唯一 real → M 被次級壓力推向去殘響
- **判讀**：擔憂的 mechanism 真實存在，但用戶提的 mitigation（ConcatDataset 混合 real）會破壞 [rebuild_checklist §C](rebuild_checklist.md) 設計（D_mel 升級為 pro-direction 推力）
- **架構主防線已存在**：
  - **Risk 2 L2** dereverb 對兩 dataset 都做 → D_mel 沒有「殘響=amateur 簽名」捷徑
  - λ_adv_mel = 0.2 << λ_adv_z = 1.0：D_mel 是輔助
  - L_PatchNCE 鎖 z-frame 對應，過激去殘響會被懲罰
  - **Risk 2 L5** monitor (unvoiced_concentration) 直接抓此 failure
- **已實作緩解**：[Stage2Config.dmel_mix_amateur_real](nsvb/task/stage2.py) + CLI `--dmel-mix-amateur-real`（預設 off）；訓中 monitor 顯示 `unvoiced_concentration > 0.65` 連續兩次時 resume 啟用作 fallback 救火

#### 3. L_adv_mel 用業餘 F0 conditioning → M 被推去補償它管不到的 F0 痕跡 — ⚠️ 部分緩解
- **原描述**：[`_decode_with_mapped_z`](nsvb/task/stage2.py) 解碼 `M(z_a)` 時，condition `g` 由業餘端
  `(ppg, f0, spk_emb)` 建，`l_adv_mel` 對這個帶業餘 F0 的 mel 算對抗 loss。疑慮：合成 mel 與
  L_adv_mel 會不會錯誤獎懲 M。
- **判讀（梯度鏈是乾淨的）**：`g` 對 M 而言是常數，不在 M 的優化變數裡；`∂l_adv_mel/∂M` 只經
  θ 對 z 輸入的 Jacobian（固定 g 下求值）回傳。**M 不會因為 F0 本身收到梯度** — F0 通道完全
  不傳梯度給 M。`mel_baseline = θ(z_a, g)` 與 `mel_g = θ(M(z_a), g)` 共用同一 `g`，唯一差別
  是 `z_a` vs `M(z_a)`。
- **真實的 confound（訊號來源層面）**：`mel_g` 是「業餘 F0 ⊗ M 的 z 編輯」的聯合產物；D_mel 的
  real 是純 pro mel（pro F0 生成）。D_mel 判 `mel_g` 為假時無法歸因是「z 音色不夠 pro」還是
  「F0 軌跡是業餘的」。若扣分有一部分來自業餘 F0 痕跡，M 唯一槓桿是改 z → **可能被推去扭曲 z
  來遮掩它根本修不了的 F0 缺陷**。亦即 D_mel 的 domain gap 有個 M 關不掉的非零地板。
- **為什麼不是 bug**：Mode A 推理本來就用業餘 F0（保留業餘表情、只修音色），訓練用業餘 F0
  conditioning 與部署一致，無 train/test mismatch；且 unpaired 資料根本沒有配對 pro F0 可用。
- **架構主防線已存在**：
  - λ_adv_mel = 0.2 << λ_adv_z = 1.0 = λ_patchnce：M 主訊號是 z 空間、F0 無關（L_adv_z、
    PatchNCE），L_adv_mel 只是輕推 → confound 被降權 5×
  - `monitor_audio_quality` 的 `voiced_spectral_ratio < 0.4` 直接抓「M 在改 F0 trajectory」
  - Failure mode B (`temporal_diff_ratio > 1.0`) 抓「M 重寫時間結構」
  - VocalVerse 已 MOS 篩到 `amateur_score ≤ 3.0`（技巧+氣息），篩掉的是音色/技巧業餘而非音準
    大歪的樣本 → 業餘 F0 軌跡不致誇張到讓 D_mel 主要靠它判別 → confound 實務上估為二階小量
- **已實作緩解**：與 #2 共用 [`--dmel-mix-amateur-real`](nsvb/task/stage2.py) fallback——啟用後
  D_mel real 混入 amateur，消掉「永遠關不掉的 F0 domain gap」。觸發條件同 #2（monitor 連兩次
  異常時 resume 啟用）。
- **觀察條件**：訓中 `voiced_spectral_ratio` 持續 < 0.4 或 `temporal_diff_ratio` > 1.0 →
  confound 正在發作。

---

### 三、 邊界條件與小細節

#### PatchNCE 取樣越界崩潰 — ℹ️ 描述錯誤，現有程式碼已防
- **原描述**：T_z=86 但 num_patches=128，`random.sample` 會 crash
- **判讀**：[losses.py PatchNCELoss._sample_indices](nsvb/model/losses.py) 早就有 `N = min(self.num_patches, T)` clamp，且用 `torch.randint`（with replacement）非 `random.sample`。永遠不會 OOB
- **唯一真風險**：T < 4 時 contrastive 訊號極弱（每 query 至多看到 3 個 negative）
- **已實作緩解**：T < 4 時 PatchNCE 發 RuntimeWarning（一次性，不 raise），建議調大 max_frames

#### 高音女歌手 F0 截斷 — ✅ 已修正
- **原描述**：fmax=1100 Hz 截掉 D6 (1175 Hz) 等流行女聲高音
- **判讀**：完全正確；CREPE 在 F0 > fmax 時通常給 octave 錯誤（D6→D5），給 D_z 一個錯誤 register 條件，比直接 unvoiced 還糟
- **已修正**：[audio_config.F0_FMAX](nsvb/utils/audio_config.py) 1100 → **1400** Hz（覆蓋到 F6=1397 Hz）
- **重要：改完要重 binarize**（已 binarized 的 .npz 是用舊 fmax 抽的不會自動更新）