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
     `scripts/audio_quality_probe.py` 計算兩 dataset 的 SFM / Reverb / HF-ratio / SNR
     分布並算 JSD，若任一 < 0.10 才 PASS，否則需加強前處理

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
