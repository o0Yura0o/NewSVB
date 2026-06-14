# v2 訓練後釐清之議題與後續驗證實驗

> **狀態**:觀察 + 部分分析階段。本文件以「**本研究方法之 honest scrutiny + validation pathway**」為基本立場;前處理(DF3 + LUFS norm)為消除資料集域混淆之**核心方法**,**不可拿掉**,本文件之實驗清單均在此前提下設計。
> **時間軸**:2026-05-19 至 2026-06-13 之多次 post-hoc 分析
> **影響範圍**:v2 之 `pro_direction_alignment +0.84` 結論之解讀;PopBuTFy 訓練之 ROI 評估;論文 §3.4.2 / §4.6 之撰寫
> **本文件目的**:把觀察集中記錄,並列出在「不違反方法前提」下能進行之驗證實驗

---

## 一、方法前提之重述

本研究主張下列方法論設計:

1. **DF3 去殘響 + BS.1770 響度正規化(目標 -22 LUFS)** 為前處理之主要環節
2. 此前處理之**設計目的**:把業餘端(VocalVerse,有殘響/噪音/響度差異)與專業端(M4Singer,錄音室乾錄)之**環境差異拉到可比較水位**,使下游 Stage 1 CVAE + Stage 2 M 能聚焦於「技巧層級」之映射,而非被「環境差異」之 shortcut 帶偏(對應§3.5 第①項風險之主要緩解策略)
3. 前處理屬於**方法的一部分**,不可移除;若違反此前提之 ablation(例如「不做 DF3」)等於拆掉方法之基石

本前提下,任何「拿掉前處理看 M 還能不能 work」之實驗都**不該做**;但「在保留前處理之前提下,證明 M 之 alignment 主要反映技巧改善而非單純之指紋補償」之驗證**必須做**。

---

## 二、主要觀察

### 2.1 評估方法論之修正(2026-05-19)

v2 時期使用之 `scripts/stage2_mel_eval.py` 舊版本將 M4 + VV 混合平均算 verdict,造成:
- 業餘端真實 `pro_direction_alignment +0.84` 被稀釋為混合平均之 +0.60
- 連帶造成「測試集相對驗證集呈現 ~19% 泛化退步」之假性結論
- `hf_extra` 之混合平均 -0.18 實際為 VV-only -0.29(marginal)

修正後使用 VV-only 主表 + M4 control 分表,並提供 `scripts/stage2_mel_eval_rerender.py` 從現有 CSV 重生新版 report。此一修正已於§4.6.2「雙軌結構於泛化誤判風險之效應」呈現於論文。

### 2.2 z-space 分離主因:env 軸主導(z-probe v3,2026-05-20)

`scripts/stage1_zprobe.py` v3 對 Stage 1 latent z 做 domain-axis 變異分解:

| 成因軸 | 變異佔比 | 細節 |
|---|---|---|
| 環境/風格 env(dereverb 後仍存) | ~57% | 主要由 sfm 差 d=0.98 主導;hf_ratio 已對齊(d=-0.26) |
| 未知(可能含 codec、F0 範圍、唱法) | ~43% | env 殘差化後 d 從 8.24 降至 3.53 |
| 技巧軸 | R²≈0.17,但與 domain axis cos=0.165(近正交) | 技巧有被 z 編碼,惟其方向**與分離軸近正交** |

**核心洞察**:D_z 之判別訊號(分 M4 vs VV)瞄準 env/風格軸,**不是**技巧軸。M 被獎勵把 z_a 推向 env/風格軸,技巧雖在 z 裡卻沒被訓練訊號充分利用。

### 2.3 Stage 1 重建之 marginal-but-functional 狀態(stage1_audit,2026-05-20)

- SSIM 0.75-0.79(borrow vocoder identity 門檻偏嚴)
- L1 0.21(marginal)
- KL ~90(偏高,posterior 未壓緊,但可能為「給技巧差留空間」之代價)
- t-SNE M4 vs VV 完全不重疊(為**設計成功**之證據,非失敗 — z 抓到業餘/專業差;見 [[feedback_stage1_zspace_interpretation]])
- 共鳴 formant 區(mel bins 16-22)per-bin MSE 0.13-0.14(業餘端聽感「軟」可能跟此有關)

判讀:Stage 1 marginal-but-functional,**沒有確鑿證據為 v3 瓶頸**。

---

## 三、方法之非對稱效應:必要代價之 honest disclosure

本研究承認前處理(DF3 + LUFS norm)對業餘端與專業端**影響非對稱**。此為「對齊環境分布」目標之必然代價,**不是 bug**;但需於論文中誠實揭露,並透過獨立驗證確認下游 M 之 alignment 主要反映技巧改善。

### 3.1 DF3 對 VV 之大改 vs 對 M4 之 no-op

於同一 VocalVerse chunk(`446255-351391558-c009`)上之三變體實驗(`scripts/compare_df3_vocoder.py`,2026-06-12):

| 指標 | A:無 DF3 | B:有 DF3(=production npz) |
|---|---:|---:|
| mel min | -6.00 | -9.17 |
| mel L1(A vs B) | — | 0.41 |
| mel HF L1 (band ≥50) | — | 0.66 |
| f0 RMSE(both voiced) | — | 3.3 Hz |

聽測:無 DF3 版之 vocoder 重建電音明顯較輕;DF3 版有顯著金屬感與電音失真。**M4 端未實測但預期 DF3 幾乎 no-op**(錄音室乾錄,無殘響/噪音可移除)— 待跑實驗 ① 量化。

### 3.2 LUFS norm 同類效應

[`audio_io.py`](../nsvb/utils/audio_io.py) 之 docstring 原本誤稱「-22 LUFS 對齊 NSVB vocoder 訓練分布」,但實際確認 NSVB vocoder ckpt `1012_hifigan_all_songs_nsf` 之 config 明確設 `loud_norm: false`(即:NSVB 訓練 vocoder 時並未做 LUFS norm,只是 `process_utterance` 函式內部 if-branch 之 hardcoded 值是 -22.0)。

NSVB-ZH 加入 LUFS -22 norm 之**真實動機**:防 D_z 利用「整體響度差」之 trivial shortcut 判別業餘/專業(amateur 錄音通常響度低於 pro)。此動機仍站得住;**代價**為:
- VV(原本響度低,且 DF3 後又再下降)被 scale up → 安靜段、DF3 殘留 artifact 也跟著被放大
- M4(已接近合理 LUFS 範圍)幾乎無事
- 兩端再次非對稱

→ docstring 已待更正,動機重新表述為「**防 shortcut**,不是對齊 vocoder」。

### 3.3 對 `pro_direction_alignment` 解讀之 caveat

公式:`pro_dir = cos(u_mod, u_dir)`,其中:
- u_mod = env(x_out) - env(x_base)
- u_dir = e_pro - env(x_base)

由於前處理對兩端非對稱:
- env(x_base) 帶有「post-DF3+LUFS 指紋」(VV 經處理後之過度乾淨/響度拉伸特徵)
- e_pro 不帶此指紋(M4 經處理後幾乎不變)
- u_dir 之方向含「移除指紋」成分,不純為技巧方向
- M 訓練時看到之訊號為「真實技巧改善 + 指紋補償」之疊加

**潛在風險**:`+0.84` 之高水位可能部分來自「M 補償非對稱指紋」而非「M 改善歌唱技巧」。

### 3.4 此一 caveat 不否定方法之合理性

前處理之**正向價值**(消除環境 shortcut)跟其**副作用**(對兩端非對稱)是同一件事的兩面。判讀 alignment 之意義需於獨立驗證下進行:
- **若 PopBuTFy 訓練(env 對稱情境)alignment 仍維持高水位 + 配對 MCD 下降** → 強證據:方法之核心 pipeline 含真實技巧軸
- **若 alignment 明顯降低** → 反向證明 v2 高水位確實有相當部分來自指紋補償;方法仍 work 但解讀需保留

任一結果都對論文有實質意義,且完全在方法前提之內。

---

## 四、PopBuTFy 為核心驗證路徑

### 4.1 為何 PopBuTFy 能於不違反方法前提下驗證 §3 之 caveat

PopBuTFy 之結構特性:
- 同歌手 × 同首歌之業餘 + 專業兩版本(paired)
- 同一歌手錄兩版,**錄音環境近似對稱**(同麥克風、同房間、同 setup)
- 業餘版有殘響時專業版通常也有(同環境下唱)

於此資料上跑**完全相同**之方法 pipeline(DF3 + LUFS + Stage 1 + Stage 2):
- 前處理仍開啟(不違反方法前提)
- DF3 + LUFS 之效應於兩端**對稱**(同環境之資料被同樣處理)
- 「post-DF3+LUFS 指紋」之方向於 u_dir 中**互相消除**
- 剩下之 u_dir 接近「純技巧方向」
- 此時 M 之 alignment 反映「方法 pipeline 對純技巧軸之掌握能力」

### 4.2 三種 scenario 之解讀

| PopBuTFy 訓練後 alignment | 解讀 | 對論文之含義 |
|---|---|---|
| 接近 v2 之 +0.84 | 方法 pipeline 即使於對稱前提下仍能達 alignment 高水位 → v2 +0.84 主要為真實技巧改善 | 強支撐 C2 contribution「技巧殘餘」假設之實證 |
| 顯著低於 v2(例如 +0.3 ~ +0.5) | 對稱情境下 alignment 降低 → v2 高水位有相當部分來自指紋補償 | 仍可發表,但 §4.6 需誠實揭露 alignment 之雙重貢獻 |
| 接近 0 或負值 | 對稱情境下方法失效 → v2 之 alignment 主要為環境補償 | 對 C2 提出嚴重挑戰,需重新思考 D_z 之判別軸設計(對應 v4 方向) |

### 4.3 配對指標之獨有價值

PopBuTFy 之 paired ground truth 提供**不受 envelope direction 影響**之客觀證據:
- **MCD vs paired pro**:M 之輸出聽起來有多像同歌手之 pro 版本
- **F0 RMSE vs paired pro**:音高軌跡改善幅度
- **SSIM vs paired pro**:mel envelope paired 相似度

若 MCD 顯著降低 + pro_dir 仍高 → 雙重支撐「M 確實在做朝同歌手 pro 版本之美化」。

### 4.4 NSVB 原版 baseline 對照(主對照,非可選)

於 PopBuTFy 上同時跑 NSVB 原版(paired supervised)+ 本研究(unpaired)兩條 pipeline,差距即為「paired supervision 之邊際價值 vs unpaired 設計之相對犧牲」 — 直接對應 C4 contribution。

執行路徑:
- NSVB 原版於 [`NSVB/`](../../NSVB) 之 repo 跑 `python tasks/run.py --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml --exp_name <name> --reset --infer`,輸出 wav 至 `checkpoints/<name>/generated_*`
- 本研究 model 之 Mode B inference 輸出依同 paired eval pipeline 計算
- 兩者於同一 eval script 之同一 DTW alignment 下對比,得到嚴格 apples-to-apples 之 paired metric

⚠ **NSVB 原版 repo 未 release metric 計算腳本**:檢查 [`NSVB/utils/metrics.py`](../../NSVB/utils/metrics.py)(僅含 `laplace_var`)與 [`NSVB/tasks/singing/svb_vae_task.py:304`](../../NSVB/tasks/singing/svb_vae_task.py)(`test_step` 只 save wav,不算 metric),NSVB 原版於 ACL-2022 paper 報之 MCD / F0 RMSE 為作者自寫之 post-hoc script 但未公開。我們之 baseline 重現視為「**best-effort matching of NSVB's evaluation protocol**」,於 §4.6 表 caption 須註明此 caveat。

### 4.5 Eval 流程設計(三個關鍵決定)

跨語言驗證之 eval 流程需於 paired metric(對標 NSVB 原版)與 unpaired metric(對標 v2 自身於中文 corpus 之 +0.84)之間做設計取捨。以下三決定鎖死整套 eval pipeline。

#### 4.5.1 推理模式:Mode B

採 Mode B([`nsvb/inference/pipeline.py:run_mode_b`](../nsvb/inference/pipeline.py))而非 Mode A。Mode B 輸入組合:
- amateur 端:PPG-encoded latent z(經 M 映射)+ spk_emb
- pro reference 端:F0 + PPG(用於 DTW 對齊到 T_p)
- 輸出時長:T_p

選擇理由:NSVB 原版為 paired training paradigm,訓中與 inference 均吃 `prof_f0`([`binarize_para.py:147`](../../NSVB/data_gen/singing/binarize_para.py))。Mode B 與此 paradigm 對齊,paired metric 才有可比性。

⚠ Mode B 之 F0 為 pro reference 之 ground truth copy,**F0 RMSE 不再衡量 model 之 F0 預測能力**,而是衡量「F0 沿 DTW alignment 之傳遞精度」(預期接近 0 Hz)。論文表 caption 須註明此前提。Mode A(全 amateur-driven)結果建議列入補充,供讀者判讀 model 之 F0 預測能力。

#### 4.5.2 DTW 對齊:EHSADTW(非 SADTW)

NSVB 原版提供 NaiveDTW / SADTW / EHSADTW / LoNDTW 等多種 DTW,本研究於 paired eval 中採 **EHSADTW**(對齊 [`NSVB/modules/voice_conversion/dtw/enhance_sadtw.py`](../../NSVB/modules/voice_conversion/dtw/enhance_sadtw.py)):

1. **避免自我設限**:NSVB 原版 binarize-time 即用 EHSADTW(`binarize_para.py:215` 之 `choosed_func='EHSADTW'`),寫入 `a2p_f0_alignment` 進 binary。若我們對自己 model 用較弱之 SADTW 算指標,等同套用比 NSVB 更不利之 yardstick;NSVB 用 EHSADTW 之較強對齊算自己時,我們本研究若退到 SADTW → 結論為 NSVB 略勝,但勝差為 DTW 選擇之 artifact 而非 model 本質差異
2. **直接 apples-to-apples**:同 DTW alignment 下算出之 MCD / F0 RMSE 才在同 yardstick
3. **實質差異不大**:SADTW 與 EHSADTW 之 distance 計算近似(EHSADTW 雖簽名收 UV mask,但 NSVB release 之 code 已 comment 掉);選 EHSADTW 主要為 1 + 2

SADTW 可作 supplementary 列,展示對 DTW 選擇之穩健性(若兩 DTW 下趨勢一致,reviewer 不易質疑結論依賴特定 DTW)。

實作須從 [`NSVB/modules/voice_conversion/dtw/enhance_sadtw.py`](../../NSVB/modules/voice_conversion/dtw/enhance_sadtw.py) port 該 module 進 NSVB-ZH(或直接以 subprocess 呼叫)。

#### 4.5.3 Eval matrix:paired + unpaired 並用

| 指標分類 | 指標 | 推理模式 | Reference / DTW | 對標 |
|---|---|---|---|---|
| **Paired(主比較表)** | MCD / SSIM / F0 RMSE | Mode B | pro chunk + EHSADTW | NSVB 原版 paper |
| **Paired supplementary** | 同上 | Mode B | pro chunk + SADTW | 穩健性 sanity check |
| **Paired Mode A 補充** | MCD / F0 RMSE | **Mode A** | pro chunk + EHSADTW | 證明 model 之 F0 預測能力(非 copy) |
| **Unpaired(v2 對照)** | `pro_direction_alignment` | Mode A 推理後算 envelope | PopBuTFy amateur/pro mean envelope(不需 DTW) | v2 之 **+0.84**(M4+VV) |
| **Unpaired(z 解讀)** | z-probe v3 | 不需推理(直接看 Stage 1 z) | 無 | v2 之 **57% env / 43% 未知**分解 |
| **Unpaired(domain gap)** | sfm / hf_ratio / unvoiced_concentration | 不需推理(直接看 mel) | 無 | v2 之 phase2_outcome 觀察 |

**為什麼 paired + unpaired 並用**:
- Paired 對標 NSVB 原版,提供業界共識之客觀數字
- Unpaired 對標 v2 自身於中文 corpus 之 +0.84,直接驗證「跨語言通用性」之主問題(§4.2 三種 scenario 之解讀)
- 兩套吻合 → 結論 robust;分歧 → 暴露 Mode B 之 F0-copy 假象,本身即論文價值

#### 4.5.4 配對之資料側預處理

PopBuTFy 之 source mp3 之 chunk 數於 amateur / pro 兩 side 不對等(amateur 14746 vs pro 14219;225 首歌之 chunk 數不一致;1 首歌 `Female15#singing#Forever` 只有 amateur 無 pro),反映上游切 chunk 並未 enforce 配對:
- **訓練不受影響**:Stage 2 為 unpaired adversarial,各 side 獨立隨機抽 batch
- **paired eval 須補處理**:[`nsvb/data/popbutfy_adapter.py:build_pairing`](../nsvb/data/popbutfy_adapter.py) 用 filename swap 配對(amateur chunk N ↔ pro chunk N),會自動跳過缺對。但 N↔N 配對之內容不保證對齊(切點不同,如 `Female16#singing#Marry_You` 之 amateur 63 / pro 7);需於 eval 階段加 `max_mel_tech_gap` filter(對應 NSVB 原版 [`para_bin.yaml:13`](../../NSVB/egs/datasets/audio/PopBuTFy/para_bin.yaml) 之 800 frames 閾值),長度差過大之 pair 於 eval 時 reject(不影響訓練)

具體實作:`scripts/popbutfy_paired_eval.py`(待 binarize 完成後撰寫),邏輯框架:
```
for am_npz, pr_npz in build_pairing(popbutfy_root).items():
    am = np.load(am_npz); pr = np.load(pr_npz)
    if abs(am['mel'].shape[0] - pr['mel'].shape[0]) > 800:
        continue  # eval-side reject(對應 NSVB max_mel_tech_gap)
    alignment = EHSADTW(am['f0'], pr['f0'], am['mel'])
    mcd, ssim, f0_rmse = compute_paired_metrics(model_output_mel, pr['mel'], alignment)
```

### 4.6 ROI 評估

PopBuTFy 訓練(方案 B:Stage 2 重訓,沿用 Stage 1)之 ROI:
- ~6.5h A100(或本機 RTX 3070 之估算,見 `scripts/estimate_popbutfy_training_time.py`)
- 一次實驗解決 §五清單中**多個議題之核心關切**(②、④、⑤ 之大部分)
- 額外提供 NSVB ver1 對照與 C4 contribution 之直接落地

**強烈建議優先執行**。

---

## 五、補充驗證實驗清單(在方法前提下)

依新前提淘汰/重定義既有清單:

### ① M4 端 DF3 no-op 量化【成本低,優先】
- 對若干 M4 sample 跑 `scripts/compare_df3_vocoder.py`
- 預期 mel L1 < 0.05,確認 M4 不受 DF3 影響
- 目的:**characterize 方法之非對稱效應量級**,作為§3.4.2 之透明揭露;**不是**「找問題」

### ② DF3 + LUFS 指紋方向於 u_dir 中之佔比【成本中】
- 計算 Δ_前處理 := env(處理後 x_a) - env(原始 x_a) 之平均向量
- 計算 cos(u_dir, Δ_前處理)
- 若 cos 中等(0.3-0.5)→ alignment 部分(但非全部)來自指紋補償
- 若 cos 接近 0 → 指紋與 pro-direction 接近正交,M 之修飾仍主要為純技巧

### ③ Linear probe on z【成本低】
- 訓小 MLP 從 z 預測 ppg cluster id / spk_emb / register / sfm 等 env proxies
- 成功預測 → z 漏條件資訊(condition decoupling 不完整)
- 失敗預測 → z 真之只剩技巧殘餘 + 未測量因素
- 獨立於前處理之 ablation,可直接執行

### ④ 跨環境推理測試【部署相關】
- 對 M4-like 環境之業餘樣本(若無現成,可用 VV 經輕度 DF3 處理之版本模擬)做推理
- 觀察 M 是否仍做有意義之修飾,或退化為 identity / 亂改
- 評估方法之 deployment 行為,不動 method

### ⑤(淘汰)~~no-DF3 baseline 訓練~~
**違反方法前提,不該做**。任何「拿掉前處理」之 ablation 等於拆掉方法之基石。

### ⑥ Domain-adversarial / condition-out-env【v4 改進方向】
- 於 Stage 1 把 sfm/風格 proxy 當條件輸入(類比 spk_emb 之處理),把 env 軸從 z 移出
- 或於 Stage 2 加 domain classifier + GRL,讓 D_z 不能用 env 分辨 → 逼 M 聚焦技巧軸
- **重定位**:這是 v4 之方法改進,不是 v2 之 validation

---

## 六、對論文呈現之建議

### 6.1 §3.4.2 去殘響與響度正規化(已有節,加 honest disclosure)

於現有內容後加一段:

> 本研究承認此一前處理對業餘端(高殘響/噪音)與專業端(乾錄)之影響非對稱:對業餘端為實質之 mel 分布改寫,對專業端近於 no-op。此一非對稱為「對齊環境分布」目標之必然代價;後續第 \ref{ch:experiments} 章將透過跨資料集驗證(PopBuTFy 之 paired data)補充驗證 M 之映射效應不單純為此一非對稱所主導。

### 6.2 §3.4.2 LUFS norm 動機之更正

[`audio_io.py:loudness_normalize`](../nsvb/utils/audio_io.py) 之 docstring 應更正為:

> 為什麼 -22 LUFS:本研究選用之響度正規化目標。**真實動機為防 D_z 利用「整體響度差」之 trivial shortcut 判別業餘/專業**(amateur 錄音之 LUFS 通常低於 pro)。雖然此值對應 NSVB `process_utterance` 函式內部之 hardcoded 值,但 NSVB 訓練 vocoder 時並未開啟此 normalization(`loud_norm: false`),故本選擇**不應稱為「對齊 NSVB vocoder 訓練分布」**。

### 6.3 §4.6 新增「跨資料集驗證對 alignment 解讀之補強」(待 PopBuTFy 結果)

於§4.6 現有「梅爾頻譜域與潛在空間指標之衝突」、「雙軌結構於泛化誤判風險之效應」、「基線扣除於指標解讀之具體效應」三節之後,新增:

> §4.6.x **跨資料集驗證對 alignment 解讀之補強**:本研究以 PopBuTFy 之 paired amateur/pro 資料(其環境對稱性中和中文主流程之前處理非對稱影響)為對照,於同一方法 pipeline 下重訓 Stage 2 並評估 alignment + 配對 MCD。[結果待補] 提供 M 之映射主要反映技巧層級而非環境指紋之經驗證據。

### 6.4 不新增「DF3 雙刃效應」獨立節

避免負面 framing。相關討論分散於 §3.4.2(honest disclosure)、§4.6(跨資料集驗證)、§4.7(若加之限制章節)。

### 6.5 C2 contribution 之「失效模式分析」具體落地

Ch1 §1.3 之 C2「將技巧殘餘假設明確化並提供其經驗驗證(含相關失效模式之分析)」 — 此處之「失效模式」可於 §4.6.x 中以「方法之非對稱前處理可能稀釋技巧軸投影之具體 caveat」具體化,而非泛泛之 future work 聲明。

---

## 七、待實驗完成後之檢視 checklist

實驗完成後檢視論文章節:

- [ ] 實驗 ①(M4 DF3 no-op)結果 → §3.4.2 之 disclosure 段加 M4 端數據
- [ ] 實驗 ②(指紋方向)結果 → §4.6.x 加「指紋方向佔比」量化證據
- [ ] 實驗 ③(linear probe)結果 → §4.6.x 加「z 之條件解耦驗證」
- [ ] 實驗 ④(跨環境推理)結果 → §4.6.x 加「deployment 行為」段
- [ ] **PopBuTFy 訓練 + 配對指標** → §4.6.x 之核心數據(走 §4.5 設計:Mode B + EHSADTW)
- [ ] **NSVB 原版 baseline 對照**(主對照,§4.4)→ 原版 `tasks/run.py --infer` 出 wav + 我們之 paired eval pipeline 算 metric
- [ ] **Unpaired metric 套用 PopBuTFy**(§4.5.3 matrix):`pro_direction_alignment` / z-probe v3 / sfm / hf_ratio,跑在 PopBuTFy 直接對照 v2 中文之 +0.84 / 57-43 分解
- [ ] **`scripts/popbutfy_paired_eval.py` 實作**:EHSADTW port + Mode B 推理 + paired metric;含 max_mel_tech_gap=800 之 eval-side filter
- [ ] LUFS norm 動機更正 → audio_io.py docstring + §3.4.2 文字

實驗 ①-④ 成本低,可於本研究時程內完成;PopBuTFy 訓練需 ~6.5h A100(或本機更長,待 `estimate_popbutfy_training_time.py` 量化);NSVB ver1 對照可作為 future work 或時間允許時補。

---

## 八、相關 codebase 與 memory 連結

**腳本**:
- `scripts/compare_df3_vocoder.py` — DF3 三變體對照
- `scripts/listen_vocoder_vv.py` — 指定樣本/chunk 之 vocoder 重建
- `scripts/estimate_popbutfy_training_time.py` — PopBuTFy 訓練時間估算(本機,跟 v2 hyperparams 一致,OOM 自動退讓)
- `scripts/stage1_audit.py` — Stage 1 重建品質檢驗
- `scripts/stage1_zprobe.py` — z-space domain-axis 變異分解(v3)
- `scripts/stage2_mel_eval_rerender.py` — 從 CSV 重生新版 report

**輸出**:
- `outputs/compare_df3_vocoder/` — DF3 三變體聽測樣本
- `outputs/listen_vocoder_vv_raw/` — 指定 chunk 之 raw 重建
- `outputs/stage1_audit/` — Stage 1 audit 結果
- `outputs/stage1_zprobe/` — z-probe v3 結果
- `outputs/stage2_v2_eval_fulltest/` — v2 完整測試集 eval(VV-only 主表 + M4 control)

**memory 連結**:
- [[project_v2_baseline_revised]]:v2 真實 alignment +0.84,評估方法論修正
- [[project_stage1_audit_findings]]:Stage 1 marginal-but-functional 之判讀
- [[feedback_stage1_zspace_interpretation]]:z 分離是設計成功,技巧軸正交於分離軸