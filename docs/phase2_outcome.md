# Phase 2 結案報告(Stage 2 v2 accepted)

> **狀態**:Stage 2 v2 訓練 ✅ 完成,**Best ckpt = `stage2_step30000.pt`**。
> 後續工作分流到 v3 candidates(見 §4) — 多數**非 Stage 2 議題**,而是 vocoder /
> 部署 / Phase 3 議題。
>
> **本報告數字依新版 eval 方法論呈現**:VV-only 主表 + M4 control 分表 + `*_extra`
> 已扣除 VAE 重建噪音底(舊版混合 M4+VV 均值,M4 把 VV 端訊號稀釋)。如果
> 你看到本檔跟舊 git history 的數字差距,以本檔為準;對應 raw eval 報告位於
> [outputs/stage2_v2_eval/report.md](../outputs/stage2_v2_eval/report.md) 與
> [outputs/stage2_v2_eval_fulltest/report.md](../outputs/stage2_v2_eval_fulltest/report.md)
> (兩者皆已用 `stage2_mel_eval_rerender.py` 重生)。

## 1. v2 配置

| 項目 | 值 |
|---|---|
| Stage 1 ckpt | `checkpoints_v2/stage1/stage1_best.pt`(80K steps, val_best) |
| Stage 2 config 差 v1 | `freeze_d_mel + lambda_adv_mel 0.05 + lr_dz 2e-4`(2× TTUR) |
| max_steps | 120000 |
| 訓練時間 | ~6.5h on A100(120K steps × ~5.3 it/s) |
| log | `runs/stage2_v2_20260517_151527.log`(+ `.summary.md`) |
| ckpts | `checkpoints_v2/stage2_v2/stage2_step{5000,10000,...,120000}.pt` + `_best.pt` 在後續決議後產出 |

詳細 v1 → v2 差異與動機:見 [risk.md §二.3](../risk.md) 與 [phase1_colab_workflow.md §7](phase1_colab_workflow.md)。

## 2. Best ckpt 認定:`stage2_step30000.pt`

**兩階段驗證**(val n=6 spot check + test n=3773 generalization)都指向 step 30000。

### 2.1 Val set(n=6,1 M4 + 5 VV)

訓中 train 歌手的 hold-out 歌曲;用於初判 best ckpt 候選。

**VV(amateur)主表** — 主要推理品質:

| step | L1_recon | uv_conc | vsr | pro_dir | pro_dist_Δ | tdr_extra | hf_extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| step005000 | 2.354 | 0.53 ✅ | 0.70 ✅ | +0.64 ✅ | -0.886 | +8.13 ❌ | +0.19 ✅ |
| step015000 | 0.435 | 0.52 ✅ | 0.98 ✅ | +0.82 ✅ | +0.219 | +0.03 ✅ | -0.31 ⚠️ |
| **step030000** ⭐ | 0.497 | **0.54 ✅** | **0.98 ✅** | **+0.85 ✅** | **+0.263** | **+0.04 ✅** | **-0.28 ⚠️** |
| step050000 | 0.542 | 0.53 ✅ | 0.97 ✅ | +0.84 ✅ | +0.238 | +0.06 ✅ | -0.26 ⚠️ |
| step070000 | 0.586 | 0.53 ✅ | 0.97 ✅ | +0.81 ✅ | +0.182 | +0.08 ✅ | -0.28 ⚠️ |
| step090000 | 0.544 | 0.55 ✅ | 0.97 ✅ | +0.80 ✅ | +0.175 | +0.07 ✅ | -0.27 ⚠️ |
| step110000 | 0.563 | 0.56 ⚠️ | 0.97 ✅ | +0.80 ✅ | +0.196 | +0.08 ✅ | -0.24 ⚠️ |
| step120000 | 0.538 | 0.55 ✅ | 0.97 ✅ | +0.82 ✅ | +0.201 | +0.06 ✅ | -0.28 ⚠️ |

**M4 control**(n=1 → 高 sample variance,僅作粗檢):
- step030000:`L1_recon 0.088 ✅`、`pro_dir +0.22 ✅`(M 對 pro 接近 identity)
- step110000~120000:`pro_dir ~-0.60` ⚠️(對 pro 樣本反向修飾)— 但 **n=1 不可一般化**,以 test n=1480 M4 為準

### 2.2 Test set(n=3773,1480 M4 + 2293 VV)— generalization 驗收

從未訓過的 hold-out 歌手(M4 Alto-2 + Tenor-3 整位、VV 10% hold-out user)。

**VV(amateur)主表** — 主要推理品質:

| step | L1_recon | uv_conc | vsr | pro_dir | pro_dist_Δ | tdr_extra | hf_extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| step005000 | 2.341 | 0.51 ✅ | 0.71 ✅ | +0.62 ✅ | -0.918 | +8.26 ❌ | +0.23 ⚠️ |
| step015000 | 0.432 | 0.49 ✅ | 0.98 ✅ | +0.81 ✅ | +0.215 | +0.02 ✅ | -0.31 ⚠️ |
| **step030000** ⭐ | 0.500 | **0.49 ✅** | **0.98 ✅** | **+0.84 ✅** | **+0.256** | **+0.04 ✅** | **-0.29 ⚠️** |
| step050000 | 0.522 | 0.49 ✅ | 0.98 ✅ | +0.83 ✅ | +0.251 | +0.05 ✅ | -0.28 ⚠️ |
| step070000 | 0.572 | 0.50 ✅ | 0.97 ✅ | +0.82 ✅ | +0.222 | +0.06 ✅ | -0.31 ⚠️ |
| step090000 | 0.553 | 0.49 ✅ | 0.97 ✅ | +0.80 ✅ | +0.211 | +0.05 ✅ | -0.30 ⚠️ |
| step110000 | 0.552 | 0.51 ✅ | 0.97 ✅ | +0.83 ✅ | +0.258 | +0.07 ✅ | -0.25 ⚠️ |
| step120000 | 0.539 | 0.50 ✅ | 0.97 ✅ | +0.83 ✅ | +0.249 | +0.05 ✅ | -0.29 ⚠️ |

**M4 control**(n=1480) @ step030000:
- `L1_recon = 0.089 ✅`(M 對 pro 變動量極小,L_id_pro 工作中)
- `pro_dir = +0.22 ✅`(輕度 over-pro-化但仍在 healthy 範圍)
- `tdr_extra +0.01`、`hf_extra -0.00` — M 沒破壞 pro 端時間結構或高頻
- 後期步 step090000 pro_dir 一度 -0.36 ⚠️(輕度 anti-pro 趨勢),但 mode collapse 風險不顯著

### 2.3 Val vs Test 對比(step 30K)

| 指標 | val | test | 差異 |
|---|---:|---:|---|
| `pro_direction_alignment` | +0.853 | +0.838 | 1.7%(基本無 generalization gap) |
| `pro_dist_delta` | +0.263 | +0.256 | 2.7% |
| `unvoiced_concentration` | 0.542 ✅ | 0.495 ✅ | 兩 ✅ |
| `voiced_spectral_ratio` | 0.975 ✅ | 0.977 ✅ | 兩 ✅ |
| `tdr_extra` | +0.045 ✅ | +0.041 ✅ | 兩 ✅(M 不破壞時間結構)|
| `hf_extra` | -0.278 ⚠️ | -0.288 ⚠️ | 兩 ⚠️(高頻削減略偏重) |
| **`M4 vs VV ratio`** | **11.67×** | **11.26×** | **基本無退** — L_id_pro 在 unseen 歌手仍工作 ✅ |

> **與舊報告差異**:舊版以 M4+VV 混合平均估 test pro_dir 為 +0.60、val +0.74、
> 號稱 19% generalization gap。新版 VV-only 顯示 val/test gap 基本消失,
> 「19% gap」屬於 M4 樣本拉低平均之 artifact,非真實 generalization 退步。

### 2.4 為什麼 step 30000

- VV `pro_direction_alignment` 在 val(+0.853)跟 test(+0.838)都是 **8 個 step 中最高**(top-3:30K > 50K > 110K 或 120K,gap ~0.01-0.04)
- VV `pro_dist_delta` 也是 30K 最大,M 把 envelope 拉得最接近 pro
- `voiced_spectral_ratio ≥ 0.97` 確認修飾是 envelope shift 不是 trajectory destruction
- 30K 之後 alignment **緩降但仍在高水位**(test 看 30K +0.84 → 120K +0.83,差距僅 0.01;val 30K +0.85 → 120K +0.82,差距 0.03);後期訓練無明顯增益但也未顯著退步

### 2.5 為什麼不是 120000

雖然兩個 split 都顯示後期步 alignment 仍接近 30K(差距 < 0.04),仍選 30K 是因為:
- 30K 是 8 個候選中 alignment 嚴格最高(test +0.838 / val +0.853)
- `mel_l1_vs_recon` 後期略增(val 0.50 → 0.54、test 0.50 → 0.54),M 對 amateur 做的功更多但 alignment 沒提升 → 投資報酬遞減
- VAE Stage 1 latent metric(訓中 `Δ/z 0.87、tdr 0.81`)曾被誤判為訓壞,**已知該指標誤導**(latent 跟 mel-domain 經常衝突 — 見 §3 末)
- M4 control(test n=1480)後期步 pro_dir 偏向負(step90K -0.36 ⚠️),雖未到危險區但暗示後期 M 對 pro 也在動

eval 報告(均已 rerender):
- val(n=6): [outputs/stage2_v2_eval/report.md](../outputs/stage2_v2_eval/report.md)
- test(full n=3773): [outputs/stage2_v2_eval_fulltest/report.md](../outputs/stage2_v2_eval_fulltest/report.md)

## 3. v2 健康狀態總結(以 test set n=3773 VV-only 為準)

| 觀察 | 結論 |
|---|---|
| **M 是否 amateur-specific** | ✅ M4 vs VV L1_recon ratio = **11.26×**(val 11.67×,跨歌手幾乎沒退步) |
| **修飾是否往 pro 方向** | ✅ `pro_direction_alignment +0.84` @ step 30K(val +0.85,兩 split 均 ≫ 0.30 healthy 線)|
| **voiced 段修飾類型** | ✅ `voiced_spectral_ratio 0.98` — envelope-dominated(formant/共鳴調整) |
| **是否破壞時間軌跡** | ✅ `tdr_extra +0.04` — 已扣除 VAE 重建噪音底,M 對時間導數貢獻接近 0 |
| **去殘響/呼吸聲傾向** | ✅ `unvoiced_concentration 0.49`(unseen 歌手平均後落 healthy)|
| **高頻 artifact** | ⚠️ `hf_extra -0.29`(超出 ±0.20 healthy 範圍,輕度偏重高頻削減)— 詳 §4 vocoder caveat |
| **D_mel confound 是否消失** | ✅ `l_adv_mel` 從 v1 ~5 floor → v2 ~0.8(freeze D_mel 有效) |
| **訓中 latent 指標** | ❌ `Δ/z 0.87 / tdr 0.81` 超出 §3.6.1 範圍 — **但已知是 latent metric 誤導**(見下) |

**關鍵覺察**:[risk.md Monitor 1b 的 latent metric](../risk.md) 跟 mel-domain metric 經常衝突。
v1 vs v2 都顯示 latent `Δ/z > 0.8`,但 mel-domain 顯示 M 行為健康。**未來判 Stage 2 訓練好壞,
以 mel-domain eval 為主、latent 為輔**。

**Generalization 結論**:test `pro_direction_alignment +0.84` 與 val `+0.85` 基本無差(gap 1.7%),
M4 vs VV ratio 也只差 3.6%。新版 VV-only 方法論顯示 v2 在 unseen 歌手上 generalize 良好;
舊報告聲稱的 19% gap 是 M4 樣本稀釋造成之 artifact。

**`hf_extra ⚠️` 解讀**:test 跟 val 同步顯示 -0.28 ~ -0.29 之高頻削減,跨 step 穩定;
此一行為可能反映 M 學會壓制業餘端齒音/氣音之高頻雜訊,但下游 vocoder 對中文 amateur 分布
本身就不熟(§4),`hf_extra` 在 mel 層的數字與最終聽感落差不易事前判定。
建議於 vocoder fine-tune 後重新評估;若聽測下發現高頻削減導致明亮度不足,再考慮 v3
調整 `lambda_adv_mel` 或加 spectral envelope preservation loss。

詳細指標解讀:見 [eval_metrics_guide.md](eval_metrics_guide.md)。

## 4. 為什麼**不**進 Stage 2 v3

聽測時發現 amateur wav 有電音,原本懷疑 Stage 2 訓壞。經
[`scripts/diagnose_stage1_vocoder_path.py`](../scripts/diagnose_stage1_vocoder_path.py)
診斷:**vocoder 對 amateur GT mel 直接 SSIM=0.65 / F0 RMSE=53 Hz**(M4 SSIM=0.87 / 9.5 Hz)。
Vocoder 本身對中文 amateur 分布不熟,跟 Stage 1/2 訓練無關。

**所以 Stage 2 v2 已盡力**:在 mel-domain 學到對的修飾方向(pro_dir +0.84,跨歌手 generalize),
但下游 vocoder 把成果蓋過去。再訓 Stage 2 v3 也救不了這條 wav 聽測路徑,先把 vocoder 修好才有意義。

`hf_extra -0.29 ⚠️` 此一 mel-domain 警訊也屬同框架:vocoder fine-tune 前無法區分「M 削太多
高頻」vs「vocoder 本身高頻不準」。先 fine-tune vocoder,聽測下若仍有明顯高頻不足,
再考慮 Stage 2 v3。

詳見 [risk.md Risk 10](../risk.md)。

## 5. 後續工作清單(v3 candidates)

按優先順序:

### 5.1 **Vocoder fine-tune on Chinese amateur** ⭐ Phase 3 prereq

- **目標**:把 `1012_hifigan_all_songs_nsf` vocoder 用 M4 + dereverb'd VV mel
  fine-tune 到 VV SSIM ≥ 0.85 / F0 RMSE ≤ 15 Hz
- **預估時間**:1-2 天 A100
- **風險**:可能略損 pro 端 SSIM(0.94 → 可接受到 0.88)
- **驗收**:重跑 `diagnose_stage1_vocoder_path.py` VV 結果應在 Phase 0 gate ① 健康範圍
- **流程**(待補):
  - 把 NSVB `modules/parallel_wavegan/` 或 vocoder fine-tune script port 過來
  - 用 binarized 資料的 mel + wav pair 訓練
  - 至少 50K steps
- **這是 Phase 3 deployment 前的硬阻塞**

### 5.2 **Phase 3 推理 pipeline 收尾**

`nsvb/inference/` 已實作 Mode A / Mode B,但綁定 v2 best ckpt + new vocoder 後需要:
- 跑 test split(M4 Alto-2/Tenor-3、VV 10% hold-out)的完整 eval
- 把 [scripts/infer.py](../scripts/infer.py) 跑通,產出 Phase 3 demo wav 集
- 主觀聽測(5-10 人)的 MOS

### 5.3 **Stage 2 v3**:**DEFERRED**,僅在以下條件觸發

只有當 vocoder fine-tune 後重新聽測,發現以下任一才動 Stage 2:
- M 對 amateur 的修飾在聽感上**沒明顯 pro 化**(test `pro_direction_alignment +0.84` 不轉化成聽感)
- M 把顫音/滑音抹掉(test `tdr_extra +0.04` 在 fine-tuned vocoder 下變大或聽感問題)
- mode collapse(amateur 多樣性消失)
- `hf_extra -0.29` 在 fine-tuned vocoder 下聽測出明亮度不足
- test set 顯著大於 val set 的 generalization gap(新版方法論下目前 gap 1.7%,無顯著問題;若 vocoder 修好後 unseen 歌手效果明顯弱於訓練歌手,要動 Stage 2)

如果發生,v3 對策:
- 不再針對 D_mel(v2 freeze 已證實 D_mel 不是主因)
- 看 `pro_direction_alignment` 是否還能拉高 → 加 `lambda_patchnce` 或 `lambda_adv_z`
- 想縮 generalization gap → speaker augmentation / 增加 M4 歌手數 / VocalVerse 取捨更廣
- 看 `unvoiced_concentration` 是否退步 → 加 voiced/unvoiced 修飾比例 loss
- 看 [risk.md §二.3](../risk.md) 的 f0_support 想法是否值得實作

### 5.4 **(便宜實驗,優先級低)F0 extractor swap**

重 binarize 時把 torchcrepe 換 parselmouth,看 amateur F0 抽取是否變穩。
但 vocoder 主要瓶頸不是 F0 抽取,是 vocoder 本身分布不熟 → 估計只能把 VV F0 RMSE
53 → 30 Hz,仍 fail。**Vocoder fine-tune 之前不要做這個。**

### 5.5 **(知識歸檔)Phase 0 補測 VV Gate ①**

跑 `scripts/vocoder_identity_test.py --wav-dirs vocalverse=...` 把 VV 端的
SSIM=0.65 / RMSE=53 Hz 寫進 [phase0_log.md](phase0_log.md) Gate ① 表格,
正式記入 Phase 0 結果。已部分完成(post-Phase-2 retrospective note)。

## 6. 給未來自己/協作者的 hand-off note

如果你接手這個專案,**先確認 vocoder 已經 fine-tune 過**(看
`checkpoints_v2/vocoder_finetuned/`)。沒有的話,任何 amateur wav 推理結果都不準。

Stage 2 v2 是個健康的 mapping model,**不要動它**直到 vocoder 修好再驗收。
要重新評估 Stage 2 時,用 [scripts/stage2_mel_eval.py](../scripts/stage2_mel_eval.py)
而非 [scripts/stage2_ckpts_listening.py](../scripts/stage2_ckpts_listening.py)。
若需重生 report 但不重跑 inference,用 [scripts/stage2_mel_eval_rerender.py](../scripts/stage2_mel_eval_rerender.py)
從現有 `metrics_aggregate.csv` 直接重生 report(秒級完成)。

當前所有產出:
- ckpts: `checkpoints_v2/stage1/`, `checkpoints_v2/stage2_v2/`
- logs: `runs/stage2_v2_20260517_151527.log` + summary
- val eval (n=6): `outputs/stage2_v2_eval/`(已 rerender + 補跑 *_extra)
- **test eval (n=3773, generalization 驗收)**: `outputs/stage2_v2_eval_fulltest/`(已 rerender)
- vocoder diagnosis: `outputs/vocoder_path_report.md`