# Phase 2 結案報告(Stage 2 v2 accepted)

> **狀態**:Stage 2 v2 訓練 ✅ 完成,**Best ckpt = `stage2_step30000.pt`**。
> 後續工作分流到 v3 candidates(見 §4) — 多數**非 Stage 2 議題**,而是 vocoder /
> 部署 / Phase 3 議題。

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

| step | `pro_direction_alignment` | `pro_dist_delta` | `unvoiced_concentration` | `voiced_spectral_ratio` |
|---|---:|---:|---:|---:|
| 5000 | +0.45 | -0.78(post-warmup chaos) | 0.59 | 0.71 |
| 15000 | +0.65 | +0.19 | 0.56 | 0.95 |
| **30000** ⭐ | **+0.74** | **+0.23** | 0.57 | 0.96 |
| 50000 | +0.67 | +0.20 | 0.58 | 0.96 |
| 70000 | +0.61 | +0.15 | 0.57 | 0.97 |
| 90000 | +0.61 | +0.15 | 0.57 | 0.96 |
| 110000 | +0.60 | +0.16 | 0.60 | 0.96 |
| 120000 | +0.61 | +0.17 | 0.59 | 0.96 |

### 2.2 Test set(n=3773,1480 M4 + 2293 VV)— generalization 驗收

從未訓過的 hold-out 歌手(M4 Alto-2 + Tenor-3 整位、VV 10% user)。**統計穩定 25× 高於 val**。

| step | `pro_direction_alignment` | `pro_dist_delta` | `unvoiced_concentration` | `voiced_spectral_ratio` | `tdr_extra` | `hf_extra` |
|---|---:|---:|---:|---:|---:|---:|
| 5000 | +0.33 | -0.56(post-warmup chaos) | 0.61 | 0.75 | +5.03 ❌ | +0.13 |
| 15000 | +0.49 | +0.13 | **0.52 ✅** | 0.92 | +0.02 | -0.19 |
| **30000** ⭐ | **+0.60** | **+0.16** | **0.54 ✅** | 0.96 | +0.03 | -0.18 |
| 50000 | +0.46 | +0.15 | 0.55 ✅ | 0.95 | +0.03 | -0.17 |
| 70000 | +0.38 | +0.13 | 0.53 ✅ | 0.97 | +0.03 | -0.21 ⚠ |
| 90000 | +0.35 | +0.12 | 0.53 ✅ | 0.97 | +0.03 | -0.20 |
| 110000 | +0.40 | +0.15 | 0.57 ⚠ | 0.96 | +0.04 | -0.16 |
| 120000 | +0.44 | +0.15 | 0.55 ⚠ | 0.96 | +0.03 | -0.18 |

### 2.3 Val vs Test 對比(step 30K)

| 指標 | val | test | 差異 |
|---|---:|---:|---|
| `pro_direction_alignment` | +0.74 | +0.60 | -19%(generalization gap,可接受 — 仍遠 > 健康線 0.3)|
| `pro_dist_delta` | +0.23 | +0.16 | -30% |
| `unvoiced_concentration` | 0.57 ⚠ | **0.54 ✅** | val 6 samples 過小有 outlier,test 平均後落 healthy |
| `voiced_spectral_ratio` | 0.96 ✅ | 0.96 ✅ | 同 |
| **`M4 vs VV ratio`** | **11.67×** | **11.26×** | **幾乎沒退** — L_id_pro 在 unseen 歌手仍工作 ✅ |
| `tdr_extra` | — | **+0.03 ✅** | 大樣本確認 M 沒破壞時間結構(baseline tdr 1.18 是 VAE 噪音底,M 只貢獻 0.03)|

### 2.4 為什麼 step 30000

- `pro_direction_alignment` 在 val(+0.74) 跟 test(+0.60) 都是 **8 個 step 中最高**
- `pro_dist_delta` 也是 30K 最大,M 把 envelope 拉得最接近 pro
- `voiced_spectral_ratio ≥ 0.95` 確認修飾是 envelope shift 不是 trajectory destruction
- 30K 之後 alignment 緩降(test 看更明顯:30K +0.60 → 70K +0.38),**後期訓練無效甚至略退步**

### 2.5 為什麼不是 120000

雖然 latent 指標(`Δ/z 0.87`、`tdr 0.81`)被誤判為訓壞,但 mel 指標顯示 alignment 從 30K 的 +0.74
跌回 +0.61(val) / +0.60 → +0.44(test) — **後期訓練是無效的**。
`mel_l1_vs_recon` 兩個 split 都沒在末期增加(0.43 → 0.46 / 0.34 → 0.35),代表 M 不是「越訓越激進」
而是「漂向 less efficient 方向」。

eval 報告:
- val: [outputs/stage2_v2_eval/report.md](../outputs/stage2_v2_eval/report.md)
- test(full): [outputs/stage2_v2_eval_fulltest/report.md](../outputs/stage2_v2_eval_fulltest/report.md)

## 3. v2 健康狀態總結(以 test set n=3773 為準)

| 觀察 | 結論 |
|---|---|
| **M 是否 amateur-specific** | ✅ M4 vs VV L1_recon ratio = **11.26×**(val 11.67×,跨歌手幾乎沒退步) |
| **修飾是否往 pro 方向** | ✅ `pro_direction_alignment +0.60` @ step 30K(val +0.74,test 略低但仍 > 0.3 healthy)|
| **voiced 段修飾類型** | ✅ `voiced_spectral_ratio 0.96` — envelope-dominated(formant/共鳴調整) |
| **是否破壞時間軌跡** | ✅ **`tdr_extra +0.03`** — M 對時間導數的貢獻接近 0;原本看 `tdr_mel 1.22` 的 ❌ 是 VAE 重建噪音底,扣除後 M 是清白的 |
| **去殘響/呼吸聲傾向** | ✅ `unvoiced_concentration 0.54` @ step 30K(test scale 平均後落 healthy,val 0.57 ⚠ 是 6-sample 抽樣偏差)|
| **高頻 artifact** | ✅ `hf_extra -0.18`(±0.20 healthy 範圍內,輕度高頻削減)|
| **D_mel confound 是否消失** | ✅ `l_adv_mel` 從 v1 ~5 floor → v2 ~0.8(freeze D_mel 有效) |
| **訓中 latent 指標** | ❌ `Δ/z 0.87 / tdr 0.81` 超出 §3.6.1 範圍 — **但已知是 latent metric 誤導**(見下) |

**關鍵覺察**:[risk.md Monitor 1b 的 latent metric](../risk.md) 跟 mel-domain metric 經常衝突。
v1 vs v2 都顯示 latent `Δ/z > 0.8`,但 mel-domain 顯示 M 行為健康。**未來判 Stage 2 訓練好壞,
以 mel-domain eval 為主、latent 為輔**。

**Generalization gap**:test 的 `pro_direction_alignment +0.60` 比 val `+0.74` 低 19%。
- 解讀:M 學到的 pro 方向對訓練分布內歌手最有效,對 unseen 歌手稍弱
- **不算 overfit**(仍 > 0.3 healthy 線,M4 vs VV ratio 維持 11.26×)
- 若 Phase 3 想拉滿,v3 可考慮加 speaker augmentation 或增加歌手多樣性(見 §5.3)

詳細指標解讀:見 [eval_metrics_guide.md](eval_metrics_guide.md)。

## 4. 為什麼**不**進 Stage 2 v3

聽測時發現 amateur wav 有電音,原本懷疑 Stage 2 訓壞。經
[`scripts/diagnose_stage1_vocoder_path.py`](../scripts/diagnose_stage1_vocoder_path.py)
診斷:**vocoder 對 amateur GT mel 直接 SSIM=0.65 / F0 RMSE=53 Hz**(M4 SSIM=0.87 / 9.5 Hz)。
Vocoder 本身對中文 amateur 分布不熟,跟 Stage 1/2 訓練無關。

**所以 Stage 2 v2 已盡力**:在 mel-domain 學到對的修飾方向,但下游 vocoder 把成果蓋過去。
再訓 Stage 2 v3 也救不了這條 wav 聽測路徑,先把 vocoder 修好才有意義。

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
- M 對 amateur 的修飾在聽感上**沒明顯 pro 化**(test `pro_direction_alignment +0.60` 不轉化成聽感)
- M 把顫音/滑音抹掉(test `tdr_extra +0.03` 在 fine-tuned vocoder 下變大或聽感問題)
- mode collapse(amateur 多樣性消失)
- test set 顯著大於 val set 的 generalization gap(目前 19% 可接受;若 vocoder 修好聽測下發現 unseen 歌手效果明顯弱於訓練歌手,要動 Stage 2)

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

當前所有產出:
- ckpts: `checkpoints_v2/stage1/`, `checkpoints_v2/stage2_v2/`
- logs: `runs/stage2_v2_20260517_151527.log` + summary
- val eval (n=6): `outputs/stage2_v2_eval/`
- **test eval (n=3773, generalization 驗收)**: `outputs/stage2_v2_eval_fulltest/`
- vocoder diagnosis: `outputs/vocoder_path_report.md`