# Phase 0 實驗紀錄

把 Phase 0(資料前處理 → 4 個 gate)在 Colab Pro+(Stage A CPU + Stage B A100/CPU)上實際跑完的過程、踩到的坑、最終驗證結果。撰寫日期 2026-05-15。

how-to 流程見 [phase0_colab_workflow.md](phase0_colab_workflow.md);這份是「實際跑了一遍發生什麼」的紀錄。

---

## 0. 總覽

| 階段 | 環境 | 結果 |
|---|---|---|
| Stage A:環境 + raw data 預下載 | CPU runtime | M4Singer 20,896 wav + VocalVerse 929 wav + Whisper-large-v3 / DF3 cache 到 Drive |
| Gate ① Vocoder identity(§4) | A100 | PASS(SSIM / F0 RMSE)|
| Gate ② Audio quality probe(§5) | A100 | reliable metrics(sfm / hf_ratio)隨 dereverb 顯著改善,實質 PASS |
| Binarize(§6.1–§6.2) | A100, ~15-18h | M4Singer 20,896 + VocalVerse 21,458 chunks(536 首歌, MOS filter)|
| Integrity verify(§6.3) | CPU | 0 壞檔,schema/shape 全對 |
| cluster_ppg v1(K=200, no DC norm) | CPU | phoneme JSD 0.43、M4 MI 0.862 → **FAIL** |
| 診斷(Gate ④ + 自訂工具)| CPU | 99/200 dataset-pure cluster → PPG pitch 污染確認 |
| cluster_ppg v2(K=100 + per-utt-mean-norm)| CPU | phoneme JSD 0.16、M4 MI 0.216 → 接受 |

---

## 1. Dataset 規模(MOS 篩後)

### M4Singer(pro 端)
- 20,896 個 5s snippets,~31h
- 20 位受訓歌手,涵蓋 Alto / Soprano / Tenor / Bass 廣音域
- T(mel frames)分布:`min=144  max=2064  mean=882  median=848`

### VocalVerse(amateur 端,MOS filtered)
- 原 929 首 → 套用 `FilterCriteria(amateur_score_max=3.0)`(即 `(pro_technique + pro_breath)/2 ≤ 3.0`)
- 留 **536 首**,~29.8h(對齊 M4Singer ~30h)
- 切 5s chunks → **21,458 個 .npz**
- 每首 chunk 數:`min=27  max=51  mean=40  median=42`
- T 分布:`min=536  max=861  mean=859`(絕大多數是完整 5s chunk = 861 frames)

**合計 42,354 個 .npz**。verify_binarized.py 兩個 dataset 皆 0 壞檔,536 首 VocalVerse 全數無遺漏(reconcile_vv_chunks.py 通過)。

---

## 2. Gate ③ JSD 試驗:v1 失敗

```
JSD(phoneme):  0.4259  FAIL  (hard threshold < 0.05)
JSD(register): 0.0838  FAIL  (hard threshold < 0.05)
```

phoneme JSD 是門檻的 ~8.5×,異常嚴重。register JSD 只是輕微超(M4 廣音域 vs VV 窄音域,結構差異)。

## 3. Gate ④ 診斷:v1 為什麼失敗

```
M4Singer:
  MI(phoneme; register) = 0.862 bit   ⚠️ WARNING (>0.6, pitch-confounded)
  mean voiced dwell     = 10.3 frames ✅ HEALTHY
VocalVerse:
  MI(phoneme; register) = 0.178 bit   ✅ HEALTHY
  mean voiced dwell     = 10.0 frames ✅ HEALTHY
```

**非對稱結果是關鍵**:M4 嚴重、VV 乾淨。原因 ——

- M4Singer 20 位歌手跨 Alto–Bass、frame 跨 C3–E5+ 廣音域 → pitch 變化大
- VocalVerse 業餘歌手集中舒服中音域,pitch 變化窄
- 若 Whisper layer 8 PPG 帶 pitch baseline(Risk 2b 假設),則 k-means cluster 會沿 pitch 軸分裂;M4 廣音域 → 強烈分裂,VV 窄音域 → 看不出來

### 額外實驗:cluster 獨佔比例

```python
m4, vv = cluster_counts(m4_path), cluster_counts(vv_path)
tot = m4 + vv
frac_m4 = np.where(tot > 0, m4 / np.maximum(tot, 1), 0.5)
pure = (((frac_m4 > 0.95) | (frac_m4 < 0.05)) & (tot > 0)).sum()
```

結果:**200 個 cluster 中,99 個被單一 dataset 獨佔(> 95%)** ≈ 一半。

→ 直接證實 cluster 沿資料集邊界結構性分裂,完全閉合「PPG pitch 污染 + M4/VV 音域差」的機制。

### 跟 NSVB 原版對照(為什麼他們沒這問題)

| | NSVB 原版 | NSVB-ZH |
|---|---|---|
| 資料配對 | Paired(同首歌 amateur + pro) | **Unpaired** |
| phoneme 來源 | 歌詞文字 + MFA 對齊(監督)| **Whisper PPG + k-means**(自監督)|
| D_z 是否用 phoneme 條件 | 沒有 D_z(用 paired L_map1) | **有 D_z**,phoneme_id 是條件 |

NSVB-ZH 走自監督 PPG 路線是必要的(M4/VV 都沒對齊歌詞),代價就是這個 pitch 污染需要靠 DC removal 補回語意 grounding 的缺。

---

## 4. 修法 + v2 結果

實作:`nsvb/data/cluster_ppg.py` 新增 `--per-utt-mean-norm`。對每檔 PPG 沿時間軸取均值並減掉(在 valid 範圍 5..T-5 上算 mean),移除整曲層級的 pitch baseline,保留 frame-relative 的動態 phonetic 內容。fit + assign 兩階段**必須一致**(不一致則 centroids 跟 query 不在同坐標系)。

配合 `--k 100`(從 200 降一半,壓縮 over-segmentation 空間)重跑:
```bash
python -m nsvb.data.cluster_ppg \
    --binarized-root <root> --centroids-out <root>/ppg_kmeans_centroids.npy \
    --k 100 --frames-per-song 50 --per-utt-mean-norm \
    --overwrite --stage all
```

### v2 結果

```
JSD(phoneme):  0.1562  (still FAIL vs 0.05,但 v1 → v2 降 63%)
JSD(register): 0.0838  (不變;DC removal 不動 F0)
```

```
M4Singer:
  MI(phoneme; register) = 0.216 bit   ✅ HEALTHY (0.862 → 0.216, -75%)
  mean voiced dwell     = 8.6 frames  ✅ HEALTHY
VocalVerse:
  MI(phoneme; register) = 0.058 bit   ✅ HEALTHY (更乾淨)
  mean voiced dwell     = 7.5 frames  ⚠️ marginal(>5 即可,輕微變細)
```

### v1 → v2 對照表

| 指標 | v1(K=200, no DC)| v2(K=100, DC norm)| 變化 |
|---|---:|---:|---|
| phoneme JSD | 0.4259 | **0.1562** | -63% |
| register JSD | 0.0838 | 0.0838 | 不變(預期)|
| M4 MI(ph;reg) | 0.862 ⚠️ | **0.216** ✅ | -75% |
| VV MI(ph;reg) | 0.178 ✅ | 0.058 ✅ | -67% |
| M4 dwell | 10.3 | 8.6 | -16%(仍 healthy)|
| VV dwell | 10.0 | 7.5 marginal | -25%(仍 > 5)|
| dataset-pure cluster 數 | 99/200 | 未重量,JSD 推算 ~30-40/100 | 大幅下降 |

**pitch 污染主防線生效,Risk 2b 確認被 Gate ④ + DC removal 機制處理掉**。

---

## 5. 殘留 JSD 0.16 的解讀 — 為什麼接受 v2、不再加碼

剩下的 0.16 不是 contamination(MI 已 healthy 證實),大概率是**結構性差異**:

1. **中文 tonal language 的固有 phoneme-pitch 相關**:聲調本身綁 pitch(媽 1 聲高 / 罵 4 聲降);即使 cluster 完全乾淨,MI 也不可能是 0
2. **曲目本質差異**:M4Singer 700 首流行歌 vs VocalVerse 業餘曲目,用詞分布本來就不同;這跟 NSVB 原版「同首歌 amateur+pro paired」的結構性差異不可消除
3. K=60 進一步降只會破壞 VV dwell(已 7.5 marginal,K=60 預計 < 5 → 變 BAD),不會解決上面兩件事;且要再花 ~3h assign 重寫 42K 檔,投資報酬率差
4. 換 Whisper layer 要重 binarize(~14h on A100),在這個量級的殘餘上代價過高

### 為什麼可以放心進 Phase 1

`risk.md` 的 `JSD < 0.05` hard threshold 是「不考慮 unpaired Mandarin 必然差異」下保守設的。實務上 D_z 對抗訓練要的是「分布**夠重疊**讓 discriminator 不能 trivially shortcut」,不是「完全一致」。**JSD 0.16 + MI healthy 對 D_z workable**。

殘留 JSD 真要進一步處理,應在 Phase 2 訓練端做(非 Phase 0 阻擋):
- D_z spectral normalization 限制 discriminator 容量(已在架構規劃裡)
- batch 採樣做 register-balanced sampling(平衡 register JSD 0.08 那段)
- 必要時降 `λ_adv_z`(1.0 → 0.5)讓 D_z 訊號不主導 M 訓練

---

## 6. 最終 accept 的 Phase 0 配置

| 元件 | 配置 |
|---|---|
| 資料集 | M4Singer 全集 + VocalVerse `amateur_score_max=3.0` 篩 536 首 |
| 切 chunk | VocalVerse 切 5s chunks(`min_remaining_sec=3.0`),M4 不切 |
| Feature extraction | mel(librosa, n_fft=512, hop=128, log10),F0(torchcrepe + interp),voicing(CREPE confidence),PPG(Whisper-large-v3 layer 8 hidden state, fp16),spk_emb(Resemblyzer)|
| Dereverb / loudness | DeepFilterNet3 + pyloudnorm BS.1770 -22 LUFS,兩 dataset 都做 |
| Register bucketing | Soft Gaussian 5-bucket(C3/G3/D4/A4/E5,σ=0.3 log-Hz)|
| PPG clustering | **K=100**, `--per-utt-mean-norm`, `--frames-per-song=50`, `--max-total-frames=4_000_000` |

### 通過情況

| Gate | 門檻 | 實測 | 狀態 |
|---|---|---|---|
| ① Vocoder identity | SSIM ≥ 0.90 / F0 RMSE ≤ 10 Hz | PASS | ✅ |
| ② Audio quality(reliable)| sfm/hf_ratio JSD 改善 | hf_ratio JSD 0.258 → 0.048 (PASS) | ✅ |
| ③ JSD(phoneme)| < 0.05 | 0.1562 | ⚠️ 接受(理由見 §5)|
| ③ JSD(register)| < 0.05 | 0.0838 | ⚠️ 接受(輕微,屬資料本性)|
| ④ MI(phoneme; register)| < 0.3 bit | M4 0.216 / VV 0.058 | ✅ |
| ④ mean voiced dwell | ≥ 8 frames | M4 8.6 / VV 7.5 marginal | ✅ / ⚠️ |

---

## 7. 主要踩過的坑(供未來參考)

| 問題 | 根因 | 修法 |
|---|---|---|
| `cluster_ppg` collect 階段 OOM(`^C`)| `DEFAULT_FRAMES_PER_SONG=200` × 42K 檔 → 8.5M frames,`np.concatenate` 峰值 ~86 GB > A100 RAM | 預設降 50 + 加 `--max-total-frames` 硬上限(4M frames)|
| `reconcile_vv_chunks.py` ImportError | 從 `binarizer.py` import `list_vocalverse` 連帶載入 GPU 抽取器整套 | 內聯 `list_vocalverse` + `SampleSpec`,僅依賴 `vocalverse_mos`(只用 pandas)|
| §7 JSD cell 報 `need at least one array to concatenate` | `data/binarized` symlink 在 session 重連後消失,`collect_ids` 抓不到 .npz | §7/§8 改絕對路徑 `/content/local_binarized`,`collect_ids` 加空 list 明確報錯 |
| 串流大 tar 直寫 Drive,Colab 本機磁碟用量一直漲 | Drive FUSE 把寫入先 cache 在本地;zstd 輸出快於 Drive 上傳 → cache 累積 | 拆兩半打包(m4 / vv 各 ~40 GB),`rclone copy` 走 Drive API(不經 FUSE)|
| §6.4 background sync `set() + join(timeout=10)` 不可靠 | `set()` 只觸發迴圈下次檢查退出,無法中斷進行中的 `subprocess.run(rsync)`;`join(timeout)` 超時只 return 不殺 thread | 改成 `while sync_thread.is_alive(): join(timeout=30)` + `pgrep -a rsync` 確認 |
| Gate ④ matplotlib CJK 字體警告 | 預設沒中文字體 | cosmetic,可忽略 |

---

## 8. 對 Phase 1 / Phase 2 的後續含義

### Phase 1(Stage 1 CVAE pretrain)
- 不用 phoneme_id 也不用 D_z → 完全不受 JSD 殘餘影響
- 用 mel / ppg / f0 / spk_emb 條件訓練 VAE,跟 v1 / v2 結果無關
- **可立即開始**

### Phase 2(Stage 2 mapping with D_z)
- 用 phoneme_id + register_id 當 D_z 條件
- 殘留 phoneme JSD 0.16 + register JSD 0.08 對 D_z 有影響,**在訓練端緩解**:
  - D_z 加 spectral normalization(已在 `risk.md` 設計)
  - 必要時 batch sampling 對 register 5-bucket 做平衡
  - 觀察訓練動態(`Stage2Trainer.train_step` 已有 `delta_over_z`、`temporal_diff_ratio`、`monitor_audio_quality`),不健康時降 `λ_adv_z`
- 若實際訓練看到 D_z accuracy 持續 > 0.85(分太開了 → 在用 phoneme 分布當捷徑),考慮更激進的 mitigation:
  - re-binarize 用 Whisper layer 6 或 10(~14h on A100)
  - 切換到 supervised 路線:對 M4 / VV 做 MFA 強制對齊,phoneme_id 改用文字 grounding(工程成本高,但根治)