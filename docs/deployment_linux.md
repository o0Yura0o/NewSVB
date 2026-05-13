# NSVB-ZH 在 Linux 機器上的完整部署指南

> 本文件給「**從沒接觸過 NSVB-ZH 重建**」的人。照著走完，可以從零跑到推理產出 wav。
>
> 配套文件：
> - [rebuild_checklist.md](../rebuild_checklist.md) — 架構決策（為什麼這樣設計）
> - [risk.md](../risk.md) — 風險清單與訓中監控指標
> - [training_flow.md](training_flow.md) — 訓練 pipeline 細節（每階段 input/output shapes）
> - [phase0_colab_workflow.md](phase0_colab_workflow.md) — **替代方案**：Phase 0 在 Colab 跑、Phase 1/2 才回訓練機，省訓練機 16-20h binarize 時間

---

## 0. 系統需求

| 項目 | 最低 | 建議 |
|---|---|---|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| Python | 3.11 | 3.11 |
| GPU | 1× 24 GB VRAM (e.g. RTX 3090, A5000) | 1× 40 GB+ (A100 / H100) |
| CUDA driver | 12.1 | 12.1+ |
| RAM | 64 GB | 128 GB |
| 磁碟 | 500 GB SSD（raw 42 GB + binarized ~100 GB + ckpts ~20 GB + buffer） | 1 TB NVMe |

**估計時間**（單卡 A100）：
- Phase 0（資料前處理）：12–20 小時（含 dereverb + Whisper PPG，~5x realtime）
- Phase 1（Stage 1 預訓練）：2–3 週
- Phase 2（Stage 2 訓練）：2–3 週
- Phase 3（推理）：每首 ~1.5s/秒音訊（含 Whisper PPG + DTW + vocoder）

---

## 1. 取得程式碼

```bash
# 假設 ~/workspace 是工作根目錄
mkdir -p ~/workspace && cd ~/workspace
git clone <你的 NSVB-ZH repo URL> NSVB-ZH
cd NSVB-ZH
```

> 如果是私有 repo，先設好 SSH key 或 HTTPS PAT。

---

## 2. 建立 Python 環境

### 2.1 安裝 Miniconda（若機器上還沒有）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
# 重啟 shell 或 source ~/.bashrc
```

### 2.2 建立 NSVB-ZH 環境

```bash
cd ~/workspace/NSVB-ZH
conda env create -f environment.yml
conda activate NSVB-ZH
```

### 2.3 安裝 PyTorch（CUDA 12.1）

`environment.yml` **不會自動裝 PyTorch**，因為 PyPI 上 cu121 builds 要走 PyTorch 自己的 index。手動裝：

```bash
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 驗證環境

```bash
python -c "
import torch, librosa, transformers, torchcrepe, deepfilternet
from df.enhance import init_df
print(f'torch  : {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'librosa: {librosa.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
"
```

期望輸出：`torch 2.4.1+cu121 cuda=True` 且能列出 GPU 名稱。

---

## 3. 取得預訓練 ckpt（NSVB 作者提供）

NSVB-ZH 重用 NSVB 作者提供的兩個 ckpt：
- `1030_vae_mle/model_ckpt_steps_200000.ckpt`（CVAE backbone，**Phase 1 強烈建議用此初始化**，省 ~1 週訓練）
- `1012_hifigan_all_songs_nsf/model_ckpt_steps_*.ckpt`（HifiGAN-NSF vocoder，**Phase 0 必要**）

```bash
# 從 NSVB 原 repo 下載（路徑見 NSVB README；以下假設 GoogleDrive 下載到本機）
mkdir -p checkpoints/nsvb_1030_vae_mle
mkdir -p checkpoints/1012_hifigan_all_songs_nsf

# 把下載好的 ckpt 放到對應目錄：
#   checkpoints/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt
#   checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt
#   checkpoints/1012_hifigan_all_songs_nsf/config.yaml
```

> 沒有原作者 ckpt 也能跑，但 Phase 1 從零訓 CVAE 要多 ~1 週，且 vocoder 必須自己訓（更花時間）。建議務必拿到。

---

## 4. 下載資料集

### 4.1 VocalVerse（業餘歌聲，929 個 full-length 錄音）

```bash
python scripts/download_vocalverse.py --out-dir data/VocalVerse
# 解壓後約 31 GB；33 位歌手 × 約 28 首 = 929 個錄音，每個 ~3.4 分鐘 full-length
# 下載時間視網速 30–120 分鐘
```

子目錄結構：

```
data/VocalVerse/
├── 443212/                   # 歌曲id（每個目錄一個歌手的所有錄音）
│   ├── 340406604.wav         # 录音id.wav（一首歌全長）
│   ├── 341525169.wav
│   └── ...                    # 該歌手約 28 個錄音
├── 445425/
└── VocalVerse_Datasets-human_labels/
    ├── Amateur_overall_mos_avg5.xlsx                       # 5 位業餘評審 MOS
    └── Professional_multidim_annotations_raw_...xlsx        # 1 位 pro 教練 4-dim 評分
```

### 4.2 M4Singer（職業歌聲，20,896 個 5-sec snippets）

去 [M4Singer 官方頁](https://m4singer.github.io/) 下載 zip，解壓到 `data/m4singer/`：

```
data/m4singer/
├── Alto-1#newboy/
│   ├── 0000.wav             # ~5 秒 snippet
│   ├── 0000.TextGrid
│   └── ...
├── Alto-1#云烟成雨/
└── ...                       # 699 個 {歌手}#{歌名} 子目錄
```

解壓後約 11 GB。每個 `{歌手}#{歌名}/` 子目錄含多個 `*.wav` 與配套 TextGrid
（NSVB-ZH 不使用 TextGrid，但解壓時會一起出來）。

### 4.3 確認結構

```bash
ls data/                           # 應看到 VocalVerse/  m4singer/
du -sh data/*                      # 預期：VocalVerse ~31 GB，m4singer ~11 GB
find data/VocalVerse -name "*.wav" | wc -l    # 預期：929
find data/m4singer -name "*.wav" | wc -l       # 預期：~20896
```

> **時長對照**：VocalVerse 約 **52 h**（929 × ~3.4 min full song），M4Singer 約 **31 h**（21K × ~5.4 sec snippets）。VocalVerse 經 `--vocalverse-amateur-score-max 3.0` 過濾後降至 ~30 h，與 M4Singer 對齊（詳見 [§5.3](#53-binarize-兩個-datasetrisk-2-l1--l2--dereverb--響度正規化)）。

---

## 5. Phase 0 — 資料前處理（gate 階段，必須全部 PASS 才進 Phase 1）

### Risk 2 防線總覽（重要）

NSVB-ZH 最大的訓練風險是 **Risk 2 — 音質域與技術域混淆**：M4Singer（錄音室）vs VocalVerse（user-generated）的錄音環境差異可能讓 M 學成「降噪/去殘響濾波器」而非「修技術」。Phase 0 / Phase 2 的多層機制都是為了防這件事，**全部都已寫進 codebase**：

| 層級 | 在哪裡 | 做什麼 | 預設狀態 |
|---|---|---|---|
| **L1 採樣率 + 響度正規化** | [audio_io.loudness_normalize](../nsvb/utils/audio_io.py) | 22050 Hz / -22 LUFS | 必開（binarize 內建） |
| **L2 DeepFilterNet3 dereverb + denoise** ⭐ | [audio_io.dereverb_wav](../nsvb/utils/audio_io.py) | 對兩 dataset 都跑（不是只 amateur） | **必開**（binarize 預設） |
| **L3 z 層解耦** | Stage 1 CVAE | 錄音環境差異被 mel 層吸收 | 自動 |
| **L4 Phase 0 audio quality probe** ⭐ | [scripts/audio_quality_probe.py](../scripts/audio_quality_probe.py) | SFM / Reverb / HF-ratio / SNR 的 JSD 必須 < 0.10 | **必跑**（gate） |
| **L5 訓中 audio quality monitor** | [Stage2Trainer.monitor_audio_quality](../nsvb/task/stage2.py) | 每 5000 步抽樣計算 unvoiced_concentration | 訓練自動 |

**為什麼 dereverb 對兩個 dataset 都做**：只對 amateur 端 dereverb 會引入新的「兩邊處理不一致」差異，反而讓 D_z 學到「有沒有過 dereverb」這個捷徑。處理後 M4Singer 仍是乾淨的（dereverb 對乾淨輸入近乎 noop），但統計上保證兩邊走過同一條 pipeline。

### 5.1 Vocoder identity test（dealbreaker）

驗證 NSVB 原作者的 HifiGAN 能不能重建中文歌聲 mel。**建議跑兩次**（兩種設定都要 PASS）：

**A. Raw wav baseline**（vocoder ckpt 訓練分布，loud_norm=false）：

```bash
python -m scripts.vocoder_identity_test \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 \
    --save-wavs \
    --out-dir outputs/phase0_vocoder_raw
```

**B. Loud-normed**（驗證 NSVB-ZH binarize 端的實際 mel 分布與 vocoder 兼容）：

```bash
python -m scripts.vocoder_identity_test \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 \
    --save-wavs \
    --apply-loudness-norm \
    --out-dir outputs/phase0_vocoder_loudnormed
```

**通過條件**：**兩次跑**的兩個資料集 verdict 都是 PASS（mel SSIM ≥ 0.90 且 F0 RMSE ≤ 10 Hz）。

> **為什麼跑兩次**：NSVB 原版訓練 vocoder 時 `loud_norm: false`，但 NSVB-ZH binarize 為了 Risk 2（amateur/pro 響度對齊）預設啟用 BS.1770 -22 LUFS。這形成 vocoder 訓練分布 vs 我們實際餵的 mel 分布的潛在不匹配，必須跑 B 確認 vocoder 對 loud-normed mel 仍能 PASS。
>
> A 不過：vocoder 對中文歌聲本身就不行 → fine-tune vocoder 於中文歌聲後再進 Stage 1。
> A 過、B 不過：loud_norm 把 mel 推離 vocoder 訓練分布 → 拿掉 binarize 端 loud_norm（用其他方式緩解 Risk 2 響度差）或 fine-tune vocoder 於 loud-normed mel。
> 兩者皆過：放心進 Phase 1。

### 5.2 Audio quality probe（Risk 2 L4 — 兩 dataset 音質統計差距）

```bash
python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --out-dir outputs/phase0_audio_quality
```

**通過條件**：所有 metric 的 JSD < 0.10：
- **SFM**（spectral flatness）— 估計訊號 vs 白噪音的相似度（高表示噪音含量大）
- **Reverb**（estimated direct-to-reverberant ratio）— 估計殘響量
- **HF-ratio**（high-frequency energy ratio）— 估計頻譜亮度
- **SNR**（estimated SNR via ITU-R P.56）— 估計訊噪比

**不過時的處理**：
1. **首選**：確認 binarizer 是用預設 `dereverb=True`（**不要加 `--no-dereverb`**）；dereverb 通常能把 reverb / SNR JSD 拉進 0.10 內
2. 仍不過：在外部做 SNR 篩選，把 VocalVerse 過糟的樣本拿掉再 binarize
3. 嚴重不過（JSD > 0.20）：考慮換更乾淨的 amateur dataset，或顯著縮小選曲（同類型 / 同錄音環境）

> 跑這支 probe 對 raw wav（dereverb 前），是為了知道**原始**差距；binarize 後實際進 model 的是 dereverb 過的 wav，差距會更小。

### 5.3 Binarize 兩個 dataset（Risk 2 L1 + L2 — dereverb + 響度正規化）

> **⚠️ 重要**：若 [audio_config.py](../nsvb/utils/audio_config.py) 的 `F0_FMIN` / `F0_FMAX` / `HOP_SIZE` / `SAMPLE_RATE` 等 **任一改過**，已存在的 `data/binarized/` 必須**全部重 binarize**（這些常數會 freeze 進 .npz 的特徵值，舊檔不會自動更新）。
> 重 binarize 流程：刪 `data/binarized/{dataset}/` 重跑下面命令。


```bash
# Pro side：M4Singer（注意：不加 --no-dereverb，要對兩邊都做 dereverb）
python -m nsvb.data.binarizer \
    --dataset m4singer \
    --data-root data \
    --out-root data/binarized

# Amateur side：VocalVerse（推薦加多維過濾留「真業餘」）
python -m nsvb.data.binarizer \
    --dataset vocalverse \
    --data-root data \
    --out-root data/binarized \
    --vocalverse-amateur-score-max 3.0
```

#### VocalVerse 多維過濾 ⭐ 推薦

VocalVerse 自帶 **兩份** 標記（[作者論文 arXiv:2512.06999](https://arxiv.org/abs/2512.06999)）：
- `Amateur_overall_mos_avg5.xlsx`：5 位業餘評審平均 1-5 分
- `Professional_multidim_..._.xlsx`：1 位 pro 教練/筆，4 個維度（音色/情感/技巧/氣息控制）1-5 分

929 筆中含「near-pro 業餘」（pro 評分 4-5 接近 pro 水準）；若直接全用，D_z 會看到 z 分布近 pro 的矛盾訊號 → M 修飾訊號變弱。

##### 推薦過濾分數：`amateur_score = (技巧 + 氣息控制) / 2`

只取 pro 4-dim 中跟 vocal mechanics 相關的兩個維度（技巧 + 氣息），**不**用音色（屬 physiological，由 spk_emb 鎖）也**不**用情感（M 不直接訓練）。amateur MOS 與 pro 標記僅 0.38 spearman 相關，做次要 corroborator。

| `amateur_score ≤ X` | 筆數 | per-singer 中位 | 總時長 | 與 M4Singer 30h 對比 |
|---:|---:|---:|---:|---|
| 2.0 | 202 | 6 | 11.1 h | 太少，過擬合 |
| 2.5 | 371 | 11 | 20.6 h | 強過濾 |
| **3.0** | **536** | **17** | **29.8 h** | ⭐ **與 pro 對齊** |
| 3.5 | 676 | 21 | 37.8 h | 含 average，不夠 amateur |
| 不加 flag | 929 | 28 | ~54 h | 含 high-pro 雜訊 |

##### 多維組合（進階）

| 指令 | 結果 |
|---|---|
| `--vocalverse-amateur-score-max 3.0` | 主推 536 / 29.8 h |
| `--vocalverse-amateur-score-max 3.0 --vocalverse-mos-max 3.5` | 440 / 24.6 h（去掉 pro 看差但群眾覺得好聽的雜訊） |
| `--vocalverse-amateur-score-max 2.5` | 371 / 20.6 h（強過濾） |

**Sanity check** 標籤分布（不會跑 binarize）：

```bash
python -m nsvb.data.vocalverse_mos --vocalverse-root data/VocalVerse
```

> 過濾**只影響 binarize 輸出量**——Phase 1/2 訓練程式碼完全不變；max_steps / loss 配方都保留。
> 過濾後 binarize 約少 43% 樣本 → 時間從 ~1.5 h 降到 ~0.85 h。
> `--vocalverse-mos-threshold` 為舊 flag、deprecated 但仍相容（建議改用 `--vocalverse-amateur-score-max`）。

#### Binarize 執行說明

每首歌會產出一個 `data/binarized/{dataset}/{item_id}.npz`。

**時間估計**（A100 GPU；瓶頸是 dereverb 與 Whisper-large-v3 PPG 抽取，~5x realtime）：

| 子集 | 樣本數 | 每樣本長度 | 預估時間 |
|---|---:|---|---:|
| M4Singer | 20,896 | ~5 秒 | ~6–10 h |
| VocalVerse（過濾 amateur_score≤3.0） | 536 | ~3.4 min | ~6–10 h |
| VocalVerse（不過濾） | 929 | ~3.4 min | ~12–18 h |

可以中斷，重跑會自動 skip 已存在的 `.npz`。CPU-only 推理會慢 5-10×。

**磁碟用量**（PPG fp16 [T, 1280] 是大宗）：
- 全 binarized（M4 + VV filtered）：~110 GB
- M4Singer：~46 GB、VocalVerse filtered：~64 GB（每首 ~120 MB × 536）

> **不要加 `--no-dereverb`**——這會違反 Risk 2 主防線。`--no-dereverb` 只在 vocoder identity test、smoke test、或刻意做「無 dereverb」對照實驗時才用。

> binarizer 內部執行順序：`load → dereverb → loudness norm → mel`（[audio_io.load_and_extract](../nsvb/utils/audio_io.py)）。
> dereverb 必須在 loudness norm 之前，否則殘響會被算進 LUFS 統計、dereverb 後音量會偏低。

> 速度監控：開另一個 terminal `watch -n 5 'ls data/binarized/m4singer | wc -l'`。
>
> DeepFilterNet3 model 第一次跑會自動下載到 `~/.cache/DeepFilterNet/`（~50 MB，需要外網）；
> 離線機器要事先在有網路的機器上跑一次 `python -c "from df.enhance import init_df; init_df()"` 然後把 cache 目錄拷貝過去。

### 5.4 PPG k-means → phoneme_id

binarize 完後，跑 k-means 把連續 PPG 分成 200 個 cluster：

```bash
python -m nsvb.data.cluster_ppg \
    --binarized-root data/binarized \
    --centroids-out data/binarized/ppg_kmeans_centroids.npy \
    --k 200 \
    --stage all
```

執行後每個 .npz 多一個 `phoneme_id` key（int16, [T_mel]）。約 30–60 分鐘。

### 5.5 PPG cluster 品質檢查（Phase 0 gate ④）⭐

Whisper 是 speech-heavy 模型，layer 8 雖近 phonetic 但**沒完全把 prosody 抽掉**；對 singing（pitch 是主要變異）會被 k-means 放大成「同一母音不同 pitch → 不同 cluster」，導致 `phoneme_id` 與 `register_id` 強相關 → D_z 兩個條件實質塌成一個（**隱形 F0 shortcut**）。

```bash
python -m scripts.cluster_ppg_inspect \
    --binarized-root data/binarized \
    --datasets m4singer vocalverse \
    --phoneme-vocab-size 200 \
    --out-dir outputs/phase0_cluster_inspect
```

**通過條件**（兩 dataset 都要）：
- `MI(phoneme_id; register_id)` < 0.3 bit（< 0.6 為 marginal）
- `mean_dwell_frames_voiced` ≥ 8（< 3 為 warning）

**腳本同時產出**：
- 抽 5 首歌的 `phoneme_id` + F0 + voicing 三層疊圖（眼睛交叉檢查 sustained 音/vibrato 段 cluster 是否穩定）
- voiced 段 dwell length 分布 histogram
- `report.json` 含完整指標

**WARNING 補救**（依優先級嘗試）：
1. 降 K（`--k 100` 重跑 cluster_ppg）：粗化 cluster 避免 pitch 微差分到不同桶
2. 換 Whisper layer（[audio_config.py](../nsvb/utils/audio_config.py) 改 `WHISPER_HIDDEN_LAYER`）試 6 或 10
3. binarize 時對 PPG 做 per-utterance 去 DC（暫時需手動改 [binarizer.py](../nsvb/data/binarizer.py) 在存 `ppg` 前 `ppg -= ppg.mean(0)`）
4. 對輸出 `phoneme_id` 跑長度 5 mode filter 平滑

### 5.6 JSD 檢查（Phase 0 gate ③）

讓 binarized 資料的 register/phoneme 分布跨 dataset JSD < 0.05（沒有單一 CLI；需要寫小 script 或 jupyter，本文略過。實作上可開 python REPL 對 .npz 統計）。

### 5.7 Phase 0 通過？

```bash
# 確認所有 gate 都過
cat outputs/phase0_vocoder/report.json | grep verdict
cat outputs/phase0_audio_quality/report.json | grep verdict
ls data/binarized/m4singer/*.npz | wc -l   # 應接近 700+
ls data/binarized/vocalverse/*.npz | wc -l # 應接近 7000+
```

---

## 6. Phase 1 — Stage 1 CVAE 預訓練

### 6.1 啟動訓練

**強烈建議從 NSVB 1030 ckpt 初始化**（省 ~1 週訓練）：

```bash
python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --init-from-nsvb checkpoints/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --max-steps 80000 \
    --batch-size 16 \
    --num-workers 4 \
    --ckpt-dir checkpoints/stage1
```

或從零訓（需 ~200k 步）：
```bash
python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --max-steps 200000 \
    --ckpt-dir checkpoints/stage1
```

### 6.2 監控訓練進度（terminal）

訓練 log 會直接印到 stdout，含 tqdm 進度條 + 每 50 步的 loss 摘要：

```
stage1: 12%|############| 9500/80000 [3:42:11<27:30:18, 0.71it/s, l_total=0.234, kl=0.046]
[step   9500 ep14 0.71it/s] l_total=0.2336 l_l1=0.1124 l_l2=0.0143 kl=0.0462 kl_beta=0.0095 ...
[stage1] ckpt saved: checkpoints/stage1/stage1_step10000.pt
[stage1] ckpt saved: checkpoints/stage1/stage1_latest.pt
```

**長時間訓練建議用 tmux/screen**，避免 ssh 斷線：

```bash
tmux new -s nsvb-zh-stage1
# 在 tmux 內執行訓練命令
# 按 Ctrl+B 然後 D 離開（不會中斷訓練）
# 重連：tmux attach -t nsvb-zh-stage1
```

或開另一個 terminal 即時看 log：

```bash
# 把訓練輸出 redirect 到檔案
python -m nsvb.task.stage1 ... 2>&1 | tee logs/stage1_$(date +%Y%m%d_%H%M%S).log
# 另一個 terminal：
tail -f logs/stage1_*.log
```

### 6.3 ckpt 存檔機制

訓練每 5000 步（`--save-interval` 可調）會存：
- `checkpoints/stage1/stage1_step{N}.pt`（永久保留）
- `checkpoints/stage1/stage1_latest.pt`（每次覆蓋，方便 resume）

訓練結束時多存：
- `checkpoints/stage1/stage1_final.pt`

### 6.4 中段續訓

斷線、機器重啟、調參想接著跑都用 `--resume`：

```bash
# 從最新 ckpt 接著跑
python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --max-steps 80000 \
    --ckpt-dir checkpoints/stage1 \
    --resume latest
```

`--resume latest` 是 `--resume checkpoints/stage1/stage1_latest.pt` 的簡寫。也可指定特定步：

```bash
--resume checkpoints/stage1/stage1_step50000.pt
```

恢復內容：model + optimizer state + step + epoch。

### 6.5 通過條件

訓練到 loss 穩定（`l_total` 不再下降約 5000 步）即可進 Phase 2。經驗值：
- `l_l1 < 0.15`
- `l_l2 < 0.025`
- 用 [scripts/infer.py](../scripts/infer.py) 對幾首歌跑 a→a 重建，聽起來品質與原版接近。

---

## 7. Phase 2 — Stage 2 Mapping 訓練（最關鍵階段）

### 7.1 啟動訓練

```bash
python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --amateur-dataset vocalverse \
    --pro-dataset m4singer \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --max-steps 120000 \
    --batch-size 16 \
    --num-workers 4 \
    --ckpt-dir checkpoints/stage2
```

### 7.2 進度監控（同 Stage 1）

```
stage2: 12%|############| 14400/120000 [2:11:42<16:08:50, 1.81it/s, m=0.821, d_z=1.247, Δ/z=0.038]
[step  14400 1.81it/s] m_total=0.8214 d_z=1.2467 d_mel=0.9133 l_nce=0.5821 l_adv_z=0.2412 ... delta_over_z=0.0381
[stage2-monitor step 15000] Δ_voiced_E=0.0124  Δ_unvoiced_E=0.0061  unvoiced_concentration=0.330  (< 0.55 良好, ...)
[stage2] ckpt saved: checkpoints/stage2/stage2_step15000.pt
```

**監控重點**（參考 [risk.md](../risk.md) Monitor 1-5）：
- `d_z` accuracy（在 0.55–0.75 內代表 G/D 平衡良好；過高表 M 學不贏 D_z，過低表 M 走捷徑）
- `delta_over_z`（‖Δ‖/‖z‖；不應 > 0.30，否則 M 改太多）
- `unvoiced_concentration`（每 5000 步的音質監控；< 0.55 良好，> 0.65 警訊代表 M 在去殘響）

#### 7.2.1 訓中音質監控細節（Risk 2 L5）

[Stage2Trainer.monitor_audio_quality](../nsvb/task/stage2.py)（每 `audio_quality_monitor_interval` 步=預設 5000）會：
1. 抽 `audio_quality_monitor_n_samples`（預設 4）個 amateur 樣本
2. 計算 `Δ_mel = mel_modified - mel_baseline`（modified = 過 M；baseline = 不過 M）
3. 比較 voiced 段 vs unvoiced 段 Δ 能量分布
4. 報告 **`unvoiced_concentration = unvoiced_E / (voiced_E + unvoiced_E)`**

**判讀**（risk.md Risk 2 補強 4）：
| 範圍 | 含義 | 該怎辦 |
|---|---|---|
| < 0.55 | M 修飾集中在 voiced 段 — 真技術修正 | 繼續訓 |
| 0.55–0.65 | marginal | 留意，再觀察 5000 步 |
| > 0.65 | **Risk 2 警訊** — M 在去殘響/降噪 | 停下檢查；可能要降 `lambda_adv_mel`，或重新確認 binarize 時 dereverb 有開 |

每次抽樣會把 mel spectrogram 對比存成 `.npz` 到 `checkpoints/stage2/audio_monitor/step{N}_sample0.npz`，包含：
- `mel_gt` — 原 amateur mel
- `mel_baseline` — z 不過 M 的 decoder 輸出
- `mel_modified` — z 過 M 的 decoder 輸出
- `delta_mel` — 兩者差
- `f0`, `voiced`, `unvoiced_concentration`

可用 matplotlib 視覺化檢查 M 的修飾集中在哪些 frame：

```bash
python -c "
import numpy as np, matplotlib.pyplot as plt
d = np.load('checkpoints/stage2/audio_monitor/step015000_sample0.npz')
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axs[0].imshow(d['mel_gt'].T, origin='lower', aspect='auto');     axs[0].set_title('GT mel')
axs[1].imshow(d['mel_modified'].T, origin='lower', aspect='auto'); axs[1].set_title('M(z) decoded')
axs[2].imshow(d['delta_mel'].T, origin='lower', aspect='auto', cmap='RdBu'); axs[2].set_title(f'Δ (uv_conc={d[\"unvoiced_concentration\"]:.3f})')
plt.tight_layout(); plt.savefig('audio_monitor_step15000.png', dpi=100)
"
```

> 不想監控可調 `audio_quality_monitor_interval` 設大數值（例如 999999999）讓它實質不跑；但**強烈不建議關**，這是 Risk 2 訓中唯一警報機制。

### 7.2.2 訓中自動健康檢查

[Stage2Trainer.fit](../nsvb/task/stage2.py) 還有兩個自動 fallback 警告（不會中斷訓練，只 print 提示）：

**A. M kernel_size=1 太保守警告**
- 觸發條件：step ≥ `delta_health_check_step`（預設 30000）後，`‖Δ‖/‖z‖` 移動平均仍 < `delta_health_check_threshold`（預設 0.03）
- 一次性訊息：「M 可能太保守，無法生成顫音/滑音等時間軸動態。建議：以 `--m-kernel-size 3` 重新訓練」
- 處理：停下來看是否認同，認同就 resume + `--m-kernel-size 3` 重訓（**注意**：M 結構改了，舊 ckpt 的 M weight 不能直接 load；要從 stage1 ckpt 重啟 stage2）

**B. PatchNCE T_z 過小警告**
- 觸發條件：`T_z < 4`（通常表示 `--max-frames` 設太小，或 batch 內全是極短 sample）
- 一次性 RuntimeWarning：建議 `max_frames ≥ 64`

### 7.2.3 Risk 2 救火選項（D_mel fallback）

若 [§7.2.1](#721-訓中音質監控細節risk-2-l5) 的 `unvoiced_concentration` 持續 > 0.65（連兩次採樣），代表 M 在學去殘響。處理：

```bash
# 停下來，加 --dmel-mix-amateur-real 重 resume：
python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --resume latest \
    --dmel-mix-amateur-real \
    --ckpt-dir checkpoints/stage2
```

`--dmel-mix-amateur-real` 把 D_mel real 改餵 pro+amateur 混合，犧牲 mel 層 pro-direction 訊號換取「絕不鼓勵去殘響」的安全。

> **建議**：先確認 binarize 是否正確開了 dereverb（[§5.3](#53-binarize-兩個-datasetrisk-2-l1--l2--dereverb--響度正規化)），如果 dereverb 沒做好，啟此 flag 也救不了根本問題。

### 7.3 ckpt 機制 / 中段續訓（同 Stage 1）

```bash
python -m nsvb.task.stage2 \
    --binarized-root data/binarized \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --resume latest \
    --ckpt-dir checkpoints/stage2
```

恢復內容：M / D_z / D_mel / PatchNCE proj head + 三個 optimizer + step。

> CVAE backbone 不在 ckpt 內，每次啟動都從 `--stage1-ckpt` 重新載入並凍結。

### 7.4 通過條件

`m_total` loss 穩定 ~ 5000 步，`unvoiced_concentration` 始終 < 0.55，且抽樣推理聽起來：
- 業餘歌聲音準變好、共鳴變豐富
- 音色仍是業餘歌手本人（沒被換成 pro）

---

## 8. Phase 3 — 推理

### 8.1 Mode A（純自動，最常用）

業餘錄音 → 修飾後輸出（與輸入同長度，可直接配原伴奏）：

```bash
python -m scripts.infer \
    --stage2-ckpt checkpoints/stage2/stage2_latest.pt \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --input-a path/to/amateur.wav \
    --output outputs/mode_a_result.wav
```

### 8.2 Mode B（用 pro reference 做節奏 + 音高模板）

```bash
python -m scripts.infer \
    --stage2-ckpt checkpoints/stage2/stage2_latest.pt \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --input-a path/to/amateur.wav \
    --pro-ref path/to/pro_reference.wav \
    --output outputs/mode_b_result.wav
```

> Mode B 輸出長度 = pro ref 長度（**不等於** amateur 長度），需配 pro ref 的伴奏。

### 8.3 推理端 dereverb（Risk 2 推理對齊）

[scripts/infer.py](../scripts/infer.py) 預設對輸入做 dereverb + loudness norm，與訓練 binarize 對齊（避免分布偏移讓 M 看到沒見過的音質域）。
若 user 提供的是已是乾淨 studio 錄音，加 `--no-dereverb` 跳過：

```bash
python -m scripts.infer ... --input-a clean_studio.wav --no-dereverb --output result.wav
```

> **不確定就保持預設**。dereverb 對乾淨輸入近乎 noop（DeepFilterNet3 偵測無 reverb 時改動極小），但對 user 隨手錄的手機音檔關鍵。

### 8.4 跨機器推理（Stage 1 ckpt 路徑不同時）

Stage 2 ckpt 內紀錄了訓練機上的 Stage 1 路徑；遷到別機需顯式覆寫：

```bash
python -m scripts.infer \
    --stage2-ckpt checkpoints/stage2/stage2_latest.pt \
    --stage1-ckpt checkpoints/stage1/stage1_latest.pt \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --input-a path/to/amateur.wav \
    --output outputs/result.wav
```

---

## 9. 故障排除

### 9.1 OOM（GPU 記憶體不足）

降 `--batch-size` 與 `--max-frames`：
```bash
--batch-size 8 --max-frames 400
```
或縮 PPG 維度（用 `whisper-medium` 而非 `whisper-large-v3`，需在 [audio_config.py](../nsvb/utils/audio_config.py) 調 `WHISPER_MODEL_NAME` 與重 binarize；不建議）。

### 9.2 DataLoader 卡住

Linux 預設 fork 在 CUDA 下不安全。已知症狀：訓練啟動後第一個 batch 後 hang。
解法：開 trainer 之前加 `mp.set_start_method('spawn', force=True)`，或設 `--num-workers 0` 暫時繞開。

### 9.3 DeepFilterNet 載入失敗（重要）

第一次跑 binarizer 會自動下載 DeepFilterNet3 model 到 `~/.cache/DeepFilterNet/`；網路不穩會卡。處理方式：

1. **首選**：在有網路的機器跑一次 `python -c "from df.enhance import init_df; init_df()"`，把 `~/.cache/DeepFilterNet/` 整包拷貝到目標機器的同位置
2. 設代理：`export HTTPS_PROXY=...`
3. **不得已**才用 `--no-dereverb` — 會違反 Risk 2 主防線（L2），失去訓練最重要的音質域對齊機制；**會直接讓 M 學成去殘響濾波器**，僅 vocoder identity test / smoke test 可用，production 訓練嚴禁

### 9.4 Whisper 模型下載慢

預先設好 HF cache 並下載：
```bash
export HF_HOME=/path/to/large/cache
python -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-large-v3')"
```
之後 binarize 與 inference 都用此 cache。

### 9.5 NSVB ckpt 路徑找不到（推理時）

`--stage1-ckpt` 與 `--vocoder-ckpt` 都用絕對路徑或相對於目前工作目錄的路徑。檢查：
```bash
ls -la checkpoints/stage1/stage1_latest.pt
ls -la checkpoints/1012_hifigan_all_songs_nsf/
```

---

## 10. 跨平台守則（已在 codebase 強制）

訓練在 Linux 跑，本地開發在 Windows 也都該過。NSVB-ZH 的程式遵守：

1. **路徑全用 `pathlib.Path`** — `os.path` 已淘汰
2. **檔案 I/O 一律 `encoding="utf-8"`** — Windows 預設 cp950，Linux 預設 utf-8
3. **不用 `os.system`/`subprocess` 跑 bash 內建命令** — 改用 `pathlib`/`shutil`
4. **CUDA device 字串只用 `"cuda"`** — 不寫 `"cuda:0"`，多卡時 framework 自動分配
5. **HF cache 用 `HF_HOME` 環境變數控制**
6. **無 hardcode 路徑**

詳見 [rebuild_checklist.md §J](../rebuild_checklist.md)。

---

## 11. 文件導讀清單（按閱讀順序）

| 順序 | 文件 | 一句話描述 |
|---|---|---|
| 1 | 本文 (`deployment_linux.md`) | 怎麼從零跑完所有流程 |
| 2 | [training_flow.md](training_flow.md) | 每個訓練階段的 input/output 細節 |
| 3 | [rebuild_checklist.md](../rebuild_checklist.md) | 為什麼這樣設計（架構決策） |
| 4 | [risk.md](../risk.md) | 風險清單與訓中監控指標 |