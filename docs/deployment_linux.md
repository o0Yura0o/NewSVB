# NSVB-ZH 在 Linux 機器上的完整部署指南

> 本文件給「**從沒接觸過 NSVB-ZH 重建**」的人。照著走完，可以從零跑到推理產出 wav。
>
> 配套文件：
> - [rebuild_checklist.md](../rebuild_checklist.md) — 架構決策（為什麼這樣設計）
> - [risk.md](../risk.md) — 風險清單與訓中監控指標
> - [training_flow.md](training_flow.md) — 訓練 pipeline 細節（每階段 input/output shapes）

---

## 0. 系統需求

| 項目 | 最低 | 建議 |
|---|---|---|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| Python | 3.11 | 3.11 |
| GPU | 1× 24 GB VRAM (e.g. RTX 3090, A5000) | 1× 40 GB+ (A100 / H100) |
| CUDA driver | 12.1 | 12.1+ |
| RAM | 64 GB | 128 GB |
| 磁碟 | 1 TB SSD（dataset + binarize 後 PPG fp16 ~660 GB） | 2 TB NVMe |

**估計時間**（單卡 A100）：
- Phase 0（資料前處理）：8–14 小時
- Phase 1（Stage 1 預訓練）：2–3 週
- Phase 2（Stage 2 訓練）：2–3 週
- Phase 3（推理整合）：~1 小時

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

### 4.1 VocalVerse（業餘歌聲，~7600 首）

```bash
python scripts/download_vocalverse.py --out-dir data/VocalVerse
# 約 80 GB，下載時間視網速 30–120 分鐘
```

### 4.2 M4Singer（職業歌聲，~700 首）

去 [M4Singer 官方頁](https://m4singer.github.io/) 下載 zip，解壓到 `data/m4singer/`：

```
data/m4singer/
├── Alto-1#newboy/
│   ├── 0000.wav
│   ├── 0000.TextGrid
│   └── ...
├── Alto-1#云烟成雨/
└── ...
```

每個 `{歌手}#{歌名}/` 子目錄含多個 `*.wav` 與配套 TextGrid（NSVB-ZH 不使用 TextGrid，但解壓時會一起出來）。

### 4.3 確認結構

```bash
ls data/  # 應看到 VocalVerse/  m4singer/
du -sh data/*  # VocalVerse 約 80 GB，m4singer 約 30 GB
```

---

## 5. Phase 0 — 資料前處理（gate 階段，必須全部 PASS 才進 Phase 1）

### 5.1 Vocoder identity test（dealbreaker）

驗證 NSVB 原作者的 HifiGAN 能不能重建中文歌聲 mel：

```bash
python -m scripts.vocoder_identity_test \
    --vocoder-ckpt checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1512000.ckpt \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 \
    --save-wavs \
    --out-dir outputs/phase0_vocoder
```

**通過條件**：兩個資料集的 verdict 都是 PASS（mel SSIM ≥ 0.90 且 F0 RMSE ≤ 10 Hz）。

> 不過則 vocoder 要在中文歌聲上 fine-tune（不在本指南範圍）；不過不能跳過——後續 M 的所有改進都會被 vocoder 重建誤差掩蓋。

### 5.2 Audio quality probe（兩 dataset 音質統計差距）

```bash
python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --out-dir outputs/phase0_audio_quality
```

**通過條件**：所有 metric (SFM / Reverb / HF-ratio / SNR) 的 JSD < 0.10。

> 不過則 binarizer 一定要開 dereverb（預設就開）；嚴重不過時要在外部做 SNR 篩選把過糟的 amateur 樣本拿掉。

### 5.3 Binarize 兩個 dataset

```bash
# Pro side：M4Singer
python -m nsvb.data.binarizer \
    --dataset m4singer \
    --data-root data \
    --out-root data/binarized

# Amateur side：VocalVerse
python -m nsvb.data.binarizer \
    --dataset vocalverse \
    --data-root data \
    --out-root data/binarized
```

每首歌會產出一個 `data/binarized/{dataset}/{item_id}.npz`。**會跑很久**（~3-5 秒/歌 × 8000 首 ≈ 8–14 小時）。可以中斷，重跑會自動 skip 已存在的。

> 速度監控：開另一個 terminal `watch -n 5 'ls data/binarized/m4singer | wc -l'`。

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

### 5.5 JSD 檢查（Phase 0 gate ②）

讓 binarized 資料的 register/phoneme 分布跨 dataset JSD < 0.05（沒有單一 CLI；需要寫小 script 或 jupyter，本文略過。實作上可開 python REPL 對 .npz 統計）。

### 5.6 Phase 0 通過？

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

### 8.3 跨機器推理（Stage 1 ckpt 路徑不同時）

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

### 9.3 DeepFilterNet 載入失敗

第一次跑 binarizer 會自動下載 DeepFilterNet3 model 到 `~/.cache/DeepFilterNet/`；網路不穩會卡。離線機器可預先下載，或用 `--no-dereverb` 關掉（**會違反 Risk 2 主防線**，僅 smoke test 可用）。

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