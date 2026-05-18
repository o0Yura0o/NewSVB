# NSVB-ZH 後續訓練計畫(無人監督版)

> 受眾:在另一台(Windows)機器上代管訓練的 agent。
> 前置:Stage 2 v2 已 lock 為 baseline(見 [phase2_outcome.md](phase2_outcome.md))。
> 本文件給 **5 個獨立 plan + 共用前置 + 共用 eval + 決策樹**,agent 依此執行。
>
> **核心約束**:
> - 不修改原 dataset(M4Singer raw / VocalVerse raw 不動)
> - 預設不重訓 Stage 1(Plan C 例外,因 audio 分布變了)
> - 每 plan 都用 mel-domain 指標驗收(vocoder 問題另外處理,不阻塞模型訓練判定)
> - 每 plan 都需先做 **smoke test**(本機 Win CPU 5-10 min 跑通 100 steps),通過再上 remote 全訓

## 目錄

| § | 內容 |
|---|---|
| [0](#0-總覽--決策樹) | 總覽 + 決策樹 |
| [1](#1-共用前置共資料機器準備) | 共用前置:資料 / 機器準備 |
| [2](#2-plan-a-v3-loss-tuning-baseline) | Plan A:v3 loss tuning(baseline)|
| [3](#3-plan-b-f0-support) | Plan B:f0_support |
| [4](#4-plan-c-換-dereverb-backend--重-binarize) | Plan C:換 dereverb backend + 重 binarize |
| [5](#5-plan-d-pro-distribution-matching-loss) | Plan D:pro-distribution matching loss |
| [6](#6-plan-e-m-架構放大) | Plan E:M 架構放大 |
| [7](#7-共用-eval--score-sheet) | 共用 eval / score sheet |
| [8](#8-handoff-checklist) | Handoff checklist |

---

## 0. 總覽 / 決策樹

### 5 個 plan 對照

| Plan | 改動性質 | 對應 tmp_ref 方向 | 預估時間 (A100) | 風險 |
|---|---|---|---|---|
| **A. v3 loss tuning** | hyperparam (loss 權重 + 早停)| (5)模型參數調整 | ~5h | 低 |
| **B. f0_support** | 訓中 fake-mel decode 用平滑 F0 | (2)pitch 條件 + (4)新 loss | ~7h | 中 |
| **C. 換 dereverb backend** | 重 binarize 兩 dataset | (1)更強去殘響 + (3)拉近 pro 分布 + (6)資料前處理 | ~20h | 中-高 |
| **D. Pro-distribution matching loss** | 新 G loss | (3)拉近 pro + (4)新 loss | ~8h | 中 |
| **E. M 架構放大** | model size | (5)模型參數調整 | ~5h | 低 |

### 推薦執行順序

```
                 [Plan A]                  ← 必做(baseline,快,只動 hyperparams)
                    │
        ┌───────────┼───────────┐
        │           │           │
     [Plan E]    [Plan B]    [Plan D]      ← 可並行,各自獨立(都基於 v2 stage1_best.pt)
        │           │           │
        └─── 各自跑完 → 對照 7 §score sheet ───┘
                    │
                    ▼
              選最佳 → 鎖為 v3
                    │
                    ▼
           測 pro_direction_alignment
            提升 >= v2 +0.05?
                    │
            ┌───────┴───────┐
           是               否
            │                │
       v3 lock            [Plan C]         ← 殺手鐧:換 dereverb 重 binarize
                              │              (~20h,動資料側)
                              ▼
                          重訓 Stage 1+2
                              │
                              ▼
                       仍未達標 → 結案,寫 final report
                              │
                              ▼
                       達標 → v3 lock
```

**為什麼這個順序**:
- A 風險最低、改動最小、最快給答案 — 必做 baseline
- E 是配套(arch 改動不衝突 A 的 hyperparam)— 跟 A 平行
- B / D 是新增 loss / 訓練機制 — 平行 A/E 跑,因為都基於同一個 stage1_best.pt
- C 是 nuclear option:重 binarize → 重訓 stage1 → 重訓 stage2,投資最大,留到最後

**所有 plan 平行可行的硬條件**:
- 4 個 GPU(每 plan 一個);或
- 1 個 GPU 排隊跑(plan A 先,~5h;再 plan E ~5h;再 B ~7h;再 D ~8h ≈ 25h)。Plan C 視結果決定。

---

## 1. 共用前置(資料 / 機器準備)

所有 plan 共用這套前置。**只跑一次**。

### 1.1 在新機器(Windows)上 clone repo 並裝環境

```powershell
# Windows PowerShell
# git
git clone https://github.com/<your-org>/NSVB-ZH C:\NSVB-ZH
cd C:\NSVB-ZH
git checkout (Get-Content COMMIT_HASH.txt)

# Python 3.10+, PyTorch with CUDA
# 假設機器有 conda
conda create -n nsvbzh python=3.10 -y
conda activate nsvbzh
pip install -r requirements.txt
# 或手動裝(看 Requirements.txt 或 NSVB 原版的 environment.yml 對照)
```

### 1.2 從 Drive 公開分享連結下載資料 + ckpts

本機**沒有 rclone / GD2CLL 設定**,改走 `gdown` 吃 Google Drive 公開分享連結。
user 會在交付時提供下列分享連結。

```powershell
# 一次性裝 gdown
pip install gdown

# 下載 binarized v2(壓縮 tar.zst,~75 GB)
# user 提供 share_id / share_url(例如 https://drive.google.com/file/d/<ID>/view?usp=share_link)
$BINARIZED_TAR_ID = "<user_provides>"
gdown "https://drive.google.com/uc?id=$BINARIZED_TAR_ID" -O data\binarized_v2.tar.zst

# 解壓到 data\binarized_v2\(zstd 需先裝;Win 可裝 7-Zip 或 zstd Win build)
# 推薦走 Python:
pip install zstandard
python -c "import zstandard, tarfile; `
    dctx = zstandard.ZstdDecompressor(); `
    with open('data/binarized_v2.tar.zst', 'rb') as ifh: `
        with dctx.stream_reader(ifh) as zr: `
            with tarfile.open(fileobj=zr, mode='r|') as tf: `
                tf.extractall('data/')"
# tar 內結構是 binarized/{m4singer,vocalverse,ppg_kmeans_centroids.npy}
# 解開後路徑會是 data\binarized_v2\... — 我們改名統一為 data\binarized_v2
Rename-Item data\binarized data\binarized_v2

# 下載 v2 ckpts (stage1_best.pt + stage2_v2 全 ckpts,~3 GB)
$CKPTS_ZIP_ID = "<user_provides>"
gdown "https://drive.google.com/uc?id=$CKPTS_ZIP_ID" -O checkpoints_v2.zip
Expand-Archive checkpoints_v2.zip -DestinationPath .

# 下載 vocoder ckpt(NSVB pretrained,~150 MB)— 後續 eval / Plan C 用到
$VOCODER_ID = "<user_provides>"
gdown "https://drive.google.com/uc?id=$VOCODER_ID" -O `
    checkpoints\1012_hifigan_all_songs_nsf\model_ckpt_steps_1170000.ckpt
```

> ⚠ **三個 share ID 由 user 在交付時提供**,agent 不可自己亂猜或產生 URL。
> 若 gdown 因為「too many requests」失敗,等 10-30 min 再試,或改用 user 給的備援連結。

### 1.3 確認資料完整

```powershell
# 確認 splits 跟兩 dataset 都在
Get-ChildItem C:\NSVB-ZH\data\binarized_v2
# 應看到 m4singer/ vocalverse/ ppg_kmeans_centroids.npy splits/

# 確認 Stage 1 ckpt
Test-Path C:\NSVB-ZH\checkpoints_v2\stage1\stage1_best.pt
# 應 True

# 確認 vocoder ckpt
Test-Path C:\NSVB-ZH\checkpoints\1012_hifigan_all_songs_nsf\model_ckpt_steps_1170000.ckpt
```

### 1.4 GPU smoke test

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期: CUDA: True, <GPU model>
```

### 1.5 訓練/結果存放策略

本機**不會 session 中斷**,所有 ckpts / logs / 中間 eval 結果**全留本地**,**不用背景 sync 回 Drive**。
只在每 plan 結束後上傳「重要結果」(見 §7.5):

| 資源 | 存放 |
|---|---|
| 訓中 ckpts `stage2_step{N}.pt`(每 5K 一個) | 本地 only,訓完不上傳 |
| 訓中 audio_monitor `.npz` | 本地 only,debug 用 |
| 訓中 raw log | 本地 only |
| `stage2_best.pt`(早停 hook 自動產出) | **上傳** |
| `stage2_final.pt`(=最後一步) | **上傳** |
| `outputs/.../report.md`(eval 報告)| **上傳** |
| `outputs/.../mel_grid.png` 子集(代表性 5-10 張)| **上傳** |
| `runs/stage2_v3_plan{X}.summary.md`(log summary) | **上傳** |

---

## 2. Plan A: v3 loss tuning (baseline)

### 2.1 改了什麼(對照 v2)

| 項目 | v2 | Plan A | 動機 |
|---|---|---|---|
| `max_steps` | 120000 | **50000** | best @ 30K |
| `lambda_patchnce` | 1.0 | **2.0** | 強化 content anchor |
| `lambda_identity_pro` | 0.1 | **0.2** | 強化 anti-漂移 |
| `lambda_adv_mel` | 0.05 | **0.1** | mid-ground |
| `freeze_d_mel` | True | **False** | 解凍 D_mel |
| `d_z_warmup_steps` | 5000 | **10000** | 多給 PatchNCE 打底 |
| 早停 hook | 無 | `--val-eval-interval 5000` | 自動存 best ckpt |

### 2.2 Smoke test(Win 本機 CPU,~10 min)

```powershell
# 跑 100 步驗證 pipeline 通
$env:PYTHONPATH = "."
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints\smoke_a `
    --split-dir data\binarized_v2\splits `
    --max-steps 100 `
    --batch-size 4 `
    --num-workers 0 `
    --device cpu `
    --lambda-patchnce 2.0 `
    --lambda-identity-pro 0.2 `
    --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 50 `
    --val-eval-n-samples 4
```

通過條件:跑完 100 步無 crash,log 看到 `[val-eval step 50]` 跟 `[val-eval step 100]` 各印一次。

### 2.3 全訓(remote GPU,~5h on A100 / ~10h on consumer GPU)

```powershell
$env:PYTHONPATH = "."
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planA `
    --split-dir data\binarized_v2\splits `
    --max-steps 50000 `
    --batch-size 16 `
    --num-workers 4 `
    --lambda-patchnce 2.0 `
    --lambda-identity-pro 0.2 `
    --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 `
    --val-eval-n-samples 20 `
    2>&1 | Tee-Object -FilePath logs\stage2_v3_planA.log
```

訓中可觀察 `[val-eval step N] pro_dir_align=+0.XX  ⭐ new best` 行,確認 val alignment 上升。

### 2.4 Eval(訓完跑)

走 §7 共用 eval 流程,輸出比較表。

---

## 3. Plan B: f0_support

### 3.1 改了什麼

對 fake mel decode(D_mel real-vs-fake 那條 path)餵 **smoothed F0**,讓 D_mel 不能用「F0 jitter」當 amateur 簽名。

**理論**:Risk §二.3 confound — D_mel real 是 pro mel(由 pro F0 生)、fake 是 decoder 出(由 amateur F0 生),兩者 F0 jitter 程度天然不同 → D_mel 用 F0 trajectory 平滑度當判別捷徑 → M 學不到真 pro 化。

**做法**:用 median filter / Savgol filter 在 log-F0 空間平滑 amateur F0,只在 fake mel decode 那條 path 餵 smoothed 版。encoder 仍餵 raw F0(它是 frozen 不能換 distribution)。

**已實作模組**:[nsvb/data/f0_smoothing.py](../nsvb/data/f0_smoothing.py)

**已知 trade-off**:訓中 decoder 看到 smoothed F0、推理時看到 raw F0 → distribution shift。**緩解**:Mode A 推理時也加同樣 smoothing(`--no-f0-interp` 之後再加 F0 平滑步驟,但目前 Mode A 不平滑 F0 trajectory,只在 vocoder 前 interp unvoiced)。本 Plan 接受 train/infer mismatch,看 mel-domain 指標是否仍進步。

### 3.2 Smoke test(Win 本機 CPU,~10 min)

```powershell
$env:PYTHONPATH = "."
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints\smoke_b `
    --max-steps 100 `
    --batch-size 4 `
    --num-workers 0 `
    --device cpu `
    --f0-support median `
    --f0-support-window 5 `
    --val-eval-interval 50 `
    --val-eval-n-samples 4
```

通過條件:`[stage2] f0_support: median window=5` 在 init log 出現 + 跑完 100 步無 crash。

### 3.3 全訓(remote, ~7h on A100)

```powershell
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planB `
    --max-steps 50000 `
    --batch-size 16 --num-workers 4 `
    --f0-support median --f0-support-window 5 `
    --lambda-patchnce 2.0 `
    --lambda-identity-pro 0.2 `
    --lambda-adv-mel 0.2 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 --val-eval-n-samples 20 `
    2>&1 | Tee-Object -FilePath logs\stage2_v3_planB.log
```

注意:Plan B 也跟 v3 loss tuning 同時用(共用 Plan A 的 lambda 調整);差異在多 `--f0-support median`,並把 `lambda_adv_mel` 從 0.1 → 0.2(因為 D_mel 不再有 F0 捷徑,可以給更多 mel adv 訊號)。

### 3.4 Plan B 特定觀察

訓中追蹤 `l_adv_mel`:
- v1: ~5(F0 confound floor)
- v2: ~0.8(freeze D_mel)
- **Plan B 預期**:1-3,介於 v1/v2 之間,代表 D_mel 仍有效但不被 F0 捷徑卡住

若 `l_adv_mel` 仍 floor 在 ~5 → f0_support 沒效,confound 來源不只 F0 jitter。

---

## 4. Plan C: 換 dereverb backend + 重 binarize

> ⚠ **最貴的 plan,只有 A/B/D/E 都未達標才動**。需 ~20h 全程,且改 binarized 等於改訓練分布,要重訓 Stage 1。

### 4.1 改了什麼

**假設**:VocalVerse 跟 M4Singer 的 mel 分布差距,主要來自 VV 殘留的 reverb / 環境噪音(DF3 沒清乾淨)。把 dereverb 換成更強的 backend → VV mel 分布拉近 M4 → vocoder 可能也對 VV 友好(解 Risk 10 vocoder 不熟分布問題)。

**已實作 scaffold**:[nsvb/utils/audio_io.py](../nsvb/utils/audio_io.py) `dereverb_wav(wav, backend=...)`,目前支援:
- `df3`(預設,v2 用)
- `df3_cascade`(DF3 跑兩次,**內建,可直接用**)— 適合 VV 重 reverb
- `demucs`(stub,**需要 agent 先 `pip install demucs` 並補 backend 實作**)
- `voicefixer`(stub,同上)

[nsvb/data/binarizer.py](../nsvb/data/binarizer.py) 已新增 `--dereverb-backend` CLI flag。

### 4.2 Plan C 兩條子路徑

**C-a:用 df3_cascade(零實作成本)**
- 直接重 binarize 兩 dataset
- 跑 audio_quality_probe 確認 hf_ratio / sfm JSD 改善
- 重訓 Stage 1(因為 VV mel 變了 → encoder 要重學)
- 重訓 Stage 2

**C-b:用 demucs 或 voicefixer(需先實作 backend)**
- agent 先在 [`nsvb/utils/audio_io.py`](../nsvb/utils/audio_io.py) `dereverb_wav` 內補新 backend(模仿 `df3` block 那段)
- 寫個 standalone smoke test(`scripts/test_dereverb_backend.py`,本文件 §4.4 提供 spec)確認新 backend on VV wav 效果好
- 同 C-a 重 binarize + 重訓

### 4.3 Smoke test(Win 本機,~30 min,需 wav 樣本)

```powershell
# 在本機抽 1-2 個 VV raw wav,試新 backend 是否能跑通 + 聽感對嗎
python scripts/audio_quality_probe.py `
    --wav-dirs vv=data\VocalVerse `
    --n-per-dir 10 `
    --apply-dereverb `
    --dereverb-backend df3_cascade `
    --out-dir outputs\smoke_c_df3_cascade
```

⚠ 注意:`scripts/audio_quality_probe.py` 目前**沒**支援 `--dereverb-backend` CLI,agent 需要先補上(模仿 binarizer.py 那段 ~3 行)。

通過條件:probe 跑完輸出 4 個 metric,hf_ratio JSD 應顯著低於 df3 baseline。

### 4.4 全訓(remote, ~20h)

```powershell
# 1. 重 binarize 兩 dataset(M4 + VV 都重跑,~12h)
python -m nsvb.data.binarizer `
    --dataset m4singer --data-root data --out-root data\binarized_v2_planC `
    --dereverb-backend df3_cascade `
    2>&1 | Tee-Object logs\binarize_m4_planC.log

python -m nsvb.data.binarizer `
    --dataset vocalverse --data-root data --out-root data\binarized_v2_planC `
    --dereverb-backend df3_cascade `
    --vocalverse-amateur-score-max 3.0 --vocalverse-chunk-sec 5.0 `
    2>&1 | Tee-Object logs\binarize_vv_planC.log

# 2. 重 cluster PPG(K=100,per-utt-mean-norm) — ~30 min
python -m nsvb.data.cluster_ppg fit-assign `
    --binarized-root data\binarized_v2_planC --k 100 --per-utt-mean-norm

# 3. 重切 splits(seed=42 保持 deterministic)
python scripts/make_splits.py `
    --binarized-root data\binarized_v2_planC `
    --m4-test-singers Alto-2 Tenor-3 `
    --m4-val-songs-per-singer 2 `
    --vv-test-singer-frac 0.10 `
    --vv-val-utterance-frac 0.05 `
    --seed 42

# 4. 重訓 Stage 1(~5h on A100)
python -m nsvb.task.stage1 `
    --binarized-root data\binarized_v2_planC --ppg-dim 1280 `
    --batch-size 16 --max-steps 80000 `
    --init-from-nsvb checkpoints\nsvb_ckpts\nsvb_1030_vae_mle\model_ckpt_steps_200000.ckpt `
    --ckpt-dir checkpoints_v2\stage1_planC `
    --split-dir data\binarized_v2_planC\splits `
    --val-interval 1000 --val-max-batches 50 `
    2>&1 | Tee-Object logs\stage1_planC.log

# 5. 訓 Stage 2(套用 Plan A v3 配方,~5h on A100)
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2_planC `
    --stage1-ckpt checkpoints_v2\stage1_planC\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planC `
    --split-dir data\binarized_v2_planC\splits `
    --max-steps 50000 `
    --batch-size 16 --num-workers 4 `
    --lambda-patchnce 2.0 --lambda-identity-pro 0.2 --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 --val-eval-n-samples 20 `
    2>&1 | Tee-Object logs\stage2_v3_planC.log
```

### 4.5 Plan C 特定額外驗收

除了 §7 共用 eval,Plan C 多跑 **vocoder identity test** 驗證新 binarized 對 vocoder 友好度:

```powershell
python -m scripts.vocoder_identity_test `
    --vocoder-ckpt checkpoints\1012_hifigan_all_songs_nsf\model_ckpt_steps_1170000.ckpt `
    --wav-dirs m4=data\m4singer vocalverse=data\VocalVerse `
    --n-per-dir 20 `
    --out-dir outputs\vocoder_test_planC
```

預期目標:VV SSIM 從 v2 0.65 拉到 ≥ 0.80(若失敗 → df3_cascade 沒解 vocoder 問題,要試 demucs)。

---

## 5. Plan D: pro-distribution matching loss

### 5.1 改了什麼

新增 G loss:`l_pro_match = ‖mean(mel_out) - pro_mean_env‖²`,把每 batch 的輸出 envelope 拉向 pro_mean_env(從 train split 抽 200 個 pro sample 預先算)。

**理論**:既有 `pro_direction_alignment` 是「修飾向量是否往 pro 走」,但只是後驗評估指標。Plan D 把這個指標**轉成訓練信號**,直接逼 M 學到「往 pro 走」。

**已實作**:[stage2.py](../nsvb/task/stage2.py) 內 `_compute_pro_mean_env_tensor()` + `train_step` 3e。CLI flag `--lambda-pro-match`。

**風險**:模型可能 overfit 到單一 envelope,失去個體多樣性(Risk 5 Mode Collapse 變體)。**緩解**:`--lambda-pro-match 0.5`(不要 ≥ 1.0)+ 維持 PatchNCE 跟 spk_emb 個體 anchor。

### 5.2 Smoke test(Win,~10 min)

```powershell
$env:PYTHONPATH = "."
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints\smoke_d `
    --max-steps 100 `
    --batch-size 4 --num-workers 0 --device cpu `
    --lambda-pro-match 0.5 `
    --pro-match-n-samples 20 `
    --val-eval-interval 50 --val-eval-n-samples 4
```

通過條件:`[stage2] pro_mean_env computed (n=20, range=[...])` 在 init log 出現 + 跑完 100 步無 crash + log 中 `l_pro_match` 值出現。

### 5.3 全訓(remote, ~8h)

```powershell
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planD `
    --max-steps 50000 `
    --batch-size 16 --num-workers 4 `
    --lambda-pro-match 0.5 `
    --pro-match-n-samples 200 `
    --lambda-patchnce 2.0 --lambda-identity-pro 0.2 --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 --val-eval-n-samples 20 `
    2>&1 | Tee-Object logs\stage2_v3_planD.log
```

### 5.4 Plan D 特定觀察

- `l_pro_match` 應從 ~0.5 緩慢降到 ~0.1 區間
- 若 `l_pro_match` 從一開始就 < 0.05 → pro_mean_env 太接近 baseline,沒 push 效果 → 不會壞但無增益
- 若 `l_pro_match` 訓中飆高 → M 跟其他 loss 衝突,要降 `lambda_pro_match` 到 0.2

---

## 6. Plan E: M 架構放大

### 6.1 改了什麼

目前 M 是 `m_hidden_dim=256, m_num_layers=4, m_kernel_size=1`(0.20M params,pointwise MLP 結構)。

Plan E 試兩個變體:
- **E-1**:`m_hidden_dim 256→384`(model 1.5x 大)
- **E-2**:E-1 加 `m_kernel_size 1→3`(加時間感受野)

### 6.2 Smoke test

```powershell
# E-1
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints\smoke_e1 `
    --max-steps 100 --batch-size 4 --num-workers 0 --device cpu `
    --m-hidden-dim 384 `
    --val-eval-interval 50 --val-eval-n-samples 4

# E-2
# 同上 + --m-kernel-size 3
```

通過條件:init log 顯示 `M=0.XX M` 大於 v2(原 0.20M,E-1 應該 ~0.45M)。

### 6.3 全訓(remote, ~5h each)

```powershell
# E-1: 大 hidden_dim
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planE1 `
    --max-steps 50000 --batch-size 16 --num-workers 4 `
    --m-hidden-dim 384 `
    --lambda-patchnce 2.0 --lambda-identity-pro 0.2 --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 --val-eval-n-samples 20 `
    2>&1 | Tee-Object logs\stage2_v3_planE1.log

# E-2: 大 hidden_dim + kernel=3
python -m nsvb.task.stage2 `
    --binarized-root data\binarized_v2 `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --ckpt-dir checkpoints_v2\stage2_v3_planE2 `
    --max-steps 50000 --batch-size 16 --num-workers 4 `
    --m-hidden-dim 384 --m-kernel-size 3 `
    --lambda-patchnce 2.0 --lambda-identity-pro 0.2 --lambda-adv-mel 0.1 `
    --d-z-warmup-steps 10000 `
    --val-eval-interval 5000 --val-eval-n-samples 20 `
    2>&1 | Tee-Object logs\stage2_v3_planE2.log
```

---

## 7. 共用 eval / score sheet

### 7.1 每 plan 訓完跑的 eval pipeline

每 plan 訓完(`stage2_step50000.pt` + 自動產出的 `stage2_best.pt`)走同一套 eval:

```powershell
# 1. test set inference(無 vocoder,~3h on A100 for 3773 samples)
$PLAN = "planA"   # 改成對應 plan
$env:PYTHONPATH = "."
python scripts/stage2_ckpts_listening.py `
    --stage1-ckpt checkpoints_v2\stage1\stage1_best.pt `
    --stage2-ckpt-dir checkpoints_v2\stage2_v3_$PLAN `
    --binarized-root data\binarized_v2 `
    --val-split data\binarized_v2\splits\test.txt `
    --out-dir outputs\stage2_v3_${PLAN}_listening_test `
    --all-samples --skip-vocoder --dump-mel --seed 42 `
    --steps "5000,10000,20000,30000,40000,50000"   # 含 best ckpt 的鄰近 steps

# 2. mel-domain eval(~10 min)
python scripts/stage2_mel_eval.py `
    --listening-dir outputs\stage2_v3_${PLAN}_listening_test `
    --binarized-root data\binarized_v2 `
    --pro-mean-n 500 --max-viz 20 `
    --out-dir outputs\stage2_v3_${PLAN}_eval_test
```

(Plan C 路徑變 `data\binarized_v2_planC`,其餘同。)

### 7.2 score sheet 模板

把每 plan 的 [report.md](../outputs/stage2_v2_eval_fulltest/report.md) 對照表填入:

| Plan | best step | test `pro_dir_align` | M4 vs VV ratio | `uv_conc` | `vsr` | `tdr_extra` | `hf_extra` | l_pro_match (if D) | l_adv_mel | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 baseline | 30000 | +0.60 | 11.26× | 0.54 ✅ | 0.96 ✅ | +0.03 ✅ | -0.18 ✅ | — | 0.8 | reference |
| A (v3) | ? | ? | ? | ? | ? | ? | ? | — | ? | |
| B (f0_support) | ? | ? | ? | ? | ? | ? | ? | — | ? | |
| C (df3_cascade) | ? | ? | ? | ? | ? | ? | ? | — | ? | + VV vocoder SSIM |
| D (pro_match) | ? | ? | ? | ? | ? | ? | ? | ? | ? | |
| E-1 (m_hidden 384) | ? | ? | ? | ? | ? | ? | ? | — | ? | |
| E-2 (E-1 + kernel=3) | ? | ? | ? | ? | ? | ? | ? | — | ? | |

### 7.3 Lock 判定標準(每 plan 獨立判)

| 結果 | 判定 |
|---|---|
| test `pro_dir_align` ≥ v2 + 0.05 (≥ +0.65) **且** 其他指標都 ✅ **且** M4 vs VV ratio ≥ 10× | **lock** 為新 baseline |
| test `pro_dir_align` ≥ v2 但其他指標退步 | 視退步幅度;`uv_conc > 0.65` 或 `tdr_extra > +0.10` 直接 drop |
| test `pro_dir_align` < v2 + 0.02 | **drop**(沒明顯進步)|
| M4 vs VV ratio < 5× | **drop**(L_id_pro 失效,M 對所有輸入同樣修飾)|

多個 plan 都 lock 時:取 `pro_dir_align` 最高且 ratio 維持 ≥ 10× 的。

### 7.4 全部 drop 怎麼辦

若 A / B / D / E 都 drop → 走 §0 決策樹的 Plan C(換 dereverb 重 binarize)。

若 Plan C 也 drop → 結案。寫 final report 列出所有 plan 的 metric,note v2 是局部最優,進一步進步需:
1. f0_support + 改 Mode A 推理也加 F0 平滑 → 解 train/infer mismatch
2. Stage 1 重訓改 fvae KL annealing
3. Larger speaker pool(加更多 M4 歌手或新的 pro singing dataset)

### 7.5 結果上傳(交付給 user)

本機訓練不會被中斷,中間檔案全留本地。每 plan 結束後**只**整理「重要結果」到 `handoff\` 資料夾,user 再透過 Drive / remote desktop / 隨身碟拿。

```powershell
# 每 plan 結束後跑這段(替換 $PLAN)
$PLAN = "planA"
$HANDOFF = "handoff\$PLAN"
New-Item -ItemType Directory -Force -Path $HANDOFF | Out-Null

# 1. best + final ckpt(每個 ~150 MB,只兩個)
Copy-Item checkpoints_v2\stage2_v3_$PLAN\stage2_best.pt $HANDOFF\
Copy-Item checkpoints_v2\stage2_v3_$PLAN\stage2_latest.pt $HANDOFF\stage2_final.pt

# 2. eval 報告 + per-step aggregate csv
Copy-Item outputs\stage2_v3_${PLAN}_eval_test\report.md $HANDOFF\
Copy-Item outputs\stage2_v3_${PLAN}_eval_test\metrics_aggregate.csv $HANDOFF\

# 3. 代表性 mel_grid.png 子集(取前 5 張不要全部)
$gridDir = "outputs\stage2_v3_${PLAN}_eval_test"
Get-ChildItem $gridDir -Recurse -Filter "mel_grid.png" | Select-Object -First 5 | `
    ForEach-Object {
        $relPath = $_.FullName.Substring((Resolve-Path $gridDir).Path.Length + 1)
        $destDir = Join-Path $HANDOFF (Split-Path $relPath -Parent)
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        Copy-Item $_.FullName $destDir
    }

# 4. 訓 log 的 summary(不要傳整份 11 MB log)
python scripts\summarize_stage2_log.py logs\stage2_v3_$PLAN.log
Copy-Item logs\stage2_v3_$PLAN.summary.md $HANDOFF\

# 5. tail 100 行原 log 給人看訓中收尾狀況
Get-Content logs\stage2_v3_$PLAN.log -Tail 200 | Out-File $HANDOFF\train_log_tail.txt

# 預期 $HANDOFF 大小:~500 MB(兩個 ckpt + 文字檔 + 5 張 PNG),容易傳
```

**最後總結:** 全部 plan 跑完後,寫 `handoff\training_results_v3.md`:
- 把 §7.2 score sheet 填完
- 推薦 lock 為 v3 的 plan + 理由
- 把所有 `handoff\plan*` 打包成 `handoff_all.zip` 給 user

---

## 8. Handoff checklist

agent 啟動前確認以下都備好。

### 8.1 程式碼

- [ ] git checkout 對應 commit hash(從 `COMMIT_HASH.txt`)
- [ ] `python -m nsvb.task.stage2 --help` 跑通,看到 `--f0-support`、`--lambda-pro-match`、`--m-hidden-dim`、`--val-eval-interval` 等 flag

### 8.2 資料

- [ ] `data/binarized_v2/{m4singer,vocalverse}/*.npz` 都在(從 §1.2 gdown + 解壓 + rename 完)
- [ ] `data/binarized_v2/splits/{train,val,test}.txt` 都在
- [ ] `data/binarized_v2/ppg_kmeans_centroids.npy` 在
- [ ] `checkpoints_v2/stage1/stage1_best.pt` 在
- [ ] `checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt` 在(eval 用)

### 8.3 環境

- [ ] CUDA 可用 (`torch.cuda.is_available() == True`)
- [ ] `pip install deepfilternet` 可正常 import
- [ ] `pip install matplotlib scipy soundfile gdown zstandard` 都裝好(eval + 下載用)

### 8.4 預期執行時間總和(本機 GPU,無 session 中斷)

| 場景 | 時間 |
|---|---|
| 一次性前置(下載 + 解壓) | ~3-5h(看網速) |
| Plan A only | ~5h 訓 + 3h eval = 8h |
| Plan A + E1 + E2(連續排隊,單 GPU) | ~24h |
| 全 plan (A/B/D/E1/E2 連續) + 視結果決定 C | 40-60h 直到結論 |

不需要 session 中斷處理,本機 GPU 可一路跑。

### 8.5 失敗模式 & 回報

agent 訓中如果遇到:
- **OOM**:降 `--batch-size` 16 → 12 或 8 → 4
- **NaN loss**:把對應 plan log 末段抓出來,看出問題的 step + loss 各分量;塞到 handoff 內 `nan_report.txt`
- **早停 hook 卡住**:val_eval 一次 ~15s,若超過 5 min 還沒回應 → 降 `--val-eval-n-samples` 到 10
- **訓完 eval 沒產 mel_grid.png**:很可能是 listening dir 內 .mel.npy 沒 dump,確認跑 listening 時有加 `--dump-mel`

每 plan 跑完後 agent 應產出(全部在 `handoff\plan{X}\`,見 §7.5):
1. `stage2_best.pt` + `stage2_final.pt`
2. `report.md`(eval)+ `metrics_aggregate.csv` + 5 張 mel_grid.png
3. `stage2_v3_plan{X}.summary.md`(訓 log summary)
4. `train_log_tail.txt`(原 log 末 200 行)
5. **plan 主 log 末段 print 一段 v2-vs-this-plan 單行結論**(供 user 快讀)

最後一份結案在 `handoff\training_results_v3.md`,填完 §7.2 score sheet + 推薦 lock 為 v3 的 plan + 理由,然後把所有 `handoff\plan*` 打包成 `handoff_all.zip` 給 user。