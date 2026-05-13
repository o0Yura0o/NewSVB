# Phase 0 跑在 Colab → Phase 1/2 跑在訓練機：完整步驟

把 Phase 0（資料下載 + binarize + cluster + 4 個 gate）卸載到 Colab Pro+ A100，
僅把 `~110 GB` binarized 輸出帶回訓練機跑 Phase 1/2。

## 0. Pre-flight 檢查（5 分鐘）

| 項目 | 確認 |
|---|---|
| **Colab 訂閱** | Pro+（A100 access + 200 GB 磁碟） |
| **Google Drive 空間** | ≥ 150 GB（raw 42 GB + binarized 110 GB） |
| **訓練機磁碟** | ≥ 200 GB SSD 空閒（binarized 110 GB + ckpt + buffer） |
| **訓練機 conda 環境** | 已照 [deployment_linux.md §2](deployment_linux.md) 建好 NSVB-ZH env |
| **預訓練 ckpt 已在訓練機** | `1030_vae_mle/model_ckpt_steps_200000.ckpt`（Phase 1 用）+ `1012_hifigan_all_songs_nsf/`（Phase 3 用） |

---

## 1. 兩階段流程概覽 ⭐

把 Colab session 拆兩段，**A100 compute units 省 ~7%**（約 36 units / 月配額 500），並把網路下載階段卸到便宜的 CPU runtime：

| Stage | runtime | 做什麼 | 時長 | A100 units 估計 |
|---|---|---|---:|---:|
| **A** | **CPU**（不勾 GPU）| 環境設定 + 預下載 Whisper/DF3 model + 下載 VocalVerse / M4Singer raw 資料 | ~2-3 h | 0 |
| **B** | **A100 GPU** | 完整 pip install + 跑 4 個 gate + binarize + cluster | ~16-20 h | ~200 |

**為什麼可以拆**：所有需要持久化的東西（pip 套件之外）都放 Drive——raw data、Whisper HF cache、git commit hash、vocoder ckpt 路徑。runtime 切換只清掉 `/content/`，Drive 保留。Stage A 用 CPU 跑下載階段，省下的 GPU session 配額留給真正的 binarize。

> **想單階段全在 A100 跑也可以**：跳過 §2 直接從 §3 開始（要在 §3.4 之後手動加 §2.5 / §2.6 / §2.7 / §2.8 內容），多花 ~36 units 與 ~3h GPU 時間。本文件以兩階段為主軸。

---

## 2. Stage A：CPU session（~2-3 h，~0 GPU units）

### 2.1 開 CPU runtime

**Runtime → Change runtime type → 硬體加速器 = CPU**（不勾任何 GPU）；「大量 RAM」不用切。

### 2.2 掛 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/drive/MyDrive/NSVB-ZH'
os.makedirs(f'{WORK}/data', exist_ok=True)
os.makedirs(f'{WORK}/data/binarized', exist_ok=True)
os.makedirs(f'{WORK}/checkpoints', exist_ok=True)
os.makedirs(f'{WORK}/outputs', exist_ok=True)
os.makedirs(f'{WORK}/hf_cache', exist_ok=True)
```

### 2.3 Git clone + pin commit

```python
%cd /content
!git clone <你的 repo URL> NSVB-ZH
%cd NSVB-ZH

# 紀錄 commit hash 到 Drive，Stage B 與訓練機都要對齊到這個
!git rev-parse HEAD > /content/drive/MyDrive/NSVB-ZH/COMMIT_HASH.txt
!cat /content/drive/MyDrive/NSVB-ZH/COMMIT_HASH.txt
```

> **為什麼 pin commit**：`audio_config.py` 的 `F0_FMAX / HOP_SIZE / WHISPER_HIDDEN_LAYER` 等常數會 freeze 進 .npz。Stage B 或訓練機若用不同 commit 載入 .npz，行為未定義。

### 2.4 最小 pip + 把 HF cache 指向 Drive

CPU 階段只裝下載 / 預快取需要的最小集合；完整 pip install 留到 Stage B。

⚠ **Python 3.12 + deepfilternet 必須先裝 Rust toolchain**：

`deepfilterlib`（DF3 的 Rust backend）在 PyPI 只有 cp310/cp311 wheel，**沒有 cp312 wheel**。Colab 預設 Python 3.12，pip 會嘗試從 source 編譯，需要 Rust。

```python
# 1. 裝 Rust（~3 min，~250 MB）
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os
os.environ['PATH'] = f"/root/.cargo/bin:{os.environ['PATH']}"
!rustc --version   # 確認看到 rustc 1.x.x

# 2. 裝套件（deepfilternet 會從 source 編譯 deepfilterlib，~5-10 min）
# 不加 -q：pip 過程訊息要留，方便 debug
!pip install huggingface-hub transformers deepfilternet

# 3. ⭐ 把編好的 wheel 存到 Drive，Stage B 直接用，不用重編
#    --no-deps 避免連同 numpy/requests 等 transitive dep 一起下載（省 ~26 MB Drive 空間）
!mkdir -p /content/drive/MyDrive/NSVB-ZH/wheels
!pip wheel --no-deps deepfilterlib==0.5.6 deepfilternet==0.5.6 \
    -w /content/drive/MyDrive/NSVB-ZH/wheels/
!ls -lh /content/drive/MyDrive/NSVB-ZH/wheels/
# 預期看到（⚠ DeepFilterLib 是大小寫混合）：
#   DeepFilterLib-0.5.6-cp312-cp312-linux_x86_64.whl
#   deepfilternet-0.5.6-py3-none-any.whl

# 4. HF cache 指向 Drive，Stage B 切 GPU 時 Whisper 不會被重新下載
os.environ['HF_HOME'] = '/content/drive/MyDrive/NSVB-ZH/hf_cache'
print('HF_HOME =', os.environ['HF_HOME'])
```

> **不在 Stage A 做 `from df.enhance import init_df` sanity check**：
> Colab CPU runtime 預載的 torchaudio 太新（>= 2.5，移除 `backend.common`），會 `ModuleNotFoundError: No module named 'torchaudio.backend'`。Stage B 明確裝 torchaudio==2.4.1+cu121，那邊 import 可正常運作（含 deprecation warning），所以這個 sanity check 在 Stage A 沒意義。Stage A 的目標僅是把 wheel 編好存 Drive。

> **numpy / packaging 降版警告無害**：deepfilternet 把 numpy 從 Colab 預設的 2.0 降到 1.26.4，正好是 [environment.yml](../environment.yml) 想要的版本。Colab 預載的 opencv/jax/xarray 會抱怨但我們**不會用到那些**，不用 restart runtime。

> **為什麼把 wheel 存 Drive**：deepfilterlib 是 CPU-only Rust extension（不依賴 CUDA），同 Ubuntu/glibc/Python 版本完全跨 runtime 相容。Stage B 直接 `pip install /content/drive/.../wheels/*.whl`，省掉每次 session 重連都要等 5-10 min 編譯。

> **若 deepfilternet 仍裝不上**的常見對策：
> - Rust 已裝但編譯失敗：`!rustup update && pip install --no-cache-dir deepfilternet`
> - 暫時網路 / crates 下載抽風：直接重跑同 pip 命令
> - HF_TOKEN 警告（橘色 warning）：與此錯誤無關，公開 model 不需要 token，可忽略

> **不要用 conda**：Colab notebook cell 是非互動 shell，`!conda activate` 不跨 cell 持續；Colab session 本身就是 ephemeral 環境，無 conda 隔離意義。若已 terminal 裝了 conda env，刪除省 ~3 GB：`!rm -rf ~/miniconda3 ~/.conda ~/.condarc`。

### 2.5 預下載 Whisper-large-v3 到 Drive

```python
# Whisper-large-v3 (~3 GB)，下載到 HF_HOME = Drive
from transformers import WhisperFeatureExtractor, WhisperModel
print("Downloading Whisper-large-v3...")
WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype='auto')
print("Whisper cached")

# 確認 cache 已寫進 Drive
!ls /content/drive/MyDrive/NSVB-ZH/hf_cache/                 # 預期看到 hub/ 與 xet/
!du -sh /content/drive/MyDrive/NSVB-ZH/hf_cache/hub/         # 預期 ~2.9 GB
!ls /content/drive/MyDrive/NSVB-ZH/hf_cache/hub/             # 預期看到 models--openai--whisper-large-v3
```

CPU 預計 5-10 分鐘。

> **huggingface-hub 1.x cache 結構**：根目錄分 `hub/`（傳統 model storage）與 `xet/`（XET content-addressed 大檔）。Whisper-large-v3 預設走 `hub/`（除非 hf-xet 主動把整個 blob 搬到 xet/，目前看 Colab 1.11.0 + hf-xet 1.4.3 預設仍走 hub/）。Stage B 的 huggingface-hub 0.24.0 只認 `hub/`，xet/ 會忽略——所以**只要 hub/ ≈ 3 GB 就 OK**。若 hub/ 反而 < 100 MB 而 xet/ ≈ 3 GB，需要 `HF_HUB_DISABLE_XET=1` 重下載。

> **為什麼不在 Stage A 預下載 DF3 model？**
> `df.enhance.init_df()` 把 DF3 model（~50 MB）存到 `~/.cache/DeepFilterNet/`，**這個路徑不是 Drive**，session 一斷就消失，預下載沒幫到 Stage B。
> 而且 Stage A 上 Colab 預載的 torchaudio 太新（移除了 `backend.common`），呼叫 `init_df` 會 `ModuleNotFoundError: No module named 'torchaudio.backend'`。
> Stage B 我們明確裝 torchaudio==2.4.1+cu121，會有舊 `backend.common`（含 deprecation warning），`init_df` 在那邊可正常跑。
> DF3 model 50MB 在 Stage B 重下只要 ~5 秒，不值得為它折騰。

### 2.6 下載 VocalVerse（~31 GB，1-2 h）

```python
%cd /content/NSVB-ZH
!PYTHONPATH=. python scripts/download_vocalverse.py \
    --out-dir /content/drive/MyDrive/NSVB-ZH/data/VocalVerse
```

### 2.7 解壓 M4Singer（~10 min）

M4Singer 官方頁無公開 API，先**手動把 zip 上傳到你的 Google Drive 任意位置**，再 Colab 解壓：

```python
import zipfile
zip_path = '/content/drive/MyDrive/m4singer.zip'   # ⚠ 改成你實際的 zip 路徑
extract_dir = '/content/drive/MyDrive/NSVB-ZH/data/m4singer'
with zipfile.ZipFile(zip_path) as z:
    z.extractall(extract_dir)

# 驗證
!find /content/drive/MyDrive/NSVB-ZH/data/m4singer -name "*.wav" | wc -l   # 預期 20896
!find /content/drive/MyDrive/NSVB-ZH/data/VocalVerse -name "*.wav" | wc -l  # 預期 929
```

### 2.8 上傳 NSVB 1012 vocoder ckpt（~200 MB）

Phase 0 Gate ① 需要。上傳到你的 Drive 任意位置，記下路徑（Stage B 引用用）：

```python
# ⚠ 改成你實際的路徑
VOCODER_CKPT = '/content/drive/MyDrive/nsvb_ckpts/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt'
!ls -lh "{VOCODER_CKPT}"
```

### 2.9 Stage A 完成檢查 + Disconnect

```python
print("=== Stage A 完成檢查 ===")
!ls -la /content/drive/MyDrive/NSVB-ZH/
print("\n--- Commit hash ---")
!cat /content/drive/MyDrive/NSVB-ZH/COMMIT_HASH.txt
print("\n--- Raw data ---")
!find /content/drive/MyDrive/NSVB-ZH/data/VocalVerse -name "*.wav" | wc -l
!find /content/drive/MyDrive/NSVB-ZH/data/m4singer -name "*.wav" | wc -l
print("\n--- HF cache ---")
!ls /content/drive/MyDrive/NSVB-ZH/hf_cache/
print("\n--- Vocoder ckpt ---")
!ls -lh /content/drive/MyDrive/nsvb_ckpts/
```

確認都對之後，**Runtime → Disconnect and delete runtime**（釋放 CPU session，停止計時）。

---

## 3. Stage B：切到 A100 session（所有 GPU 工作）

### 3.1 開 A100 runtime

**Runtime → Change runtime type → 硬體加速器 = A100 GPU**；「大量 RAM」自動。確認：

```python
!nvidia-smi   # 預期 A100-SXM4-40GB
```

若拿不到 A100 可退 L4（多開 High-RAM toggle、預期慢 1.5-2×）。

### 3.2 重新掛 Drive + 重指 HF cache

```python
from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/drive/MyDrive/NSVB-ZH'

# ⭐ 關鍵：重指 HF cache 到 Drive，Stage A 預下載的 Whisper 不會被重新下載
os.environ['HF_HOME'] = f'{WORK}/hf_cache'
print('HF_HOME =', os.environ['HF_HOME'])

# Stage A 上傳的 vocoder ckpt 路徑（改成你的實際路徑）
VOCODER_CKPT = '/content/drive/MyDrive/nsvb_ckpts/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt'
```

### 3.3 重新 git clone + 對齊 commit

```python
%cd /content
!git clone <你的 repo URL> NSVB-ZH
%cd NSVB-ZH

# 對齊 Stage A 記下的 commit
!git checkout $(cat /content/drive/MyDrive/NSVB-ZH/COMMIT_HASH.txt)
!git rev-parse HEAD  # 應與 COMMIT_HASH.txt 一致
```

### 3.4 完整 pip install（~3-5 min）

**Python 版本注意事項**：Colab 預設 Python 3.12，本 repo `environment.yml` pin 為 3.11。Phase 0 輸出 .npz 是純 numpy array，**只要底層數值庫版本一致，3.11 與 3.12 跑出的特徵在 fp32 ULP 級別相同**——訓練機 3.11 讀 Colab 3.12 出的 .npz 完全 OK。

兩個 pinned 版套件沒有 3.12 wheel，改用 `>=` 版：
- `pyworld==0.3.4` → `pyworld>=0.3.5`
- `praat-parselmouth==0.4.3` → `praat-parselmouth>=0.4.5`（PyPI 最新 0.4.7）

```python
%cd /content/NSVB-ZH

# PyTorch cu121 完整三件套（含 torchvision，避免 Colab 預載 fastai/timm 因
# torchvision 被卸掉產生的非致命警告）
!pip uninstall -y torch torchaudio torchvision -q
!pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# 從 Stage A 預編好的 deepfilternet wheel 安裝（省 5-10 min Rust 編譯）
# ⚠ DeepFilterLib 檔名是大小寫混合（Linux glob 大小寫敏感，不能用 deepfilterlib-*.whl）
!pip install /content/drive/MyDrive/NSVB-ZH/wheels/DeepFilterLib-*.whl \
             /content/drive/MyDrive/NSVB-ZH/wheels/deepfilternet-*.whl -q

# 其餘依賴。
# ⚠ --force-reinstall 必要：Colab A100 runtime 預載的 scipy/sklearn/librosa 等
#   是對 numpy 2.x 編譯的；pip 看版本號相同會 skip，但 .so 仍是舊的 → 後續
#   import 會踩 ABI mismatch `numpy.dtype size changed, may indicate binary
#   incompatibility. Expected 96 from C header, got 88 from PyObject`
# 為什麼不全 --force-reinstall 整個 list：太慢、且非 C 擴展套件不需要
!pip install --force-reinstall --no-deps --no-cache-dir \
    numpy==1.26.4 scipy==1.13.0 librosa==0.10.2 soundfile==0.12.1 \
    resampy==0.4.3 scikit-learn==1.5.1 scikit-image==0.24.0 -q

!pip install \
    pytorch-lightning==1.9.5 \
    pyloudnorm==0.1.1 \
    "pyworld>=0.3.5" "praat-parselmouth>=0.4.5" \
    webrtcvad==2.0.10 einops==0.8.0 \
    torchcrepe==0.0.24 \
    resemblyzer==0.1.4 pypinyin==0.53.0 g2pM==0.1.2.5 jieba==0.42.1 \
    transformers==4.40.0 huggingface-hub==0.24.0 tokenizers==0.19.1 \
    datasets==2.20.0 accelerate==0.30.0 \
    pandas==2.2.2 openpyxl==3.1.5 tensorboard==2.17.0 tqdm==4.66.4 \
    pyyaml==6.0.2 \
    matplotlib==3.9.1 portalocker==2.10.1 filelock==3.15.4 tabulate==0.9.0 \
    -q

# ⭐ 必要：裝完後 restart runtime
# 因為 §3.2 之前 import 過 numpy（mount Drive 也會載 numpy），降版後的新 .so
# 不會生效，必須 restart Python process 才能載入一致版本
print("\n=== 請手動 Runtime → Restart runtime（Ctrl+M .），然後重做 §3.2 / §3.3 / §3.5（不用重跑 §3.4）===")
```

> **若 wheel 路徑不存在**（沒做 Stage A §2.4 步驟 4 的 `pip wheel`）：
> 把 `deepfilternet==0.5.6` 加回主 pip line，**並先裝 Rust**：`!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && export PATH=/root/.cargo/bin:$PATH`，會多花 ~5-10 min 編譯。

### 3.5 Symlink Drive → 本地相對路徑

```python
%cd /content/NSVB-ZH
!ln -sfn /content/drive/MyDrive/NSVB-ZH/data data
!ln -sfn /content/drive/MyDrive/NSVB-ZH/outputs outputs
!ln -sfn /content/drive/MyDrive/NSVB-ZH/checkpoints checkpoints
!ls -la | head -10
```

> 為什麼用 symlink：所有腳本預設讀 `./data/...` 相對路徑；symlink 讓它指向 Drive 而不用改參數。

### 3.6 驗證：GPU + Whisper 從 Drive cache 載入（不重下）

```python
import torch
print(f'torch: {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'GPU:   {torch.cuda.get_device_name(0)}')

# Whisper：應該快速載完（HF cache 已在 Drive）
import time
from transformers import WhisperModel
t0 = time.time()
m = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16)
elapsed = time.time() - t0
print(f'Whisper loaded in {elapsed:.1f}s (cache hit if < 60s)')
del m
torch.cuda.empty_cache()

# DeepFilterNet（也用 Drive cache）
from df.enhance import init_df
init_df()
print('DF3 ready')
```

> 若 Whisper load 時間 > 60s 或開始重新下載，代表 `HF_HOME` 沒指對；回 §3.2 確認。

### 3.7 Stage B 啟動完成

至此 A100 session 就緒：Drive 內有所有 raw data + Whisper cache + ckpt，git checkout 對齊 commit，pip 完整。直接進 §4 跑 Gate ①。

> **Session timeout 後重連 Stage B**：重做 §3.2 / §3.3 / §3.4 / §3.5（mount + git + pip + symlink，約 5 分鐘）；Whisper cache 在 Drive 不會重下。

---

## 4. Phase 0 Gate ①：Vocoder identity test（~35 分鐘，兩跑）

**為什麼比想像中久**：VocalVerse 是 200-秒 full-length 歌（M4Singer 才 5 秒 snippet），每首要全長跑 pyworld F0（~30-60s）+ torchcrepe F0 RMSE（~20-40s）+ vocoder forward（~10-15s）= ~90s/首。20 首 VocalVerse × 兩跑（raw + loud_norm）≈ 30 min，加 M4Singer ~5 min。整體 ~35 min。

> 想加速可加 `--max-test-duration 30`（每首切前 30s 測，但目前腳本沒這個旗標；若需要可加 ~10 行補丁，VocalVerse 部分 90s → 15s，整體 ~8-10 min）

**先跑這個再 binarize**——vocoder 過不了就不用浪費 16h binarize：

```python
%cd /content/NSVB-ZH

# Run A: raw wav baseline（vocoder 原訓練分布）
!PYTHONPATH=. python -m scripts.vocoder_identity_test \
    --vocoder-ckpt "{VOCODER_CKPT}" \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 --save-wavs --f0-method pyworld --f0-interp \
    --out-dir outputs/phase0_vocoder_raw

# Run B: loud-normed（NSVB-ZH binarize 端實際分布）
!PYTHONPATH=. python -m scripts.vocoder_identity_test \
    --vocoder-ckpt "{VOCODER_CKPT}" \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 20 --save-wavs --f0-method pyworld --f0-interp \
    --apply-loudness-norm \
    --out-dir outputs/phase0_vocoder_loudnormed
```

**兩個 report.json 都要看到** `"verdict": "PASS"`：

```python
import json
for name in ['phase0_vocoder_raw', 'phase0_vocoder_loudnormed']:
    with open(f'outputs/{name}/report.json') as f:
        r = json.load(f)
    for ds, d in r['results'].items():
        print(f"{name}/{ds}: {d['verdict']}  SSIM={d['mel_ssim_mean']:.3f}")
```

> 不過：vocoder 對中文歌聲不適配，需 fine-tune vocoder（超出本流程，停下來規劃）。

---

## 5. Phase 0 Gate ②：Audio quality probe（5 分鐘）

```python
!PYTHONPATH=. python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --out-dir outputs/phase0_audio_quality

# 看 verdict
import json
with open('outputs/phase0_audio_quality/report.json') as f:
    r = json.load(f)
print(r.get('verdict', r))
```

通過條件：所有 metric（SFM / Reverb / HF / SNR）的 JSD < 0.10。

---

## 6. Phase 0 Binarize（12-18 小時，**可中斷續跑**）

### 6.1 開大 cell 跑 M4Singer

```python
%cd /content/NSVB-ZH
# M4Singer 21K snippets，~6-10h on A100
!PYTHONPATH=. python -m nsvb.data.binarizer \
    --dataset m4singer \
    --data-root data \
    --out-root data/binarized 2>&1 | tee data/binarized/m4_log.txt
```

**Session timeout 後**：重連 Colab → 重跑 §3.2 / §3.3 / §3.4 / §3.5（mount + git + pip + symlink）→ 直接重跑這個 cell。`skip_existing` 預設開，會跳過已存的 .npz。

進度檢查（另開 cell）：

```python
!ls /content/drive/MyDrive/NSVB-ZH/data/binarized/m4singer/ | wc -l
# 目標：~20896；中途看到逐漸接近即為健康
```

### 6.2 跑 VocalVerse（含 amateur_score 過濾 + chunk 切片，~6-10h）

```python
!PYTHONPATH=. python -m nsvb.data.binarizer \
    --dataset vocalverse \
    --data-root data \
    --out-root data/binarized \
    --vocalverse-amateur-score-max 3.0 \
    --vocalverse-chunk-sec 5.0 \
    2>&1 | tee data/binarized/vv_log.txt
```

兩個關鍵旗標：
- `--vocalverse-amateur-score-max 3.0`：依 (技巧+氣息)/2 ≤ 3 過濾，留 **536 筆**真業餘
- `--vocalverse-chunk-sec 5.0`：每首歌切成 5-sec chunks，解決與 M4Singer 平均 5.4s snippets 的長度失衡

> **為什麼要 chunk**：VV 平均 200s × 536 首 vs M4 5.4s × 21K 個 → 樣本數 1:39。chunk 後 VV ≈ 536 × 40 = **21,440 chunks**，與 M4 21K 平衡。對齊 NSVB 設計「每樣本=一個 phrase 訓練單位」哲學。詳見 [training_flow.md §1.1.2](training_flow.md)。

進度檢查：

```python
!ls /content/drive/MyDrive/NSVB-ZH/data/binarized/vocalverse/ | wc -l
# 目標：~21000-21500 個 .npz（536 songs × ~40 chunks/song）
# 檔名格式：{歌曲id}__{录音id}__c{NNN}.npz
!ls /content/drive/MyDrive/NSVB-ZH/data/binarized/vocalverse/ | head -3
```

### 6.3 PPG k-means → phoneme_id（30-60 分鐘）

```python
!PYTHONPATH=. python -m nsvb.data.cluster_ppg \
    --binarized-root data/binarized \
    --centroids-out data/binarized/ppg_kmeans_centroids.npy \
    --k 200 \
    --stage all
```

完成後，每個 .npz 多了 `phoneme_id` key，`data/binarized/` 多了 `ppg_kmeans_centroids.npy`。

---

## 7. Phase 0 Gate ③：JSD（register / phoneme 跨 dataset 一致性）

目前 `nsvb/utils/jsd_check.py` 只有 self-test，沒有 CLI 跑實際 .npz 對比。手動跑：

```python
import numpy as np
from pathlib import Path
from nsvb.utils.jsd_check import jensen_shannon_divergence, frame_freq_distribution

def collect_ids(root, key, n_classes):
    ids = []
    for p in sorted(Path(root).glob('*.npz')):
        with np.load(p) as d:
            ids.append(d[key])
    return np.concatenate(ids)

# Phoneme JSD
m4_ph = collect_ids('data/binarized/m4singer', 'phoneme_id', 200)
vv_ph = collect_ids('data/binarized/vocalverse', 'phoneme_id', 200)
p_m4 = frame_freq_distribution(m4_ph, 200)
p_vv = frame_freq_distribution(vv_ph, 200)
jsd_ph = jensen_shannon_divergence(p_m4, p_vv)

# Register JSD
m4_reg = collect_ids('data/binarized/m4singer', 'register_id', 5)
vv_reg = collect_ids('data/binarized/vocalverse', 'register_id', 5)
p_m4 = frame_freq_distribution(m4_reg, 5)
p_vv = frame_freq_distribution(vv_reg, 5)
jsd_reg = jensen_shannon_divergence(p_m4, p_vv)

print(f'JSD(phoneme):  {jsd_ph:.4f}  {"PASS" if jsd_ph < 0.05 else "FAIL"} (< 0.05)')
print(f'JSD(register): {jsd_reg:.4f}  {"PASS" if jsd_reg < 0.05 else "FAIL"} (< 0.05)')
```

不過：考慮重新對 vocalverse 用更嚴的 `amateur_score_max` 或對 M4Singer 子採樣對齊分布。

---

## 8. Phase 0 Gate ④：PPG cluster 品質檢查（5 分鐘）

```python
!PYTHONPATH=. python -m scripts.cluster_ppg_inspect \
    --binarized-root data/binarized \
    --datasets m4singer vocalverse \
    --phoneme-vocab-size 200 \
    --out-dir outputs/phase0_cluster_inspect
```

通過條件：
- `MI(phoneme_id; register_id)` < 0.3 bit（< 0.6 marginal）
- `mean_dwell_frames_voiced` ≥ 8 frames

不過：依腳本建議的補救順序（降 K → 換 layer → 去 DC → mode filter）處理。視覺化圖在 `outputs/phase0_cluster_inspect/{dataset}/*.png`。

---

## 9. 打包準備傳輸

### 9.1 確認所有 gate PASS

```python
# 一頁總結
import json
print("=== Phase 0 Gate Summary ===")
for path in [
    'outputs/phase0_vocoder_raw/report.json',
    'outputs/phase0_vocoder_loudnormed/report.json',
    'outputs/phase0_audio_quality/report.json',
    'outputs/phase0_cluster_inspect/report.json',
]:
    try:
        with open(path) as f:
            r = json.load(f)
        print(f"{path}: {r}")
    except FileNotFoundError:
        print(f"{path}: MISSING")
print(f"\nJSD: phoneme={jsd_ph:.4f}, register={jsd_reg:.4f}")  # 從 §7 變數
```

### 9.2 壓縮 binarized（可選，省 20% 傳輸時間）

```python
%cd /content/drive/MyDrive/NSVB-ZH/data
# zstd level 3 平衡壓縮率與速度；level 19 慢 10× 但只多省 ~5%
!apt-get install -y zstd > /dev/null
!tar -cf - binarized/ | zstd -3 -T0 > binarized.tar.zst
!ls -lh binarized.tar.zst
# 預期 ~80-90 GB（PPG fp16 壓縮率 ~20%）
```

不壓縮也可，但 Drive → 訓練機若走 rclone 多檔有 overhead，**單檔 tar 通常較快**。

---

## 10. 傳輸到訓練機（1-3 小時）

### 10.1 方法 A：Drive ↔ 訓練機（rclone，最簡單）

訓練機上裝 rclone（一次性）：

```bash
# on 訓練機
curl https://rclone.org/install.sh | sudo bash
rclone config
# Interactive：選 13 (Google Drive)，照提示走 OAuth
# 命名 remote 為 gdrive
```

下載：

```bash
# on 訓練機
cd ~/workspace/NSVB-ZH
mkdir -p data
rclone copy gdrive:NSVB-ZH/data/binarized.tar.zst ./data/ \
    --progress --transfers 4 --checkers 8
# 速度：Drive Free ~10 MB/s（要 ~2.5h）；Drive Workspace ~50 MB/s（30 min）
```

### 10.2 方法 B：GCS / S3 中介（最快，>100 MB/s）

Colab 上傳到 cloud：

```python
# Colab
!pip install -q google-cloud-storage
# 假設你有 GCS bucket 和 service account；省略 auth 細節
!gsutil -m cp /content/drive/MyDrive/NSVB-ZH/data/binarized.tar.zst \
    gs://your-bucket/nsvb-zh/
```

訓練機下載：

```bash
gsutil -m cp gs://your-bucket/nsvb-zh/binarized.tar.zst ./data/
```

### 10.3 解壓 + 同時搬其他必要檔

```bash
# on 訓練機
cd ~/workspace/NSVB-ZH/data
zstd -d binarized.tar.zst -c | tar -xf -
ls binarized/                  # 應看到 m4singer/ vocalverse/ ppg_kmeans_centroids.npy
rm binarized.tar.zst           # 解壓完可刪
```

順手把 Phase 0 報告也帶回（debug 用）：

```bash
rclone copy gdrive:NSVB-ZH/outputs ./outputs --progress
rclone copy gdrive:NSVB-ZH/COMMIT_HASH.txt ./  # 用來對 git
```

---

## 11. 訓練機端 sanity check（5 分鐘）

### 11.1 同步 git commit

```bash
cd ~/workspace/NSVB-ZH
git fetch
git checkout $(cat COMMIT_HASH.txt)
git rev-parse HEAD  # 確認與 COMMIT_HASH.txt 一致
```

> 不能 checkout 別的 commit 跑訓練——`audio_config.py` 改過會與 .npz 的 freeze 值不一致。

### 11.2 樣本數量驗證

```bash
ls data/binarized/m4singer/*.npz | wc -l           # 預期 ~20896
ls data/binarized/vocalverse/*.npz | wc -l         # 預期 ~21000+（536 songs × ~40 chunks）
ls -lh data/binarized/ppg_kmeans_centroids.npy     # 必須存在
du -sh data/binarized                              # ~100 GB
```

### 11.3 .npz 可讀且 schema 正確

```bash
python -c "
import numpy as np
from pathlib import Path
for ds in ['m4singer', 'vocalverse']:
    p = sorted(Path(f'data/binarized/{ds}').glob('*.npz'))[0]
    with np.load(p) as d:
        keys = sorted(d.files)
        print(f'{ds}: {p.name}')
        print(f'  keys: {keys}')
        for k in ('mel', 'ppg', 'f0', 'spk_emb', 'phoneme_id', 'register_id'):
            assert k in keys, f'  MISSING: {k}'
        print(f'  mel shape={d[\"mel\"].shape} ppg shape={d[\"ppg\"].shape} ppg dtype={d[\"ppg\"].dtype}')
        # 應看到 ppg dtype=float16
"
```

### 11.4 快速重跑 Gate ③ + ④ 確認沒在傳輸過程中壞掉

```bash
# Gate ④（純讀 npz，~1 min）
python -m scripts.cluster_ppg_inspect \
    --binarized-root data/binarized \
    --datasets m4singer vocalverse \
    --phoneme-vocab-size 200 \
    --out-dir outputs/phase0_cluster_inspect_on_train_machine

# 比對 MI / dwell 值與 Colab 上跑的結果應該一致
diff <(jq '.datasets' outputs/phase0_cluster_inspect/report.json) \
     <(jq '.datasets' outputs/phase0_cluster_inspect_on_train_machine/report.json)
# 預期：完全相同（純讀同樣的 .npz）
```

### 11.5 確認預訓練 ckpt 就位

```bash
ls -lh checkpoints/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt
ls -lh checkpoints/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt
```

若沒有，按 [deployment_linux.md §3](deployment_linux.md) 下載。

---

## 12. 開始 Phase 1 訓練

```bash
cd ~/workspace/NSVB-ZH

# 用 tmux 避免 ssh 斷線中斷訓練
tmux new -s nsvb-stage1

# 在 tmux 內：
PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 \
    --batch-size 16 \
    --max-steps 80000 \
    --num-workers 4 \
    --init-from-nsvb checkpoints/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1 \
    2>&1 | tee logs/stage1_$(date +%Y%m%d_%H%M%S).log

# Ctrl+B 然後 D 離開 tmux（訓練繼續）；重連用 tmux attach -t nsvb-stage1
```

預估 2-3 週 on A100。

詳細 monitoring / resume / Phase 2 接續見 [deployment_linux.md §6-§7](deployment_linux.md)。

---

## 附錄 A：Colab 常見故障處理

| 症狀 | 對策 |
|---|---|
| Stage B Session timeout 中途中斷 binarize | 重連 → §3.2 / §3.3 / §3.4 / §3.5 重做（5 min）→ 重跑當下 cell（skip_existing 會接續） |
| A100 不可用 | 改 L4（多開 High-RAM），加倍預期時間；最壞 T4 也能跑（Whisper 載 fp16，~12h binarize） |
| Drive quota exceeded（讀寫太多次） | 等 12-24h 自動 reset；或先寫 Colab 本地 `/content/data/binarized`，session 結束前用 rsync 推到 Drive |
| Whisper / DF3 下載卡住（Stage A） | 重跑 §2.5 cell；若仍卡，改用 mirror（`HF_ENDPOINT` 環境變數） |
| Stage B 重啟 Whisper 又重下了 3GB | `HF_HOME` 沒指對；在跑 transformers import 之前確認 `os.environ['HF_HOME']` 指向 Drive |
| OOM 跑 Whisper 大 batch | binarizer 已是 sequential 處理，不應 OOM；若發生改 `--device cpu` 確認非配額問題 |

## 附錄 B：每個階段預期耗時 / 磁碟

| 階段 | 步驟 | A100 耗時 | CPU 耗時 | 磁碟新增 | 重要性 |
|---|---|---:|---:|---:|---|
| **Stage A** | §2.1-2.4 setup + minimal pip | — | 5 min | <1 GB | 必要 |
| | §2.5 預下載 Whisper / DF3 | — | 5-10 min | 3 GB | 必要 |
| | §2.6 下載 VocalVerse | — | 1-2 h | 31 GB | 必要 |
| | §2.7 解壓 M4Singer | — | 10 min | 11 GB | 必要 |
| | §2.8 上傳 vocoder ckpt | — | 2 min | 200 MB | 必要 |
| **Stage B** | §3 環境 setup（重連也走這） | 5-10 min | — | — | 必要 |
| | §4 Gate ① vocoder identity（兩跑 raw + loud_norm） | ~35 min | — | <100 MB | **dealbreaker** |
| | §5 Gate ② audio quality probe | 5 min | — | <10 MB | 必要 |
| | §6.1 binarize M4Singer | 6-10 h | — | 46 GB | 必要 |
| | §6.2 binarize VocalVerse | 6-10 h | — | 64 GB | 必要 |
| | §6.3 cluster_ppg | 30-60 min | — | (in-place) | 必要 |
| | §7 Gate ③ JSD | 5 min | — | <1 MB | 必要 |
| | §8 Gate ④ cluster inspect | 5 min | — | <10 MB | 必要 |
| | §9.2 壓縮 | 1 h | — | +85 GB | 可選 |
| **傳輸** | §10 傳訓練機 | — | 0.5-3 h | (傳輸不增 Drive) | 必要 |
| **訓練機** | §11 驗證 | — | 5 min | (sanity only) | 必要 |
| **合計** | | **~17-21 h A100** | **~2-3 h CPU** | **~150 GB on Drive** | |
