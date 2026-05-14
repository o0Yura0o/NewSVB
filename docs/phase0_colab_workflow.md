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

# Step 4a：主依賴列表（⚠ protobuf 不能放這裡，見 4b 說明）
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

# Step 4b：protobuf 5.28 必須「單獨命令 --upgrade」，不能併進 4a
# ⚠ 為什麼分開：tensorboard==2.17.0 explicit 要求 protobuf<5；若 protobuf>=5.28
#   與 tensorboard==2.17.0 放同一個 pip install，resolver 解不開 → ResolutionImpossible
#   整批都不裝。分開命令時 pip 先裝完 tensorboard（連同 protobuf 4.x），4b 再
#   --upgrade 覆蓋成 5.28，只印 non-fatal warning（tensorboard 2.17 dep 不滿足）。
# ⚠ 為什麼一定要 protobuf>=5.28：Colab 預載的 sentencepiece（Whisper tokenizer 依賴）
#   是用 protobuf 5.x 工具鏈生成的，需要 runtime_version API（5.26+ 才有）；
#   protobuf 4.x 下 binarize 載 Whisper 會
#   `cannot import name 'runtime_version' from 'google.protobuf'`
# binarize 不用 tensorboard；Stage 1/2 寫 event 檔功能與 protobuf 5+ 仍相容。
!pip install --upgrade "protobuf>=5.28.0" -q

# ⭐ 必要：裝完後 restart runtime
# 因為 §3.2 之前 import 過 numpy（mount Drive 也會載 numpy），降版後的新 .so
# 不會生效，必須 restart Python process 才能載入一致版本
print("\n=== 請手動 Runtime → Restart runtime（Ctrl+M .）===")
```

> **Restart 後該重跑什麼**——分兩種情境：
> - **同一 session 內首次裝完後的 restart**（你剛跑完 §3.4 第一次）：restart 後 pip 套件**還在 disk 上**，只需重做 §3.2 / §3.3 / §3.5（mount + git + symlink），**不用重跑 §3.4**。
> - **全新 session / runtime 完全重啟**（Colab timeout 重連、隔天重開）：`/content/` 整個被清空，**pip 套件全失**，必須**完整重跑 §3.4**。漏跑會在 binarize 時踩 `ModuleNotFoundError: No module named 'pyloudnorm'` 之類。

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

**跑兩次**（同 vocoder identity test 的策略）：

```python
# Run A：raw wav — 量 Risk 2 緩解前的原始差距
!PYTHONPATH=. python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --out-dir outputs/phase0_audio_quality_raw

# Run B：dereverb'd wav — 量 binarize 端實際進訓練的 mel 分布
!PYTHONPATH=. python -m scripts.audio_quality_probe \
    --wav-dirs m4=data/m4singer vocalverse=data/VocalVerse \
    --n-per-dir 100 \
    --apply-dereverb \
    --out-dir outputs/phase0_audio_quality_dereverbed

# 看兩個 verdict
import json
for name in ['phase0_audio_quality_raw', 'phase0_audio_quality_dereverbed']:
    with open(f'outputs/{name}/audio_quality_report.json') as f:
        r = json.load(f)
    print(f"\n=== {name} ===")
    print(f"verdict: {r.get('verdict')}")
    for mk, jsd in r.get('jsds', {}).items():
        flag = '✅' if jsd < 0.10 else '❌'
        print(f"  {mk}: JSD={jsd:.4f}  {flag}")
```

**通過條件**（重要：4 個 metric 可信度分兩級，FAIL 不等於失敗）：

| Metric | 可信度 | 為什麼 |
|---|---|---|
| `sfm` / `hf_ratio` | ⭐ **Reliable**（直接量頻譜，DF3 影響可預測） | 真正反映 mitigation 是否生效 |
| `snr_db` | ⚠ Heuristic（voiced_E/unvoiced_E ratio）| VV 持續背景噪音讓比例 saturate，DF3 改不了 |
| `reverb_sec` | ⚠ Heuristic（能量包絡衰減）| DF3 改 transient 形狀後 heuristic 失準 |

判讀邏輯：
- Run A（raw）預期會全 FAIL — M4 studio vs VV 業餘 phone mic 本來就差很大，這就是「Risk 2 為什麼存在」的證據
- **Run B（dereverb'd）只要 reliable metric (sfm / hf_ratio) 改善或 PASS，就算實質通過**——即便 snr_db / reverb_sec 形式 FAIL 也 OK，那是 metric artifact 不是 mitigation 失敗
- 真正的 ground truth 是 Stage 2 訓中 L5 monitor（`unvoiced_concentration < 0.55`），它直接量 M 是否在去殘響

**Colab A100 實測**（n_per_dir=100）：

| Metric | Raw JSD | Dereverb JSD | 改善 | 結論 |
|---|---:|---:|---|---|
| sfm | 0.644 | 0.390 | -39% | 方向對，未達 < 0.10 但 mitigation 生效 |
| hf_ratio | 0.258 | **0.048** ✅ | -81% | PASS |
| reverb_sec | 0.113 | 0.118 | +5% | heuristic 限制（DF3 改 transient） |
| snr_db | 0.659 | 0.624 | -5% | heuristic 限制（VV 持續背景噪音）|

→ 兩條 reliable metric 改善，**視為通過進 §6 binarize**；Stage 2 訓練時看 L5 monitor 保險。

---

## 6. Phase 0 Binarize（~15-18 小時，**可中斷續跑**）

⭐ **Colab IO 優化**：寫到 Drive 直接會被 FUSE overhead 額外拖慢；改寫**本地 SSD `/content/local_binarized/`**（100-300 MB/s）+ **背景 rsync 每分鐘同步到 Drive**，省下 IO 等待時間。

> **⚠ 時間瓶頸是 Whisper compute，不是 IO**：Whisper-large-v3 強制 30 秒視窗，M4Singer 每個 5-sec snippet 仍要跑完整 30s forward（25s 是 padding 浪費）。21K snippets × 1 forward each ≈ **M4Singer ~11-12h**。VocalVerse 整首 200s 跑 Whisper（內部切 ~7 個 30s 視窗），536 songs ≈ **VV ~3-5h**。local-write 只省 IO，省不了這個 compute floor。
>
> **未來優化（backlog，非現在）**：M4Singer 6 個 5s clip 拼成 1 個 30s 餵 Whisper batch → Whisper forward 砍 ~6x，M4 可降到 ~2-3h。需改 [ppg_whisper.py](../nsvb/data/feature_extract/ppg_whisper.py) + [binarizer.py](../nsvb/data/binarizer.py)。

### 6.0 啟動 local-write + background sync pattern（**必跑，§6.1–6.5 共用**）

跑 binarize **之前**先設定好本地目錄與背景同步。這個 cell 跑一次，sync thread 會持續到 §6.5 顯式停止（或 runtime restart）：

```python
import os, time, subprocess, threading

# 本地暫存（Pro+ A100 runtime 有 ~200 GB /content 空間，足夠 ~110 GB 全部 binarized）
LOCAL_OUT = '/content/local_binarized'
DRIVE_OUT = '/content/drive/MyDrive/NSVB-ZH/data/binarized'
os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

# 把 data/binarized symlink 改指到 local（覆蓋 §3.5 的 Drive 指向）
%cd /content/NSVB-ZH
!rm -f data/binarized
!ln -sfn /content/local_binarized data/binarized
!ls -la data/binarized

# 背景 rsync：每 60 秒一次 local → Drive
_sync_stop = threading.Event()

def sync_loop():
    while not _sync_stop.is_set():
        subprocess.run(
            ['rsync', '-au', f'{LOCAL_OUT}/', f'{DRIVE_OUT}/'],
            capture_output=True,
        )
        # print 進度（本地 vs Drive 累計檔案數）
        try:
            local_count = subprocess.check_output(
                f'find {LOCAL_OUT} -name "*.npz" | wc -l', shell=True,
            ).decode().strip()
            drive_count = subprocess.check_output(
                f'find {DRIVE_OUT} -name "*.npz" | wc -l', shell=True,
            ).decode().strip()
            print(f"[sync {time.strftime('%H:%M:%S')}] "
                  f"local={local_count}  drive={drive_count}", flush=True)
        except Exception as e:
            print(f"[sync] count failed: {e}", flush=True)
        if _sync_stop.wait(timeout=60):
            break

sync_thread = threading.Thread(target=sync_loop, daemon=True)
sync_thread.start()
print('✅ Background sync started (local → Drive, every 60s)')
```

### 6.1 跑 M4Singer

兩個關鍵旗標必加：
- `--out-root /content/local_binarized`：寫 fast SSD（不寫 Drive）
- `--skip-check-extra-dir /content/drive/...binarized`：session 重連時看 Drive 上已 sync 的也跳過，**不需 pre-populate**

```python
%cd /content/NSVB-ZH
# M4Singer 21K snippets，~11-12h on A100（Whisper 30s 視窗 padding 是 bottleneck）
!PYTHONPATH=. python -m nsvb.data.binarizer \
    --dataset m4singer \
    --data-root data \
    --out-root /content/local_binarized \
    --skip-check-extra-dir /content/drive/MyDrive/NSVB-ZH/data/binarized \
    2>&1 | tee /content/local_binarized/m4_log.txt
```

**Session timeout 後**：重連 → §3.2 / §3.3 / §3.4 / §3.5 → §6.0（重啟 sync thread）→ 直接重跑這個 cell。`--skip-check-extra-dir` 會看 Drive 已 sync 的 .npz 並跳過，**不用 rsync Drive→local**（省 ~30-60 min 重連時間）。

進度檢查（另開 cell；§6.0 sync 也會每 60s 自動 print）：

```python
import subprocess
local = subprocess.check_output('find /content/local_binarized/m4singer -name "*.npz" 2>/dev/null | wc -l', shell=True).decode().strip()
drive = subprocess.check_output('find /content/drive/MyDrive/NSVB-ZH/data/binarized/m4singer -name "*.npz" 2>/dev/null | wc -l', shell=True).decode().strip()
print(f'local: {local} / 20896  |  drive: {drive} / 20896')
# 目標：local 接近 20896；drive 落後本地最多 60s（一輪 sync 週期）
```

### 6.2 跑 VocalVerse（含 amateur_score 過濾 + chunk 切片，~3-5h）

```python
!PYTHONPATH=. python -m nsvb.data.binarizer \
    --dataset vocalverse \
    --data-root data \
    --out-root /content/local_binarized \
    --skip-check-extra-dir /content/drive/MyDrive/NSVB-ZH/data/binarized \
    --vocalverse-amateur-score-max 3.0 \
    --vocalverse-chunk-sec 5.0 \
    2>&1 | tee /content/local_binarized/vv_log.txt
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

### 6.3 Binarize 完整性驗證（必跑，~30 min–2 h）

binarize 跑完不代表資料正確。Colab 中斷過 session 時，這些破綻 binarize 自己不會報：
- **背景 rsync 中途被打斷** → Drive 上留下截斷的 .npz（`np.load` 報 `BadZipFile`）
- **`np.savez` 寫到一半 session 被殺** → local 留下截斷檔，rsync 照樣推上 Drive
- **VV：整首被 skip 但 chunk 沒切完** → binarizer 用 `__c000.npz` 是否存在當「整首已處理」的代表；某首在切 chunk 途中被中斷、`c000` 已 sync 到 Drive 但 `c016+` 沒 → 下個 session 看到 `c000` 就永久 skip 這首 → 缺尾。每個倖存的 .npz 本身都是好的，**錯的是數量**

**對 Drive 跑兩支診斷腳本**（local 不完整，Drive 才是累積完整集）：

```python
%cd /content/NSVB-ZH

# (1) verify：逐檔檢查可載入 / key 齊 / shape 一致 / 無 NaN-Inf；
#     並印「來源錄音」數（M4 應 ≈ 20896；VV chunk 對應的不同來源錄音應 = 536）
!PYTHONPATH=. python scripts/verify_binarized.py \
    --root /content/drive/MyDrive/NSVB-ZH/data/binarized \
    --dataset m4singer vocalverse

# (2) reconcile：只對 VV——從原始 wav 時長推算「每首應有幾個 chunk」，
#     比對 Drive 上實際 chunk 數，抓「整首被 skip 但 chunk 沒切完」的缺尾
!PYTHONPATH=. python scripts/reconcile_vv_chunks.py \
    --vv-source /content/drive/MyDrive/NSVB-ZH/data/VocalVerse \
    --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized
```

| 失敗模式 | verify | reconcile |
|---|---|---|
| 個別 .npz 截斷 / 損毀 | ✅ | — |
| M4Singer 整首缺（不切 chunk）| ✅ 看「來源錄音」≈ 20896 | 不適用 |
| VV 整首被 skip 但 chunk 沒切完 | ❌ 抓不到 | ✅ |

**有缺料時**：兩支腳本都會把問題 item_id 寫成清單檔（`*_bad_npz.txt` / `vocalverse_incomplete.txt`）。照腳本印的指令刪掉那些殘檔（VV 要連 `c000` 一起刪，否則 `skip_existing` 會繼續跳過），回 §6.1/§6.2 重跑 binarize 補回缺的那幾首即可（只補缺的，幾分鐘）。**全部齊全、壞檔 = 0 才往下做 §6.4。**

### 6.4 PPG k-means → phoneme_id（30-60 分鐘）

⚠ **前置條件：cluster_ppg 必須看到「完整」dataset**。它兩個 stage 都掃 `--binarized-root` 底下「所有」.npz：
- **fit**：從每首歌抽 200 frames fit k-means → 只看到部分資料 → centroids 分布偏掉
- **assign**：對「每一個」.npz in-place 重寫加 `phoneme_id` → 沒掃到的檔永遠缺 `phoneme_id`，訓練時 D_z 無法用

**問題**：若你 binarize 跨過中斷的 session（用 `--skip-check-extra-dir`），`/content/local_binarized` 只有「最後一個 session 新做的部分」——被 skip 的歷史檔只在 Drive 上，沒寫進 local。此時直接對 local 跑 cluster_ppg 是錯的。

**先驗證 Drive 這份完整健康，再把完整集拉回 local**：

```python
# (a) 先確認 Drive 上的 binarize 完整無損（§6.3）
#     verify_binarized.py + reconcile_vv_chunks.py 都過了再往下

# (b) 把 Drive 完整集 rsync 回 local（~110 GB；首次拉回較久，之後增量很快）
import subprocess
subprocess.run(['rsync', '-au',
    '/content/drive/MyDrive/NSVB-ZH/data/binarized/',
    '/content/local_binarized/'], check=True)

# (c) 確認 local 檔數 = Drive 檔數（兩邊應一致）
for ds in ['m4singer', 'vocalverse']:
    for label, root in [('local', '/content/local_binarized'),
                        ('drive', '/content/drive/MyDrive/NSVB-ZH/data/binarized')]:
        n = subprocess.check_output(
            f'find {root}/{ds} -name "*.npz" | wc -l', shell=True).decode().strip()
        print(f'  {ds} {label}: {n}')
```

> 若是「單一不中斷 session 跑完 binarize」，local 本來就完整，(b)/(c) 可跳過直接往下。

cluster_ppg 對 local 跑（in-place 修改每個 .npz 加 `phoneme_id` key），背景 sync 會自動偵測 mtime 更新並把 ~42K .npz 重新 sync 到 Drive：

```python
!PYTHONPATH=. python -m nsvb.data.cluster_ppg \
    --binarized-root /content/local_binarized \
    --centroids-out /content/local_binarized/ppg_kmeans_centroids.npy \
    --k 200 \
    --stage all
```

> **為什麼對 local 跑而不直接指 Drive**：assign stage 對每個 .npz 做 `np.savez_compressed` 重寫，~42K 檔走 Drive FUSE 極慢；且 in-place 重寫直接動 Drive，中途掛掉會寫壞 Drive 的 .npz 而沒有備份。對 local 跑＝在 SSD 上快、Drive 保持完好當備份、§6.0 背景 sync 再推回。

> **注意**：phoneme_id 加進去後 Drive 上的 ~42K .npz 全部要重 sync 一輪（~10-30 min；rsync `-u` 偵測 mtime）。耐心等 §6.5 final sync 確認都同步完。

### 6.5 ⚠ 全部跑完後，**停 sync + 強制最後一次 rsync**

```python
print('Stopping background sync...')
_sync_stop.set()
sync_thread.join(timeout=10)

print('Final rsync local → Drive (this catches any pending writes)...')
result = subprocess.run(
    ['rsync', '-au', '--info=stats2',
     '/content/local_binarized/',
     '/content/drive/MyDrive/NSVB-ZH/data/binarized/'],
    capture_output=True, text=True,
)
print(result.stdout)

# 驗證 final counts
for ds in ['m4singer', 'vocalverse']:
    n = subprocess.check_output(
        f'find /content/drive/MyDrive/NSVB-ZH/data/binarized/{ds} -name "*.npz" | wc -l',
        shell=True,
    ).decode().strip()
    print(f'  {ds}: {n} .npz on Drive')
# 預期：m4singer ~20896, vocalverse ~21000+
print('✅ Done. Drive is now in sync with local.')
```

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
    'outputs/phase0_audio_quality_raw/audio_quality_report.json',
    'outputs/phase0_audio_quality_dereverbed/audio_quality_report.json',
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
| | §5 Gate ② audio quality probe（兩跑 raw + dereverb） | ~20 min | — | <10 MB | 必要 |
| | §6.1 binarize M4Singer（Whisper 30s 視窗 bottleneck）| **11-12 h** | — | 46 GB | 必要 |
| | §6.2 binarize VocalVerse | **3-5 h** | — | 64 GB | 必要 |
| | §6.3 binarize 完整性驗證（verify + reconcile）| 30 min–2 h | — | <1 MB | 必要 |
| | §6.4 cluster_ppg（含 rsync Drive→local 補完整）| 1-2 h | — | (in-place) | 必要 |
| | §7 Gate ③ JSD | 5 min | — | <1 MB | 必要 |
| | §8 Gate ④ cluster inspect | 5 min | — | <10 MB | 必要 |
| | §9.2 壓縮 | 1 h | — | +85 GB | 可選 |
| **傳輸** | §10 傳訓練機 | — | 0.5-3 h | (傳輸不增 Drive) | 必要 |
| **訓練機** | §11 驗證 | — | 5 min | (sanity only) | 必要 |
| **合計** | | **~17-21 h A100** | **~2-3 h CPU** | **~150 GB on Drive** | |
