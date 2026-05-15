# Phase 1（Stage 1 CVAE）在 Colab 上跑：完整流程

Phase 0 完成後本應傳到 Linux 訓練機跑 Phase 1/2（見 [deployment_linux.md](deployment_linux.md)），訓練機暫時無法取得 → Phase 1 改在 Colab A100 上跑。

## 跟 Linux 訓練機版差在哪

| 項目 | Linux | Colab |
|---|---|---|
| 環境 setup | 一次性 conda env | **每次 session 重連跑 §3.2-§3.5**（~5 min）|
| 資料位置 | 本機 SSD 解壓一次 | **每次 session 從 Drive tar 解壓回 local**（~30-60 min）|
| 訓練連續性 | tmux 跑 3-5 天連續 | session 12-24h 就會斷，**需頻繁 resume** |
| ckpt 持久化 | 直接寫本機 SSD | **必須持續 sync 到 Drive**（本機隨 session 結束清空）|
| 觀測 | ssh + tail / tensorboard | notebook cell 看訓練 log（或 `%tensorboard`）|

[`nsvb/task/stage1.py`](nsvb/task/stage1.py) 已實作 `--resume`（恢復 M / opt / step），所以中斷接續是現成的。

---

## 0. Pre-flight

確認 Drive 上有：

| 檔案 | 位置 | 來源 |
|---|---|---|
| binarized v2 tars | `NSVB-ZH/data/binarized_v2/{m4singer,vocalverse}.tar.zst` | Phase 0 §6 + cluster_ppg 後打包上傳 |
| v2 centroids | `NSVB-ZH/data/binarized_v2/ppg_kmeans_centroids.npy` | 同上 |
| NSVB pretrained CVAE | `nsvb_ckpts/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt` | Stage A §2.8 上傳 |
| COMMIT_HASH.txt | `NSVB-ZH/COMMIT_HASH.txt` | 訓練用的 git commit |
| `rclone` 設定 | 已配好 `GD2CLL` remote（[phase0 §1 rclone 段](phase0_colab_workflow.md)） | 一次性 |

---

## 1. 開新 session（A100，每次重連都做）

### 1.1 Runtime → A100

```python
!nvidia-smi   # 預期 A100-SXM4-40GB
```

### 1.2 Mount + 環境 setup

走 [`phase0_colab_workflow.md`](phase0_colab_workflow.md) §3.2–§3.5：
- §3.2 mount Drive + 設 HF_HOME
- §3.3 `git clone` + `git checkout $(cat COMMIT_HASH.txt)`
- §3.4 完整 pip install（包含 `--force-reinstall` numpy ABI 那段）+ runtime restart
- §3.5 symlink `data/` `outputs/` `checkpoints/` → Drive

⚠ 訓練不需 DeepFilterNet（那是 binarize 用的），但其他依賴一樣要裝；§3.4 整段跑完最省事。

### 1.2.5 rclone（若你 session 內會用到上傳 / 下載單檔）

rclone binary + config 都隨 session 結束被清，每次重連都要還原。**前提是你之前已把 rclone.conf 備份到 Drive**（一次性，見 [Phase 0 收尾](phase0_log.md) 那次設定）：

```python
import os, shutil

# 1. 重裝 binary（~30s）
!curl https://rclone.org/install.sh 2>/dev/null | sudo bash > /dev/null

# 2. 從 Drive 還原 config（refresh_token 還在，不用重跑 OAuth）
BACKUP = '/content/drive/MyDrive/NSVB-ZH/rclone.conf'
TARGET = '/root/.config/rclone/rclone.conf'
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
shutil.copy(BACKUP, TARGET)

!rclone listremotes   # 應看到 GD2CLL:
```

若還沒備份過 rclone.conf：跑 `!rclone config` 重做 OAuth，完成後**馬上備份**到 Drive：
```python
shutil.copy(TARGET, BACKUP)
```
之後新 session 直接還原即可。

### 1.3 解壓 v2 binarized 到 local

```python
!apt-get install -y zstd > /dev/null
!mkdir -p /content/local_binarized

# 從 Drive tar 串流解壓到 local（~30-60 min；110GB 寫到本機 SSD）
for ds in ['m4singer', 'vocalverse']:
    print(f'--- extracting {ds} ---', flush=True)
    !zstd -dc /content/drive/MyDrive/NSVB-ZH/data/binarized_v2/{ds}.tar.zst \
        | tar -xf - -C /content/local_binarized --strip-components=1
!cp /content/drive/MyDrive/NSVB-ZH/data/binarized_v2/ppg_kmeans_centroids.npy \
    /content/local_binarized/

# sanity check
!PYTHONPATH=. python scripts/verify_binarized.py \
    --root /content/local_binarized --dataset m4singer vocalverse
# 預期：0 bad, phoneme_id 100% (cluster_ppg 已跑)
```

### 1.4 把 `data/binarized` symlink 指 local

```python
%cd /content/NSVB-ZH
!rm -f data/binarized
!ln -sfn /content/local_binarized data/binarized
!ls -la data/binarized
```

### 1.5 切 train/val/test splits（**首次**訓練前跑一次,之後 session 也可重跑,seed 固定 → 結果相同）

```python
%cd /content/NSVB-ZH
!PYTHONPATH=. python scripts/make_splits.py \
    --binarized-root /content/local_binarized \
    --m4-test-singers Alto-2 Tenor-3 \
    --m4-val-songs-per-singer 2 \
    --vv-test-singer-frac 0.10 \
    --vv-val-utterance-frac 0.05 \
    --seed 42
# 產生 /content/local_binarized/splits/{train,val,test}.txt + report.json
```

切割設計細節見 [training_flow.md §1.6](training_flow.md)。預期輸出:
- train ~38K(M4 ~18K + VV ~19K)
- val ~2K
- test ~4K(M4 2 整位歌手 + VV ~10% user)

**為什麼明確指定 `--m4-test-singers`**:讓 holdout 在 git 內可見,reviewer 跟未來的你都知道測試集是哪兩位。要重新自動挑可改成 `--m4-test-singers` 不傳(seed 42 自動挑)。

⚠ 注意:每次 session 重連如果重新 `make_splits.py`,**只要 seed 跟參數一樣,切割完全相同** —— 所以不用備份 split 檔到 Drive,當場再產生就好。

---

## 2. 啟動 ckpt local-write + background sync（**必跑**，跟 Phase 0 §6.0 同模式）

訓練 ckpt 寫到 local SSD 才快（每幾千步存一次），背景 rsync 同步到 Drive 保證 session 斷掉時 ckpt 已落地。

```python
import os, time, subprocess, threading

LOCAL_CKPT = '/content/stage1_ckpts'
DRIVE_CKPT = '/content/drive/MyDrive/NSVB-ZH/checkpoints/stage1'
os.makedirs(LOCAL_CKPT, exist_ok=True)
os.makedirs(DRIVE_CKPT, exist_ok=True)

# 把 checkpoints/stage1 symlink 改指 local（覆蓋 §3.5 的 Drive 指向）
%cd /content/NSVB-ZH
!rm -rf checkpoints/stage1
!mkdir -p checkpoints
!ln -sfn /content/stage1_ckpts checkpoints/stage1
!ls -la checkpoints/stage1

# 背景 sync：每 120 秒 local → Drive
# 為什麼 120s 而非 binarize 期間的 60s：ckpt 檔案大（~200-500 MB / 次）、寫入間隔長
# （每 5000 步 ~ 數十分鐘）；過於頻繁的 rsync 反而浪費 Drive bandwidth
_sync_stop = threading.Event()

def ckpt_sync_loop():
    while not _sync_stop.is_set():
        subprocess.run(
            ['rsync', '-au', f'{LOCAL_CKPT}/', f'{DRIVE_CKPT}/'],
            capture_output=True,
        )
        try:
            local_n = subprocess.check_output(
                f'ls {LOCAL_CKPT}/*.pt 2>/dev/null | wc -l', shell=True,
            ).decode().strip()
            drive_n = subprocess.check_output(
                f'ls {DRIVE_CKPT}/*.pt 2>/dev/null | wc -l', shell=True,
            ).decode().strip()
            print(f"[ckpt-sync {time.strftime('%H:%M:%S')}] "
                  f"local={local_n}  drive={drive_n}", flush=True)
        except Exception as e:
            print(f"[ckpt-sync] {e}", flush=True)
        if _sync_stop.wait(timeout=120):
            break

sync_thread = threading.Thread(target=ckpt_sync_loop, daemon=True)
sync_thread.start()
print('✅ ckpt sync started (local → Drive, every 120s)')
```

---

## 3. 開始訓練（first run）

```python
%cd /content/NSVB-ZH
!mkdir -p logs

# Stage 1 = CVAE pretrain，max_steps 80000
# --init-from-nsvb 從 NSVB 1030 VAE-MLE ckpt 載入 backbone weight（cold start 比較久）
# --split-dir 預設指 data/binarized/splits（§1.5 已產出），val loop 自動啟用
!PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 \
    --batch-size 16 \
    --max-steps 80000 \
    --num-workers 4 \
    --init-from-nsvb /content/drive/MyDrive/nsvb_ckpts/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt \
    --ckpt-dir checkpoints/stage1 \
    --split-dir data/binarized/splits \
    --val-interval 1000 --val-max-batches 50 \
    2>&1 | tee logs/stage1_$(date +%Y%m%d_%H%M%S).log
```

Cell 會持續輸出訓練 log:
- 每 `log_interval` 步(50)印一次 train loss / it/s
- 每 `val_interval` 步(1000)印一次 `[val step ...] val_l1=.. val_l2=.. val_kl=.. val_total=..`
- val_total 創新低時自動存 `stage1_best.pt`,印 `[val] new best val_total=.. → saved stage1_best.pt`

tensorboard 的 events 也寫在 `checkpoints/stage1/`,會被 background sync 推到 Drive。

⭐ **Phase 1 結束時用 `stage1_best.pt` 而非 `stage1_latest.pt`** 給 Stage 2 接續 —— best 是 val loss 最低的 ckpt,代表泛化最好的時刻;latest 是最後一個 step,可能已 overfit。

### 預期速度
A100 上估 ~1.0-1.5 it/s（FVAE encoder/decoder + KL loss）。80K steps ≈ 15-22h GPU 時間，**單一 Colab session 跑不完**，必須中斷 + resume 至少 1-2 次。

---

## 4. Resume on session reconnect

Session 斷掉重連後：

### 4.1 重做環境 setup
§1.1–§1.5 全部重跑(mount + git + pip + symlink + 解壓 binarized + verify + **重新 make_splits**)。

⚠ §1.5 重跑要用**完全一樣的參數**(同 `--m4-test-singers`、`--seed`),否則切割變了會破壞訓練連續性 —— 等於 train/val 換了一批,resume 的 ckpt 拿錯資料訓。

### 4.2 從 Drive 拉回最新 ckpt 到 local

```python
# 把 Drive 上累積的 ckpt 拉回 local（覆蓋 §1.3 後尚未建立的 local 端 ckpts）
!rsync -au /content/drive/MyDrive/NSVB-ZH/checkpoints/stage1/ /content/stage1_ckpts/

!ls -lh /content/stage1_ckpts/ | tail -10
# 應看到 stage1_step{N}.pt 與 stage1_latest.pt
```

### 4.3 重啟 background sync（§2）

§2 那段 cell 整段重跑（變數 `_sync_stop` / `sync_thread` 是 session-local 的，必須重建）。

### 4.4 用 `--resume latest` 接續訓練

```python
!PYTHONPATH=. python -m nsvb.task.stage1 \
    --binarized-root data/binarized \
    --ppg-dim 1280 \
    --batch-size 16 \
    --max-steps 80000 \
    --num-workers 4 \
    --ckpt-dir checkpoints/stage1 \
    --resume latest \
    2>&1 | tee logs/stage1_resume_$(date +%Y%m%d_%H%M%S).log
```

⚠ **不要再傳 `--init-from-nsvb`** —— `--resume` 會載入完整訓練狀態（M / opt / step），重複 init 會被 resume 覆蓋，但不必要。

`--resume latest` 會自動找 `{ckpt-dir}/stage1_latest.pt`。如果想釘住特定 step，用 `--resume checkpoints/stage1/stage1_step25000.pt`。

---

## 5. 訓練監控

### 5.1 cell 內看 log
training cell 本身會持續 print，每 50 步一行（`log_interval` 預設）。觀察：
- `it/s` 穩在 1.0-1.5 之間
- `loss_recon` / `loss_kl` / `loss_d_mel` 緩慢下降
- 無 NaN/Inf

### 5.2 tensorboard（可選，需開另一個 cell）

```python
%load_ext tensorboard
%tensorboard --logdir checkpoints/stage1
```
events 透過背景 sync 也會出現在 Drive 端，本機 close session 後 Drive 上仍可下載分析。

### 5.3 ckpt 同步狀況
看 §2 那個 background sync 每 120s 印的 `local=N  drive=M`，drive 應落後 local ≤ 1（一個 rsync 週期內）。

---

## 6. Phase 1 完成判定 + 收尾

### 完成條件
- `step ≥ 80000`，且
- loss 平台（`loss_recon` 在某 step 後 5K 步幾乎不再下降）

### Phase 1 結束：停 sync + 最後 rsync

跟 Phase 0 §6.5 同模式，**真的等 sync_thread 結束**而不是 `join(timeout=10)`：

```python
print('Stopping ckpt sync...')
try:
    _sync_stop.set()
    while sync_thread.is_alive():
        print('  等 ckpt sync 跑完當前這輪 rsync...')
        sync_thread.join(timeout=30)
    print('  ckpt sync thread 已結束')
except NameError:
    print('  sync 變數不在，skip')

import subprocess
_left = subprocess.run(['pgrep', '-a', 'rsync'],
                       capture_output=True, text=True).stdout.strip()
if _left:
    print('⚠ 仍有 rsync 程序:\n' + _left)

# 強制最後一次 rsync 確保 Drive 完整
subprocess.run(['rsync', '-au', '--info=stats2',
                '/content/stage1_ckpts/',
                '/content/drive/MyDrive/NSVB-ZH/checkpoints/stage1/'],
               check=True)
print('✅ Stage 1 ckpt 已完整落 Drive')
```

### Phase 2 入口
Stage 2 的 `--stage1-ckpt` 指向 Drive 上的 `stage1_final.pt`（或 `stage1_latest.pt`）。Phase 2 流程之後再寫（會類似這份但模型不同）。

---

## 附錄 A：常見故障

| 症狀 | 對策 |
|---|---|
| Resume 後 loss 突然飆高 | ckpt 載入有問題；確認 `--resume` 載到對的檔，且 `--init-from-nsvb` 沒重複帶 |
| OOM on A100 | 降 `--batch-size`（16 → 12 或 8）；max_frames=1500 是 dataset 端設的，配 batch=8 可用 |
| ckpt sync 顯示 `drive` 數字一直不動 | Drive quota 滿 / 網路抽風；先手動 `!rclone copy ...` 測一下；必要時減少 save_interval |
| Session 斷後 latest 不見 | `_sync_stop.set()` 前如果背景 sync 卡住，最後幾分鐘的 ckpt 可能沒推上去；下次重連用次新的 ckpt resume（差幾千步無妨） |
| 訓練 cell 沒輸出但 session 還在 | 可能是 dataloader stuck；考慮 `--num-workers 0` 用主進程除錯 |

## 附錄 B：時間估算

| 階段 | 預估 |
|---|---|
| 每次 session setup（mount/git/pip/解壓 binarized） | ~30-60 min |
| Stage 1 訓練（80K steps @ ~1.2 it/s） | ~18-22h GPU 時間 |
| Colab session 中斷次數 | 1-2 次（單 session ~12-18h） |
| Wall clock 總時間 | ~2-3 天（含重連 + setup overhead） |