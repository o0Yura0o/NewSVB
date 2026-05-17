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
| binarized v2(單一 tar,含 centroids)| `NSVB-ZH/data/binarized_v2/binarized_v2.tar.zst` (~95 GB) | Phase 0 §6 + cluster_ppg 後打包上傳;tar 內結構 `binarized/{m4singer,vocalverse,ppg_kmeans_centroids.npy}` |
| NSVB pretrained CVAE | `nsvb_ckpts/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt` | Stage A §2.8 上傳 |
| COMMIT_HASH.txt | `NSVB-ZH/COMMIT_HASH.txt` | 訓練用的 git commit |
| `rclone.conf`(可選)| `NSVB-ZH/rclone.conf` | Phase 0 設定後備份 |

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

⚠ **不要直接用 `zstd -dc /content/drive/...` 餵 FUSE 路徑**(看似 stream,實際上 Drive FUSE 會把讀的檔在本機**自動 cache** ~95GB,加上解壓出來的 ~110GB,磁碟 ~235GB 直接爆 → 之前實測踩過)。

**用 `rclone cat` 走 Drive API stream,完全不落本機**:

```python
!apt-get install -y zstd > /dev/null
!rm -rf /content/local_binarized
!mkdir -p /content/local_binarized

# rclone cat → zstd -dc → tar -xf  全程 pipe,沒有任何中間檔
# 唯一落地的就是 /content/local_binarized/ 底下的 ~110 GB 解壓檔
# tar 內結構是 binarized/{m4singer,vocalverse,ppg_kmeans_centroids.npy};
# --strip-components=1 砍掉 binarized/ 前綴 → 三項直接落到 /content/local_binarized/
!rclone cat GD2CLL:NSVB-ZH/data/binarized_v2/binarized_v2.tar.zst \
    | zstd -dc \
    | tar -xf - -C /content/local_binarized --strip-components=1

!ls /content/local_binarized   # 應看到 m4singer/  vocalverse/  ppg_kmeans_centroids.npy
!df -h /content                # 應 ~140GB used(只有解壓檔 + OS,沒有 FUSE cache)

# sanity check
%cd /content/NSVB-ZH
!PYTHONPATH=. python scripts/verify_binarized.py \
    --root /content/local_binarized --dataset m4singer vocalverse
# 預期:0 bad, phoneme_id 100% (cluster_ppg 已跑)
```

> 前提:§1.2.5 的 rclone install + config restore 已做完,`rclone listremotes` 看得到 `GD2CLL:`。

### 1.4 把 `data/binarized` symlink 指 local

⚠ **跟 phase0 §3.5 衝突的處理**:§1.2 跑 phase0 §3.5 後,`data` 整個是個 **symlink 指向 Drive**(`data → /content/drive/MyDrive/NSVB-ZH/data/`),而 Drive 上又有真目錄 `binarized/`(舊 K=200 baseline)。直接 `rm -f data/binarized` 會被 OS 拒絕(`Is a directory`,`-f` 不等於 `-r`,擋下來是好事 —— 否則 `rm -rf` 會穿過 symlink 把 Drive 上的 K=200 baseline 真的刪掉)。

正確做法:**拆掉 `data` 這個 symlink,在本地建 `data/` 目錄**,再讓 `data/binarized` 指 local 解壓檔:

```python
%cd /content/NSVB-ZH

# 1. 拆掉 data symlink(只刪 symlink 本身,不動 Drive 目標)
#    [ -L path ] 測試是否為 symlink;是的話 unlink 掉
!if [ -L data ]; then unlink data; echo "data symlink removed"; fi

# 2. 建本地 data/,讓 binarized 指 local v2 解壓檔
!mkdir -p data
!rm -f data/binarized              # 防舊 binarized symlink 殘留
!ln -sfn /content/local_binarized data/binarized

# 3. 驗證
!ls -la data/binarized             # binarized -> /content/local_binarized
!ls data/binarized/                # m4singer/ vocalverse/ ppg_kmeans_centroids.npy
                                    # (splits/ 要 §1.5 跑完後才會出現)
```

**副作用**:`data` 從 symlink-to-Drive 變成本地真目錄後:
- `data/binarized` → local ✅(Phase 1 訓練讀的)
- `data/m4singer`(raw wav)、`data/VocalVerse`(raw wav) → 不再從 `data/...` 取得;Phase 1 訓練本來就不需要 raw wav,要用走絕對路徑 `/content/drive/MyDrive/NSVB-ZH/data/m4singer/...` 即可

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

> 🔴 **重要 — `checkpoints/` 不要碰**(2026-05 事故教訓)
>
> phase0 §3.5 已把 `checkpoints` 整個目錄 symlink 到 Drive (`checkpoints -> /content/drive/MyDrive/NSVB-ZH/checkpoints/`)。早期版本的這個 cell 跑 `ln -sfn /content/stage1_ckpts checkpoints/stage1`,因為 `checkpoints` 是 symlink,這個指令實際上是在 **Drive 上**建立 symlink `/content/drive/MyDrive/NSVB-ZH/checkpoints/stage1 -> /content/stage1_ckpts`。然後 rsync `/content/stage1_ckpts/ → /content/drive/.../checkpoints/stage1/` 會跟著 symlink 走回 local,變成 local→local 的空 rsync,Drive 上**永遠不會有 ckpt 落地**。`[ckpt-sync] local=17 drive=17` 是假象,因為 `ls $DRIVE_CKPT` 也跟著 symlink 走回 local,數兩遍同一份。Session 一斷 local SSD 清空 → 整批訓練成果消失。
>
> 修正方法:**完全不走 `checkpoints/` 這個目錄**。
> 1. 訓練 script 的 `--ckpt-dir` 直接指 local 絕對路徑 `/content/stage1_ckpts`(不要 `checkpoints/stage1`)。
> 2. Drive 目的地用**新前綴** `checkpoints_v2/` —— 跟 phase0 §3.5 的 `checkpoints/` 不同名,完全不會被 symlink 鏈到。
> 3. 同步前主動 verify Drive 端是真目錄,不是 symlink。

```python
import os, time, subprocess, threading
from pathlib import Path

LOCAL_CKPT = '/content/stage1_ckpts'                                       # local SSD,訓練實際寫入處
DRIVE_CKPT = '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1'        # ⚠ v2 前綴,避開 phase0 §3.5 symlink
os.makedirs(LOCAL_CKPT, exist_ok=True)
os.makedirs(DRIVE_CKPT, exist_ok=True)

# Verify:Drive 端必須是「真目錄,不是 symlink」,否則 rsync 會走回 local 形成空同步
def verify_drive_real(path: str):
    p = Path(path)
    assert p.exists(), f"{path} 不存在"
    assert not p.is_symlink(), (
        f"{path} 是 symlink → rsync 會跟著走、Drive 上不會落檔。"
        f"檢查 phase0 §3.5 是否把上層目錄 symlink 到別處,改用不同名稱前綴。"
    )
    assert p.is_dir(), f"{path} 不是目錄"
    # 寫一個 sentinel 確認 rsync 之後在 Drive 端能看到
    sentinel = p / '.write_test'
    sentinel.write_text('ok')
    assert sentinel.exists() and sentinel.read_text() == 'ok'
    sentinel.unlink()
    print(f"✅ {path} 是真目錄,寫入可達")

verify_drive_real(DRIVE_CKPT)

# 背景 sync:每 120 秒 local → Drive
# 為什麼 120s 而非 binarize 期間的 60s:ckpt 檔案大(~200-500 MB / 次)、寫入間隔長
# (每 5000 步 ~ 數十分鐘);過於頻繁的 rsync 反而浪費 Drive bandwidth
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
            # 額外印 Drive 端真實 byte 數(若被 symlink 騙回 local,這數字會跟 local 完全相等;
            # 真同步時 Drive 大小應該滯後 local 一個 rsync 週期 ~ 等於或略小於 local)
            drive_bytes = subprocess.check_output(
                f'du -sb {DRIVE_CKPT} 2>/dev/null | cut -f1', shell=True,
            ).decode().strip() or '0'
            print(f"[ckpt-sync {time.strftime('%H:%M:%S')}] "
                  f"local={local_n}  drive={drive_n}  drive_bytes={drive_bytes}",
                  flush=True)
        except Exception as e:
            print(f"[ckpt-sync] {e}", flush=True)
        if _sync_stop.wait(timeout=120):
            break

sync_thread = threading.Thread(target=ckpt_sync_loop, daemon=True)
sync_thread.start()
print(f'✅ ckpt sync started: {LOCAL_CKPT} → {DRIVE_CKPT} (every 120s)')
```

> 訓練 script 的 `--ckpt-dir` 改傳 `/content/stage1_ckpts`(絕對路徑),不再用 `checkpoints/stage1`。

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
    --ckpt-dir /content/stage1_ckpts \
    --split-dir data/binarized/splits \
    --val-interval 1000 --val-max-batches 50 \
    2>&1 | tee logs/stage1_$(date +%Y%m%d_%H%M%S).log
```

> ⚠ `--ckpt-dir` 直接傳 local 絕對路徑 `/content/stage1_ckpts`,**不要寫 `checkpoints/stage1`** —— 後者會被 phase0 §3.5 的 `checkpoints/` symlink 鏈導到 Drive,造成 §2 rsync 走 symlink 變空同步(2026-05 事故根因)。

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
# 把 Drive 上累積的 ckpt 拉回 local(覆蓋 §1.3 後尚未建立的 local 端 ckpts)
# ⚠ 走 v2 路徑(§2 已說明原因)
!mkdir -p /content/stage1_ckpts
!rsync -au /content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1/ /content/stage1_ckpts/

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
    --ckpt-dir /content/stage1_ckpts \
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

# 強制最後一次 rsync 確保 Drive 完整(走 v2 路徑)
subprocess.run(['rsync', '-au', '--info=stats2',
                '/content/stage1_ckpts/',
                '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1/'],
               check=True)
print('✅ Stage 1 ckpt 已完整落 Drive (checkpoints_v2/stage1)')
```

### Phase 2 入口
Stage 2 的 `--stage1-ckpt` 指向 Drive 上的 `stage1_best.pt`(走 v2 路徑:`/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1/stage1_best.pt`)。完整 Stage 2 流程見 §7(unattended mega-cell)。

---

## 7. Unattended overnight 訓練（Stage 1 → Stage 2 v2 一次跑完）

跑這個 cell 之前 §1.1–§1.5 全部跑完(mount + 環境 + 解壓 binarized + splits)。**§2 不需要單跑**,本 cell 內已包含 sync 啟動。整個 cell 設計成放著睡覺,中間不需要介入:

- Stage 1 (~3.5h on A100 @ 1.2it/s × 80K steps,首次 cold-start 略長;斷掉重連後本 cell 仍可重跑,會自動 `--resume latest` 接續)。
- Stage 1 結束自動切到 Stage 2 **v2 config**(`freeze_d_mel + λ_adv_mel=0.05 + lr_dz=2e-4`)。
- 全程 sync ckpt 到 Drive `checkpoints_v2/` (避開 phase0 §3.5 的 symlink 鏈)。
- log tee 到 Drive,session 斷了也看得到。

### v2 config 改了什麼、為什麼

v1 訓練(已遺失)觀察到的問題:
- `Δ/z ~0.93`、`temporal_diff_ratio ~0.77` 都遠超 [training_flow.md §3.6.1](training_flow.md) 的健康範圍(0.03–0.30 / < 0.3) → M 過度激進改 latent + 抹平時間軌跡。
- `l_adv_mel ≈ 5` 卡在 floor 下不去 → [risk.md §二.3](../risk.md) 的 amateur F0 conditioning confound:fake mel 餵 `f0_a`(業餘 F0)、real mel 來自 pro,D_mel 用「F0 軌跡不像 pro」當捷徑判 fake,M 為縮這 floor 改去動 latent 時間結構。

v2 對策:
- **`--freeze-d-mel`**:凍 D_mel 不更新(D_mel forward 仍給 M 梯度,但本身 weight 不變),讓 mel 對抗壓力不再追著 M 跑。
- **`--lambda-adv-mel 0.05`**(預設 0.2):把 mel 層 adversarial 權重從 main loss 同數量級降到 ~25% → mel 訊號當 prior,不主導 M。
- **`--lr-dz 2e-4`**(預設 4e-4):TTUR 比率 4× → 2×,削弱 latent 層 D_z 對 M 的拉力。維持 D_z > M 的 TTUR 慣例(不打破),只是溫和點。
- ⚠ **f0_support 想法評估後本次不加**:理論上對 §二.3 confound 是更根本的解(讓 fake/real mel decoder condition 都吃 smoothed F0,移除 D_mel 的 F0 捷徑),但本次先驗證「freeze + λ 降」是否足夠;若 v2 結果仍見 Δ/z > 0.5 或 tdr > 0.5,v3 再加 f0_support 並隔離變量。

### 訓練 cell(複製整段下去執行,然後關 laptop 去睡)

```python
# ============================================================
# Stage 1 → Stage 2 v2 unattended overnight 訓練
# 前置:§1.1–§1.5 已跑完(mount + env + 解壓 binarized + splits)
# Drive 目的地:checkpoints_v2/ (新前綴,避開 phase0 §3.5 symlink 鏈)
# ============================================================
import os, time, subprocess, threading, shlex
from pathlib import Path

REPO         = '/content/NSVB-ZH'
LOCAL_S1     = '/content/stage1_ckpts'
LOCAL_S2     = '/content/stage2_ckpts'
DRIVE_S1     = '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1'
DRIVE_S2     = '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage2_v2'
DRIVE_LOGS   = '/content/drive/MyDrive/NSVB-ZH/logs_v2'
NSVB_INIT    = '/content/drive/MyDrive/ckpts/nsvb_1030_vae_mle/model_ckpt_steps_200000.ckpt'

# 0. 準備目錄(全部 Drive 路徑要先 mkdir 並 verify 是真目錄)
for p in (LOCAL_S1, LOCAL_S2, DRIVE_S1, DRIVE_S2, DRIVE_LOGS):
    os.makedirs(p, exist_ok=True)

def verify_drive_real(path: str):
    p = Path(path)
    assert p.exists() and not p.is_symlink() and p.is_dir(), (
        f"{path} 不是真目錄(可能被 symlink 鏈到別處) → rsync 會走偏。"
        f"避免用 `checkpoints/` 開頭,改用 `checkpoints_v2/`。"
    )
    sentinel = p / '.write_test'
    sentinel.write_text('ok'); assert sentinel.read_text() == 'ok'; sentinel.unlink()

for p in (DRIVE_S1, DRIVE_S2, DRIVE_LOGS):
    verify_drive_real(p)
print('✅ Drive 路徑 verify 通過')

# 1. 背景 sync(stage1 → DRIVE_S1, stage2 → DRIVE_S2)
_sync_stop = threading.Event()

def _sync_pair(local: str, drive: str):
    subprocess.run(['rsync', '-au', f'{local}/', f'{drive}/'], capture_output=True)

def _count(path: str) -> str:
    try:
        return subprocess.check_output(
            f'ls {path}/*.pt 2>/dev/null | wc -l', shell=True,
        ).decode().strip()
    except Exception:
        return '?'

def ckpt_sync_loop():
    while not _sync_stop.is_set():
        _sync_pair(LOCAL_S1, DRIVE_S1)
        _sync_pair(LOCAL_S2, DRIVE_S2)
        print(f"[ckpt-sync {time.strftime('%H:%M:%S')}] "
              f"s1: local={_count(LOCAL_S1)} drive={_count(DRIVE_S1)}  "
              f"s2: local={_count(LOCAL_S2)} drive={_count(DRIVE_S2)}",
              flush=True)
        if _sync_stop.wait(timeout=120):
            break

sync_thread = threading.Thread(target=ckpt_sync_loop, daemon=True)
sync_thread.start()
print('✅ ckpt sync started (every 120s, stage1+stage2 平行)')

# 2. 從 Drive 拉回已有的 ckpt(idempotent,重連時自動接續)
subprocess.run(['rsync', '-au', f'{DRIVE_S1}/', f'{LOCAL_S1}/'], check=False)
subprocess.run(['rsync', '-au', f'{DRIVE_S2}/', f'{LOCAL_S2}/'], check=False)

# 3. Stage 1:若 stage1_best.pt 不存在則訓練,否則 skip
os.chdir(REPO)
ts = time.strftime('%Y%m%d_%H%M%S')

stage1_best = Path(LOCAL_S1) / 'stage1_best.pt'
stage1_latest = Path(LOCAL_S1) / 'stage1_latest.pt'

if stage1_best.exists():
    print(f'⏭️  Stage 1 已完成: {stage1_best} 存在,skip Stage 1 直接跑 Stage 2')
else:
    resume_flag = '--resume latest' if stage1_latest.exists() else \
                  f'--init-from-nsvb {shlex.quote(NSVB_INIT)}'
    cmd = (
        f'PYTHONPATH=. python -m nsvb.task.stage1 '
        f'  --binarized-root data/binarized '
        f'  --ppg-dim 1280 --batch-size 16 --num-workers 4 '
        f'  --max-steps 80000 '
        f'  --ckpt-dir {shlex.quote(LOCAL_S1)} '
        f'  --split-dir data/binarized/splits '
        f'  --val-interval 1000 --val-max-batches 50 '
        f'  {resume_flag} '
        f'2>&1 | tee {shlex.quote(f"{DRIVE_LOGS}/stage1_{ts}.log")}'
    )
    print(f'▶️  Stage 1 start ({"resume" if stage1_latest.exists() else "cold start"})')
    rc = subprocess.call(cmd, shell=True)
    if rc != 0:
        print(f'❌ Stage 1 exit code {rc} → 停 sync,不繼續 Stage 2')
        _sync_stop.set(); sync_thread.join(timeout=180)
        raise SystemExit(rc)
    # 最終確認 best 存在(stage1.py val loop 應該已產出)
    assert stage1_best.exists(), f'Stage 1 跑完但 {stage1_best} 不存在 — 檢查 val_interval 設定'
    # 確保 Stage 1 全部落 Drive 後再進 Stage 2(防 Stage 2 crash 帶走 Stage 1 還沒同步的 ckpt)
    _sync_pair(LOCAL_S1, DRIVE_S1)
    print(f'✅ Stage 1 done; {stage1_best.name} 已落 Drive')

# 4. Stage 2 v2:freeze_d_mel + λ_adv_mel=0.05 + lr_dz=2e-4
stage2_latest = Path(LOCAL_S2) / 'stage2_latest.pt'
resume_flag2 = '--resume latest' if stage2_latest.exists() else ''
cmd2 = (
    f'PYTHONPATH=. python -m nsvb.task.stage2 '
    f'  --binarized-root data/binarized '
    f'  --ppg-dim 1280 --batch-size 16 --num-workers 4 '
    f'  --max-steps 120000 '
    f'  --stage1-ckpt {shlex.quote(str(stage1_best))} '
    f'  --ckpt-dir {shlex.quote(LOCAL_S2)} '
    f'  --split-dir data/binarized/splits '
    f'  --freeze-d-mel --lambda-adv-mel 0.05 --lr-dz 2e-4 '
    f'  {resume_flag2} '
    f'2>&1 | tee {shlex.quote(f"{DRIVE_LOGS}/stage2_v2_{ts}.log")}'
)
print(f'▶️  Stage 2 v2 start ({"resume" if stage2_latest.exists() else "from scratch"})')
print(f'    {cmd2}')
rc2 = subprocess.call(cmd2, shell=True)
print(f'Stage 2 exit code: {rc2}')

# 5. 收尾:停 sync + 最終 rsync
print('Stopping ckpt sync...')
_sync_stop.set()
while sync_thread.is_alive():
    print('  等 ckpt sync 跑完當前這輪 rsync...')
    sync_thread.join(timeout=30)
print('  ckpt sync thread 已結束')

subprocess.run(['rsync', '-au', '--info=stats2', f'{LOCAL_S1}/', f'{DRIVE_S1}/'], check=True)
subprocess.run(['rsync', '-au', '--info=stats2', f'{LOCAL_S2}/', f'{DRIVE_S2}/'], check=True)
print('✅ All ckpts 已完整落 Drive (checkpoints_v2/stage1 + checkpoints_v2/stage2_v2)')
```

### 醒來檢查清單

1. 看 Drive `logs_v2/stage1_*.log` 末尾 → 應該到 step 80000 + `[val] new best ... saved stage1_best.pt`
2. 看 Drive `logs_v2/stage2_v2_*.log` → 看 `Δ/z` / `temporal_diff_ratio` 收斂值
   - 健康範圍:`Δ/z ∈ [0.03, 0.30]`,`tdr < 0.3`(v1 是 0.93 / 0.77)
3. 看 Drive `checkpoints_v2/stage2_v2/stage2_step{N}.pt`(每 5000 步一個 + `stage2_latest.pt`)
4. Listening test:`python scripts/stage2_ckpts_listening.py`(切到 v2 ckpt 路徑)

### 預期時間

| 階段 | 預估 |
|---|---|
| Stage 1 (80K steps @ ~1.2 it/s) | ~18-22h GPU |
| Stage 2 v2 (120K steps @ ~2.0 it/s) | ~16-18h GPU |
| 合計 wall clock | ~34-40h,需 1-2 次 session 重連 |

若一次 session 沒跑完:重連後跑 §1.1–§1.5 + §7 整段 cell 即可,本 cell 設計 idempotent —— 自動偵測 stage1_best 存在 → 跳到 Stage 2;Stage 2 也自動 `--resume latest`。

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