# Stage 2 v3 訓練流程

> 前置:[phase2_outcome.md](phase2_outcome.md) v2 已 lock。本文件描述 v3 怎麼訓。
> v3 共用 v2 的 `stage1_best.pt`,不重訓 Stage 1。

## 1. v3 與 v2 對比

| 項目 | v2 | v3 | 動機 |
|---|---|---|---|
| `max_steps` | 120000 | **50000** | best @ 30K, 訓到 120K 沒幫助 |
| `lambda_patchnce` | 1.0 | **2.0** | 強化 content anchor, 縮 generalization gap |
| `lambda_identity_pro` | 0.1 | **0.2** | 強化 anti-漂移, 限制 amateur 過修 |
| `lambda_adv_mel` | 0.05 | **0.1** | mid-ground (v1 0.2 confound、v2 0.05 underused) |
| `freeze_d_mel` | True | **False** | 解凍 D_mel, 恢復 mel adv 訊號 |
| `d_z_warmup_steps` | 5000 | **10000** | 多給 PatchNCE 時間打底 |
| **NEW** 早停 hook | 無 | `--val-eval-interval 5000` | 每 5K 步算 val mel-alignment, 自動存 `stage2_best.pt` |

預期改善:
- `pro_direction_alignment` test 從 +0.60 朝 +0.75 推
- 自動 best ckpt, 不用 post-hoc 挑
- 訓練時間從 v2 ~6.5h(A100) 降到 ~3h

## 2. 訓練 cell(Colab,接續 phase1 §1.1–§1.5 環境)

```python
# ============================================================
# Stage 2 v3 訓練 — 共用 v2 stage1_best.pt, 純訓 Stage 2
# 前置: phase1_colab_workflow.md §1.1–§1.5 跑完
# ============================================================
import os, time, subprocess, threading, shlex
from pathlib import Path
from IPython import get_ipython

REPO         = '/content/NSVB-ZH'
LOCAL_S2     = '/content/stage2_v3_ckpts'
DRIVE_S1     = '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1'      # 共用
DRIVE_S2_V3  = '/content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage2_v3'   # 新
DRIVE_LOGS   = '/content/drive/MyDrive/NSVB-ZH/logs_v2'

for p in (LOCAL_S2, DRIVE_S2_V3, DRIVE_LOGS):
    os.makedirs(p, exist_ok=True)

def verify_drive_real(path):
    p = Path(path)
    assert p.exists() and not p.is_symlink() and p.is_dir(), \
        f"{path} 不是真目錄"
    s = p / '.write_test'; s.write_text('ok'); s.unlink()
verify_drive_real(DRIVE_S2_V3); verify_drive_real(DRIVE_LOGS)
print('✅ Drive paths verified')

# 背景 sync
_sync_stop = threading.Event()
def ckpt_sync_loop():
    while not _sync_stop.is_set():
        subprocess.run(['rsync', '-au', f'{LOCAL_S2}/', f'{DRIVE_S2_V3}/'],
                       capture_output=True)
        cnt_l = subprocess.check_output(f'ls {LOCAL_S2}/*.pt 2>/dev/null | wc -l',
                                         shell=True).decode().strip()
        cnt_d = subprocess.check_output(f'ls {DRIVE_S2_V3}/*.pt 2>/dev/null | wc -l',
                                         shell=True).decode().strip()
        print(f"[ckpt-sync {time.strftime('%H:%M:%S')}] "
              f"local={cnt_l} drive={cnt_d}", flush=True)
        if _sync_stop.wait(timeout=120): break
sync_thread = threading.Thread(target=ckpt_sync_loop, daemon=True)
sync_thread.start()

# resume 用:從 Drive 拉回已存的 ckpt
subprocess.run(['rsync', '-au', f'{DRIVE_S2_V3}/', f'{LOCAL_S2}/'], check=False)

# streaming runner(同 §7 模式)
def run_streamed(cmd_argv, log_path):
    cmd = ' '.join(shlex.quote(a) for a in cmd_argv)
    full = f'{cmd} 2>&1 | tee -a {shlex.quote(log_path)}'
    ip = get_ipython()
    ip.system(full)
    return int(ip.user_ns.get('_exit_code', '0'))

# 訓練
os.chdir(REPO)
ts = time.strftime('%Y%m%d_%H%M%S')
stage1_best = Path(DRIVE_S1) / 'stage1_best.pt'
assert stage1_best.exists(), f'Stage 1 best ckpt 不存在: {stage1_best}'

stage2_latest = Path(LOCAL_S2) / 'stage2_latest.pt'
cmd = [
    'env', 'PYTHONPATH=.',
    'python', '-m', 'nsvb.task.stage2',
    '--binarized-root', 'data/binarized',
    '--ppg-dim', '1280',
    '--batch-size', '16',
    '--num-workers', '4',
    '--max-steps', '50000',
    '--stage1-ckpt', str(stage1_best),
    '--ckpt-dir', LOCAL_S2,
    '--split-dir', 'data/binarized/splits',
    # v3 loss 調參
    '--lambda-patchnce', '2.0',
    '--lambda-identity-pro', '0.2',
    '--lambda-adv-mel', '0.1',
    '--d-z-warmup-steps', '10000',
    # 早停 hook
    '--val-eval-interval', '5000',
    '--val-eval-n-samples', '20',
    # 不加 --freeze-d-mel(v3 解凍 D_mel)
    # 不加 --lr-dz(用預設 4e-4)
]
if stage2_latest.exists():
    cmd += ['--resume', 'latest']
    print('▶️  Stage 2 v3 (resume)')
else:
    print('▶️  Stage 2 v3 (cold start, M from scratch)')
    print('   注意:v3 M 從零訓,**不繼承 v2 M**(loss 配方不同)')

log_path = f"{DRIVE_LOGS}/stage2_v3_{ts}.log"
print(f'   log → {log_path}')
rc = run_streamed(cmd, log_path)
print(f'Stage 2 v3 exit code: {rc}')

# 收尾
_sync_stop.set()
while sync_thread.is_alive():
    sync_thread.join(timeout=30)
subprocess.run(['rsync', '-au', '--info=stats2',
                f'{LOCAL_S2}/', f'{DRIVE_S2_V3}/'], check=True)
print('✅ Stage 2 v3 ckpts 已完整落 Drive (checkpoints_v2/stage2_v3)')
```

## 3. 訓中監控

每 50 步印一行訓 log,**每 5000 步額外印一行 val eval**:

```
[step  5000   3.1it/s] m_total=2.34 d_z=0.18 l_nce=0.51 l_adv_z=0.00 l_adv_mel=0.85 ...
[val-eval step 5000] pro_dir_align=+0.4521  ⭐ new best → saved stage2_best.pt
[step  10000  3.2it/s] m_total=2.51 ...
[val-eval step 10000] pro_dir_align=+0.5811  ⭐ new best → saved stage2_best.pt
[val-eval step 20000] pro_dir_align=+0.6601  ⭐ new best → saved stage2_best.pt
[val-eval step 25000] pro_dir_align=+0.6512  (best=+0.6601 @ step 20000)
```

`⭐ new best` = 該步 val alignment > 之前最高 → 自動複製 `stage2_latest.pt` → `stage2_best.pt`。
看不到 `⭐` 表示 best ckpt 維持不變。

## 4. 訓練後驗收

訓完落地 ckpts:
- `stage2_step5000.pt` ... `stage2_step50000.pt`(每 5K 一個)
- `stage2_latest.pt`(最後一步)
- **`stage2_best.pt`**(自動,by val pro_direction_alignment)

### 4.1 跑 test set full eval(跟 v2 用同套工具)

```bash
PYTHONPATH=. python scripts/stage2_ckpts_listening.py \
    --stage1-ckpt /content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage1/stage1_best.pt \
    --stage2-ckpt-dir /content/drive/MyDrive/NSVB-ZH/checkpoints_v2/stage2_v3 \
    --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized \
    --val-split /content/drive/MyDrive/NSVB-ZH/data/binarized/splits/test.txt \
    --out-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v3_listening_test_full \
    --all-samples --skip-vocoder --dump-mel --seed 42

PYTHONPATH=. python scripts/stage2_mel_eval.py \
    --listening-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v3_listening_test_full \
    --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized \
    --pro-mean-n 500 --max-viz 20 \
    --out-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_v3_eval_test_full
```

### 4.2 對 v2 比較 — v3 lock / drop 判定

| 指標(test, step 跟 best 取最高) | v2 baseline | v3 目標 | lock 條件 |
|---|---|---|---|
| `pro_direction_alignment` | +0.60 | **+0.65 ↑** | 顯著高於 v2(+0.05 以上) |
| M4 vs VV ratio | 11.26× | 維持 > 10× | 沒大幅退步 |
| `unvoiced_concentration` | 0.54 ✅ | < 0.55 | 健康 |
| `voiced_spectral_ratio` | 0.96 ✅ | ≥ 0.95 | 健康 |
| `tdr_extra` | +0.03 | < +0.05 | M 沒破壞時間結構 |
| `hf_extra` | -0.18 | ±0.20 內 | 高頻沒大改 |

- **全部達標** → v3 lock,更新 [phase2_outcome.md](phase2_outcome.md) v3 取代 v2
- **某項退步** → v3 drop,回 v2
- **alignment 持平或微升,其他指標退步** → 可考慮局部回退(例如降 `lambda_patchnce` 1.5)再訓一次,或乾脆放棄這條路徑

## 5. 預估時間

| 階段 | T4 | A100 |
|---|---|---|
| Stage 2 v3 (50K steps) | ~5h | ~3h |
| Test set full eval | ~3h | ~1.5h |
| **合計** | **~8h** | **~4.5h** |

T4 單 session 可能跑不完(12h limit + 加 setup),分兩段跑沒問題:
- Session 1:訓練到斷掉(用本 cell)
- Session 2:重連跑同 cell,自動 `--resume latest` 接續到 50K → 再跑 eval

## 6. 失敗回退路徑

若 v3 全跑完 alignment 沒進步,考慮這幾個方向(複雜度遞增):

| Direction | 工作量 | 動機 |
|---|---|---|
| **a. f0_support** | ~300 行 | 解 Risk §二.3 D_mel F0 confound 的根本解,v3 解凍 D_mel 若見 confound 回歸再加 |
| **b. M kernel_size=3** | 0 行(`--m-kernel-size 3`)| 給 M 加上時間感受野,可能對 voiced 段細節有幫助 |
| **c. Speaker augmentation** | ~500 行 | 直接解 generalization gap,但 Stage 1 也得重訓 |
| **d. f0_support + b** 組合 | ~300 行 | a + b 一起,跟 v3 三變數對照 |

優先序:b 最便宜先試,再考慮 a,c 是最後的招。
