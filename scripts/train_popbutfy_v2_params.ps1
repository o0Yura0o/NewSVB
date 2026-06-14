# PopBuTFy Stage 2 訓練 launcher — 完全沿用 v2 超參數,僅換 dataset / split / ckpt_dir
#
# 跟 v2 之異同:
#   ★ 改:--amateur-dataset / --pro-dataset / --split-dir / --ckpt-dir
#   ★ 不改(逐項對齊 v2 colab cell):
#       --ppg-dim 1280 --batch-size 16 --num-workers 4
#       --max-steps 120000
#       --freeze-d-mel --lambda-adv-mel 0.05 --lr-dz 2e-4
#       其餘 Stage2Config defaults(lr_m=1e-4 / lr_dmel=1e-5 / lambda_adv_z=1.0 /
#       lambda_patchnce=1.0 / lambda_identity_pro=0.1 / identity_pro_prob=0.2 /
#       d_z_warmup_steps=5000 / max_frames=1500 / latent_size=128 / f0_support=none)
#       全跟 v2 預設一致,不用顯式覆寫
#
# Stage 1 ckpt 直接沿用 v2 訓出之 stage1_best.pt
#   - 該 ckpt 原本就是基於 NSVB 英文預訓(1030_vae_mle on PopBuTFy)做中文 fine-tune
#   - 回頭跑 PopBuTFy 等於部分 recover 原英文知識,當 frozen feature extractor 使用,合理
#
# 預估 6.5 小時 on RTX 3070(本機;scripts/estimate_popbutfy_training_time.py 量過)
#
# 中斷後恢復:此 launcher 偵測 stage2_latest.pt 自動加 --resume latest(同 v2 colab cell 邏輯)
#
# Windows 注意:num-workers 4 與 v2(Colab)對齊。若 Windows DataLoader 之 shm/spawn 開銷
# 拖慢明顯,可手動把 --num-workers 4 改為 2(僅影響資料載入速度,不影響模型收斂)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = '.'

$Ts = Get-Date -Format yyyyMMdd_HHmmss
$LogFile = "logs/stage2_popbutfy_$Ts.log"
$CkptDir = "checkpoints/stage2_popbutfy"

# v2 cell 邏輯:已有 stage2_latest.pt → --resume latest
$Stage2Latest = Join-Path $CkptDir 'stage2_latest.pt'
$ResumeArgs = @()
if (Test-Path $Stage2Latest) {
    $ResumeArgs = @('--resume', 'latest')
    Write-Host "[launcher] 偵測 $Stage2Latest,加 --resume latest" -ForegroundColor Yellow
}

Write-Host "[launcher] log  -> $LogFile" -ForegroundColor Cyan
Write-Host "[launcher] ckpt -> $CkptDir/" -ForegroundColor Cyan

& C:\Users\neo29\miniconda3\envs\NSVB-ZH\python.exe -m nsvb.task.stage2 `
    --binarized-root data/binarized `
    --amateur-dataset popbutfy_amateur `
    --pro-dataset popbutfy_pro `
    --ppg-dim 1280 --batch-size 16 --num-workers 4 `
    --max-steps 120000 `
    --stage1-ckpt checkpoints/stage1/stage1_best.pt `
    --ckpt-dir $CkptDir `
    --split-dir data/binarized/splits_popbutfy `
    --freeze-d-mel --lambda-adv-mel 0.05 --lr-dz 2e-4 `
    @ResumeArgs `
    2>&1 | Tee-Object -FilePath $LogFile