"""PopBuTFy Stage 2 訓練 launcher — 完全沿用 v2 超參數,僅換 dataset / split / ckpt_dir。

【為什麼用 Python 而非 PowerShell】
PowerShell 5.1 對 native exe 之 stderr 有兩個 gotcha:
  1. `2>&1 | Tee-Object` 會把 stderr 每行包成 `System.Management.Automation.ErrorRecord`
     物件(NativeCommandError),讓「PyTorch FutureWarning 等正常 stderr 訊息」看起來
     像 PS 錯誤
  2. 配合 `$ErrorActionPreference = 'Stop'` 會直接終止整個 cell
Python subprocess 直接 merge stderr 進 stdout 並串流寫檔 / 印 console,完全避開上述。

【跟 v2 之異同】
  改:--amateur-dataset / --pro-dataset / --split-dir / --ckpt-dir
  不改(逐項對齊 v2 colab cell):
    --ppg-dim 1280 --batch-size 16 --num-workers 4
    --max-steps 120000
    --freeze-d-mel --lambda-adv-mel 0.05 --lr-dz 2e-4
    其餘 Stage2Config defaults(lr_m / lr_dmel / lambda_adv_z / lambda_patchnce /
    lambda_identity_pro / identity_pro_prob / d_z_warmup_steps / max_frames /
    latent_size / f0_support)全跟 v2 一致,不顯式覆寫

Stage 1 ckpt 沿用 v2 訓出之 stage1_best.pt。

【使用】
    python scripts/train_popbutfy_v2_params.py

中斷後重跑同樣命令會自動偵測 stage2_latest.pt 加 --resume latest。

Windows 注意:num-workers=4 對齊 v2(Colab)。若 Windows DataLoader 之 spawn 開銷
拖慢,可改本檔之 NUM_WORKERS = 2(僅影響 IO 速度,不影響收斂)。
"""
import os
import subprocess
import sys
import time
from pathlib import Path


# ── config ──
REPO        = Path(__file__).resolve().parent.parent
PYTHON      = r"C:\Users\neo29\miniconda3\envs\NSVB-ZH\python.exe"
CKPT_DIR    = REPO / "checkpoints" / "stage2_popbutfy"
LOG_DIR     = REPO / "logs"
STAGE1_CKPT = "checkpoints/stage1/stage1_best.pt"

NUM_WORKERS = 2  # v2 colab cell 用 4;若 Windows DataLoader 卡頓改 2


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"stage2_popbutfy_{ts}.log"

    cmd = [
        PYTHON, "-m", "nsvb.task.stage2",
        "--binarized-root", "data/binarized",
        "--amateur-dataset", "popbutfy_amateur",
        "--pro-dataset", "popbutfy_pro",
        "--ppg-dim", "1280",
        "--batch-size", "16",
        "--num-workers", str(NUM_WORKERS),
        "--max-steps", "120000",
        "--stage1-ckpt", STAGE1_CKPT,
        "--ckpt-dir", str(CKPT_DIR.relative_to(REPO)).replace("\\", "/"),
        "--split-dir", "data/binarized/splits_popbutfy",
        "--freeze-d-mel",
        "--lambda-adv-mel", "0.05",
        "--lr-dz", "2e-4",
    ]

    # resume 偵測 — 同 v2 colab cell 之 resume_flag2 邏輯
    latest = CKPT_DIR / "stage2_latest.pt"
    if latest.exists():
        cmd.extend(["--resume", "latest"])
        print(f"[launcher] 偵測 {latest},加 --resume latest", flush=True)

    print(f"[launcher] cwd  = {REPO}", flush=True)
    print(f"[launcher] log  = {log_path}", flush=True)
    print(f"[launcher] ckpt = {CKPT_DIR}", flush=True)
    print(f"[launcher] cmd  = {' '.join(cmd)}", flush=True)
    print("─" * 70, flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO)

    # subprocess 直接 merge stderr 進 stdout,逐行 stream 到 console + log
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            print("\n[launcher] KeyboardInterrupt — 終止 training subprocess", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        proc.wait()

    print("─" * 70, flush=True)
    print(f"[launcher] training subprocess exit={proc.returncode}", flush=True)
    print(f"[launcher] log saved: {log_path}", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()