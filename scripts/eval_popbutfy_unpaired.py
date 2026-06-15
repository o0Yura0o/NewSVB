"""PopBuTFy 跨語言驗證:跑 unpaired eval(對標 v2 之 +0.84)。

對標 v2 (Colab) 之兩階段 eval 流程:
  Phase 1: scripts.stage2_ckpts_listening --all-samples --skip-vocoder --dump-mel
           → 對全 test set 逐 step 跑 Mode A,dump _N_label.mel.npy 到 listening_dir
           (skip_vocoder 省 ~85% 時間,因為 metric 只看 mel 不需要 wav)
  Phase 2: scripts.stage2_mel_eval --listening-dir <Phase 1 出>
           → 從 dumped mels 算 pro_direction_alignment 等指標,出 report.md

【為什麼用 Python launcher 而非 PowerShell】
跟 train_popbutfy_v2_params.py 同理:避開 PS 對 native exe stderr 之
NativeCommandError 包裝(配 $ErrorActionPreference=Stop 會把正常 warning 升格成
terminating error)。subprocess 直接 merge stderr 進 stdout,逐行 stream 到 console
+ log,完全跨過。

【使用】
    python scripts/eval_popbutfy_unpaired.py

中斷後重跑會 skip 已完成之 sample(stage2_ckpts_listening 之 dump 流程靠檔案
存在 idempotent)。
"""
import os
import subprocess
import sys
import time
from pathlib import Path


REPO         = Path(__file__).resolve().parent.parent
PYTHON       = r"C:\Users\neo29\miniconda3\envs\NSVB-ZH\python.exe"
STAGE1_CKPT  = "checkpoints/stage1/stage1_best.pt"
STAGE2_DIR   = "checkpoints/stage2_popbutfy"
VAL_SPLIT    = "data/binarized/splits_popbutfy/test.txt"
STEPS        = "5000,15000,30000,50000,70000,90000,110000,120000"

LISTENING_DIR = REPO / "outputs" / "stage2_popbutfy_listening_test_full"
EVAL_DIR      = REPO / "outputs" / "stage2_popbutfy_eval"


def run_phase(name: str, cmd: list, log_path: Path):
    print(f"\n{'═' * 70}", flush=True)
    print(f"[phase] {name}", flush=True)
    print(f"[phase] cmd = {' '.join(cmd)}", flush=True)
    print(f"[phase] log = {log_path}", flush=True)
    print(f"{'═' * 70}", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            encoding="utf-8", errors="replace",
            env=env,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            print(f"\n[phase {name}] KeyboardInterrupt — terminate subprocess", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        proc.wait()
    elapsed = (time.time() - t0) / 60
    ok = (proc.returncode == 0)
    print(f"\n[phase] {name} exit={proc.returncode}, {elapsed:.1f} min", flush=True)
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    LISTENING_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Phase 1: dump mels(全 test set × 全 step,no vocoder)──
    cmd1 = [
        PYTHON, "-m", "scripts.stage2_ckpts_listening",
        "--stage1-ckpt", STAGE1_CKPT,
        "--stage2-ckpt-dir", STAGE2_DIR,
        "--binarized-root", "data/binarized",
        "--val-split", VAL_SPLIT,
        "--steps", STEPS,
        "--all-samples",
        "--skip-vocoder",
        "--dump-mel",
        "--out-dir", str(LISTENING_DIR.relative_to(REPO)).replace("\\", "/"),
    ]
    ok1 = run_phase("dump-mels (stage2_ckpts_listening)",
                    cmd1, log_dir / f"eval_popbutfy_dump_{ts}.log")
    if not ok1:
        print("[main] Phase 1 失敗,abort", flush=True)
        sys.exit(1)

    # ── Phase 2: compute metrics(mode A 讀 dumped mels)──
    cmd2 = [
        PYTHON, "-m", "scripts.stage2_mel_eval",
        "--listening-dir", str(LISTENING_DIR.relative_to(REPO)).replace("\\", "/"),
        "--binarized-root", "data/binarized",
        "--pro-dataset", "popbutfy_pro",
        "--pro-mean-n", "500",     # 跟 v2 fulltest eval SEM 設定一致
        "--max-viz", "20",         # 只 render 20 張 mel_grid.png,其餘只算指標
        "--out-dir", str(EVAL_DIR.relative_to(REPO)).replace("\\", "/"),
    ]
    ok2 = run_phase("compute-metrics (stage2_mel_eval mode A)",
                    cmd2, log_dir / f"eval_popbutfy_metrics_{ts}.log")
    if not ok2:
        print("[main] Phase 2 失敗,abort", flush=True)
        sys.exit(2)

    print(f"\n{'═' * 70}", flush=True)
    print(f"[done] report: {EVAL_DIR / 'report.md'}", flush=True)
    print(f"{'═' * 70}", flush=True)


if __name__ == "__main__":
    main()