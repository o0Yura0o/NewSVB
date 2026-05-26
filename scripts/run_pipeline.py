"""
scripts/run_pipeline.py
================================

【這支腳本做什麼】
NSVB-ZH 的「一鍵從零到 demo」orchestrator。一行指令把以下流程接起來:

    Phase 0: bootstrap_downloads → binarize_m4 → binarize_vv
             → verify_binarized → reconcile_vv → cluster_ppg
             → cluster_inspect → make_splits
             → gate_vocoder_identity → gate_audio_quality
    Phase 1: stage1 (Stage 1 CVAE 預訓練,~12-20 hr on RTX 4090)
    Phase 2: stage2 (Stage 2 mapping,~10-15 hr on RTX 4090)
    Phase 3: infer  (Mode A 推理 demo)
             mel_eval

【設計選擇 — 為什麼這麼做】

1. **subprocess 跑每個 stage(而非 in-process import)**
   - 訓練 stage 各自吃 ~16 GB GPU + ~20 GB RAM;in-process 跑完還會殘留
     CUDA context / cached tensor,下一 stage 容易 OOM。subprocess 跑完
     process 退出,記憶體乾淨。
   - 單一 stage 炸了 traceback 完整留在 log 檔,不會吃到別的 stage 的 import path 髒污。

2. **marker file(`run_state/{stage}.done`)+ idempotent**
   - 中斷後重跑同指令會自動 skip 已完成的 stage,只接著沒做完的開始。
   - 36-50 hr 的長跑,中間網路斷、Windows update 重開機是常態,不能讓使用者
     每次都從頭跑 binarize(那是 8-12 hr)。
   - `--force` 砍掉所有 marker 重跑;`--force-from {stage}` 從某 stage 開始重跑。

3. **每個 stage 有獨立 log(`run_state/{stage}.log`)**
   - subprocess stdout / stderr 都 line-buffered tee 到 log 與 terminal,
     失敗時自動印最後 60 行幫忙 debug。
   - 完整 log 留檔,後續 post-mortem 可以 `Get-Content`。

4. **Hard gate vs soft gate**
   - `gate_vocoder_identity` 跟 `gate_audio_quality` 是 Phase 0 的 quality gate
     (rebuild_checklist.md §H)。預設 fail 就 abort,避免拿爛資料跑 50 hr 訓練。
   - Risk §10 已記:VocalVerse 在 vocoder identity test 上會 fail
     (SSIM ~0.65, F0 RMSE ~53 Hz)。如果你已經接受這結論,加 `--skip-gates`
     繞過;或加 `--continue-on-gate-fail` 印警告但繼續。

5. **不放進預設流程的選項**
   - Plan B(f0_support)/ Plan D(pro-distribution matching)是 Stage 2 救火配方,
     主流程預設不啟用。要跑加 `--stage2-variant b` 或 `d`。
   - v3 早停 hook 預設啟用(`--val-eval-interval 5000`)因為它對 ~50k 步沒額外成本。

6. **強制 `--phoneme-vocab-size 100`**
   - Stage2Config 預設是 200(來自 NSVB en 版),但我們 `cluster_ppg --k 100`,
     不對齊會在 embedding lookup 階段炸。phase0_log.md 註明的健康配置即是 100。

【最少參數的指令】
    pip install gdown
    python scripts/run_pipeline.py

【常見變體】
    # 只跑到 Phase 0 結束(確認資料 OK 再決定要不要開長跑訓練)
    python scripts/run_pipeline.py --until make_splits

    # 假設你已經跑完 Phase 0,只想重跑 Stage 1 + Stage 2
    python scripts/run_pipeline.py --from stage1

    # 接受 VocalVerse vocoder identity 會 fail,繞過 gates
    python scripts/run_pipeline.py --skip-gates

    # 試跑(印出每個 stage 的 cmd,不真的執行)
    python scripts/run_pipeline.py --dry-run

    # 從某 stage 開始強制重跑(會清掉該 stage 之後的 marker)
    python scripts/run_pipeline.py --force-from cluster_ppg

    # 只跑單一 stage
    python scripts/run_pipeline.py --only verify_binarized
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional


# ── 路徑與常數 ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ 的上層
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_CKPT = REPO_ROOT / "checkpoints"
DEFAULT_STATE = REPO_ROOT / "run_state"

# 訓練超參(對應手冊 §2-3 與 phase0_log 健康配置)
CLUSTER_K = 100
PHONEME_VOCAB = 100
STAGE1_MAX_STEPS = 30_000
STAGE2_MAX_STEPS = 50_000
STAGE2_VAL_INTERVAL = 5_000


# ── 工具:跑 subprocess 並 tee 到 log ────────────────────
def run_cmd(cmd: list[str], log_path: Path, env: Optional[dict] = None) -> int:
    """
    跑 subprocess,stdout/stderr 同時 print + 寫 log_path。

    為什麼用 Popen + line-by-line:
        subprocess.run(capture_output=True) 會在記憶體裡 buffer 整個輸出,
        對訓練腳本(每步都 print loss)會吃幾 GB RAM。Popen + iter readline
        是 streaming,完全不 buffer。

    回傳 exit code(0 = 成功)。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # PYTHONPATH=. 讓 `python scripts/xxx.py` 找得到 nsvb package
    full_env.setdefault("PYTHONPATH", str(REPO_ROOT))
    # 強制 utf-8,Windows console cp950 對中文 / emoji 會炸
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    full_env.setdefault("PYTHONUNBUFFERED", "1")

    started = datetime.now()
    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n")
        lf.write(f"# started: {started.isoformat(timespec='seconds')}\n")
        lf.write(f"# cwd: {REPO_ROOT}\n")
        lf.write(f"# {'='*64}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
        ended = datetime.now()
        lf.write(f"# {'='*64}\n")
        lf.write(f"# ended: {ended.isoformat(timespec='seconds')}\n")
        lf.write(f"# elapsed: {ended - started}\n")
        lf.write(f"# exit: {proc.returncode}\n")
    return proc.returncode


def tail_log(log_path: Path, n: int = 60) -> str:
    """讀 log 最後 n 行(stage 失敗時印出來幫忙 debug)。"""
    if not log_path.exists():
        return "(no log file)"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def human_duration(td: timedelta) -> str:
    """把 timedelta 轉成人話 (e.g. '2h 13m')。"""
    sec = int(td.total_seconds())
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    h, rem = divmod(sec, 3600)
    m = rem // 60
    return f"{h}h {m}m"


# ── stage 定義 ─────────────────────────────────────────
@dataclass
class Stage:
    """
    每個 stage 的元資料。

    cmd_fn:回傳 list[str] cmd(讓 cmd 能引用 runtime config)
    marker:相對於 state_dir 的 marker 檔名(完成後 touch)
    check_fn:額外的 marker 檢查,若有則覆蓋預設 marker 機制
              (用於檢查目錄 / 多檔案存在比 single marker file 更可靠的情境)
    est_min:預估時間(分鐘),用於 ETA 顯示
    """
    name: str
    desc: str
    cmd_fn: Callable[["RunConfig"], list[str]]
    marker: str
    est_min: int = 0
    check_fn: Optional[Callable[["RunConfig"], bool]] = None


@dataclass
class RunConfig:
    """全局 runtime config,各 stage cmd_fn 從這裡讀。"""
    data_root: Path
    ckpt_root: Path
    state_dir: Path
    stage2_variant: str = "default"  # "default" | "b" | "d" | "freeze_d_mel"
    stage1_max_steps: int = STAGE1_MAX_STEPS
    stage2_max_steps: int = STAGE2_MAX_STEPS
    skip_inference: bool = False
    extra_infer_input: Optional[Path] = None  # 推理用的 amateur wav 路徑

    @property
    def vocoder_ckpt(self) -> Path:
        return (self.ckpt_root / "nsvb_pretrained"
                / "1012_hifigan_all_songs_nsf"
                / "model_ckpt_steps_1170000.ckpt")

    @property
    def vae_init_ckpt(self) -> Path:
        return (self.ckpt_root / "nsvb_pretrained" / "1030_vae_mle"
                / "model_ckpt_steps_200000.ckpt")

    @property
    def stage1_best(self) -> Path:
        return self.ckpt_root / "stage1" / "stage1_best.pt"

    @property
    def stage2_best(self) -> Path:
        return self.ckpt_root / "stage2" / "stage2_best.pt"


# ── 各 stage 的 cmd 構造 ─────────────────────────────
def cmd_bootstrap(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/bootstrap_downloads.py",
        "--data-root", str(cfg.data_root),
        "--ckpt-root", str(cfg.ckpt_root),
    ]


def cmd_binarize_m4(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "-m", "nsvb.data.binarizer",
        "--dataset", "m4singer",
        "--data-root", str(cfg.data_root),
        "--out-root", str(cfg.data_root / "binarized"),
    ]


def cmd_binarize_vv(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "-m", "nsvb.data.binarizer",
        "--dataset", "vocalverse",
        "--data-root", str(cfg.data_root),
        "--out-root", str(cfg.data_root / "binarized"),
        "--vocalverse-amateur-score-max", "3.0",
        "--vocalverse-chunk-sec", "5.0",
    ]


def cmd_verify(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/verify_binarized.py",
        "--root", str(cfg.data_root / "binarized"),
        "--dataset", "m4singer", "vocalverse",
    ]


def cmd_reconcile_vv(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/reconcile_vv_chunks.py",
        "--vv-source", str(cfg.data_root / "VocalVerse"),
        "--binarized-root", str(cfg.data_root / "binarized"),
        "--amateur-score-max", "3.0",
        "--chunk-sec", "5.0",
    ]


def cmd_cluster_ppg(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "-m", "nsvb.data.cluster_ppg",
        "--k", str(CLUSTER_K),
        "--per-utt-mean-norm",
        "--stage", "all",
        "--binarized-root", str(cfg.data_root / "binarized"),
    ]


def cmd_cluster_inspect(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/cluster_ppg_inspect.py",
        "--binarized-root", str(cfg.data_root / "binarized"),
    ]


def cmd_make_splits(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/make_splits.py",
        "--m4-test-singers", "Alto-2", "Tenor-3",
        "--seed", "42",
        "--binarized-root", str(cfg.data_root / "binarized"),
    ]


def cmd_gate_vocoder(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/vocoder_identity_test.py",
        "--vocoder-ckpt", str(cfg.vocoder_ckpt),
        "--wav-dirs",
        f"m4={cfg.data_root / 'm4singer'}",
        f"vocalverse={cfg.data_root / 'VocalVerse'}",
        "--f0-method", "parselmouth",
        "--f0-interp",
        "--apply-loudness-norm",
        "--save-wavs",
    ]


def cmd_gate_audio(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/audio_quality_probe.py",
        "--wav-dirs",
        f"m4={cfg.data_root / 'm4singer'}",
        f"vocalverse={cfg.data_root / 'VocalVerse'}",
        "--apply-dereverb",
    ]


def cmd_stage1(cfg: RunConfig) -> list[str]:
    cmd = [
        sys.executable, "-m", "nsvb.task.stage1",
        "--init-from-nsvb", str(cfg.vae_init_ckpt),
        "--max-steps", str(cfg.stage1_max_steps),
        "--binarized-root", str(cfg.data_root / "binarized"),
        "--ckpt-dir", str(cfg.ckpt_root / "stage1"),
    ]
    return cmd


def cmd_stage2(cfg: RunConfig) -> list[str]:
    base = [
        sys.executable, "-m", "nsvb.task.stage2",
        "--stage1-ckpt", str(cfg.stage1_best),
        "--phoneme-vocab-size", str(PHONEME_VOCAB),
        "--max-steps", str(cfg.stage2_max_steps),
        "--val-eval-interval", str(STAGE2_VAL_INTERVAL),
        "--binarized-root", str(cfg.data_root / "binarized"),
        "--ckpt-dir", str(cfg.ckpt_root / "stage2"),
    ]
    if cfg.stage2_variant == "b":
        base.append("--f0-support")
    elif cfg.stage2_variant == "d":
        base.append("--pro-match")
    elif cfg.stage2_variant == "freeze_d_mel":
        base.extend(["--freeze-d-mel-after", "10000"])
    return base


def cmd_infer(cfg: RunConfig) -> list[str]:
    # 若使用者沒指定 amateur wav,從 binarized test set 撈一個當 demo
    input_wav = cfg.extra_infer_input
    if input_wav is None:
        # 試著從 binarized test 抓第一首 amateur,這支腳本內部處理「找不到就 skip」
        input_wav = cfg.data_root / "VocalVerse"  # placeholder;infer 腳本支援目錄
    out_dir = cfg.ckpt_root / "stage2" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "-m", "scripts.infer",
        "--stage2-ckpt", str(cfg.stage2_best),
        "--vocoder-ckpt", str(cfg.vocoder_ckpt),
        "--input-a", str(input_wav),
        "--output", str(out_dir / "demo_modeA.wav"),
    ]


def cmd_mel_eval(cfg: RunConfig) -> list[str]:
    return [
        sys.executable, "scripts/stage2_mel_eval.py",
        "--stage2-ckpt", str(cfg.stage2_best),
        "--binarized-root", str(cfg.data_root / "binarized"),
        "--split", "test",
    ]


# ── check_fn:某些 stage 用「目錄 / 多檔案」做完成判斷 ─────
def check_stage1_done(cfg: RunConfig) -> bool:
    # stage1 task 完成後應該有 stage1_best.pt;也可能寫成 model_ckpt_steps_*.pt
    return cfg.stage1_best.exists() or bool(
        list((cfg.ckpt_root / "stage1").glob("*.pt")) if (cfg.ckpt_root / "stage1").exists() else []
    )


def check_stage2_done(cfg: RunConfig) -> bool:
    return cfg.stage2_best.exists() or bool(
        list((cfg.ckpt_root / "stage2").glob("*.pt")) if (cfg.ckpt_root / "stage2").exists() else []
    )


# ── stage 註冊表 ─────────────────────────────────────
PIPELINE: list[Stage] = [
    # ── Phase 0 ──
    Stage("bootstrap",       "下載 dataset + pretrained ckpt",
          cmd_bootstrap,     "bootstrap.done",     est_min=120),
    Stage("binarize_m4",     "M4Singer 整首 binarize",
          cmd_binarize_m4,   "binarize_m4.done",   est_min=180),
    Stage("binarize_vv",     "VocalVerse 切 chunk + binarize",
          cmd_binarize_vv,   "binarize_vv.done",   est_min=480),
    Stage("verify_binarized", "驗證 binarize 產物完整性",
          cmd_verify,        "verify_binarized.done", est_min=5),
    Stage("reconcile_vv",    "VV chunk 對照 source filter rule",
          cmd_reconcile_vv,  "reconcile_vv.done",  est_min=10),
    Stage("cluster_ppg",     f"PPG k-means k={CLUSTER_K} + per-utt mean norm",
          cmd_cluster_ppg,   "cluster_ppg.done",   est_min=30),
    Stage("cluster_inspect", "cluster 健康度檢查",
          cmd_cluster_inspect, "cluster_inspect.done", est_min=2),
    Stage("make_splits",     "建 train/val/test splits",
          cmd_make_splits,   "make_splits.done",   est_min=2),
    Stage("gate_vocoder",    "Gate ①: vocoder identity test (※ VV 預期 fail, risk §10)",
          cmd_gate_vocoder,  "gate_vocoder.done",  est_min=15),
    Stage("gate_audio",      "Gate ②: audio quality probe",
          cmd_gate_audio,    "gate_audio.done",    est_min=10),
    # ── Phase 1 ──
    Stage("stage1",          f"Stage 1 CVAE 預訓練 ({STAGE1_MAX_STEPS} 步)",
          cmd_stage1,        "stage1.done",        est_min=900,
          check_fn=check_stage1_done),
    # ── Phase 2 ──
    Stage("stage2",          f"Stage 2 mapping ({STAGE2_MAX_STEPS} 步)",
          cmd_stage2,        "stage2.done",        est_min=720,
          check_fn=check_stage2_done),
    # ── Phase 3 ──
    Stage("infer",           "Mode A 推理 demo",
          cmd_infer,         "infer.done",         est_min=5),
    Stage("mel_eval",        "mel-domain 評估(繞過 vocoder)",
          cmd_mel_eval,      "mel_eval.done",      est_min=10),
]
GATE_STAGES = {"gate_vocoder", "gate_audio"}  # 失敗會 abort 的硬性 gate


# ── runtime ───────────────────────────────────────────
def stage_done(stage: Stage, cfg: RunConfig) -> bool:
    """判一個 stage 是否已完成。優先 check_fn,沒有就看 marker file。"""
    if stage.check_fn is not None:
        return stage.check_fn(cfg)
    return (cfg.state_dir / stage.marker).exists()


def mark_done(stage: Stage, cfg: RunConfig) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / stage.marker).write_text(
        datetime.now().isoformat(timespec="seconds")
    )


def clear_marker(stage: Stage, cfg: RunConfig) -> None:
    p = cfg.state_dir / stage.marker
    if p.exists():
        p.unlink()


def plan_stages(args: argparse.Namespace) -> list[Stage]:
    """依 CLI flags 決定要跑哪些 stage。"""
    names = [s.name for s in PIPELINE]

    if args.only:
        for n in args.only:
            if n not in names:
                raise SystemExit(f"未知 stage: {n}\n可選: {names}")
        return [s for s in PIPELINE if s.name in args.only]

    start = 0
    end = len(PIPELINE)
    if args.from_:
        if args.from_ not in names:
            raise SystemExit(f"未知 --from stage: {args.from_}\n可選: {names}")
        start = names.index(args.from_)
    if args.until:
        if args.until not in names:
            raise SystemExit(f"未知 --until stage: {args.until}\n可選: {names}")
        end = names.index(args.until) + 1
    return PIPELINE[start:end]


def confirm_long_run(cfg: RunConfig, plan: list[Stage]) -> None:
    """總時間估計;若 >2 hr 提示 user 並等 enter 確認。"""
    total = sum(s.est_min for s in plan)
    if total < 120:
        return
    print(f"\n[plan] 總估計時間 ≈ {human_duration(timedelta(minutes=total))}")
    print(f"       建議用 nohup / screen / tmux 跑,避免 SSH 斷線中斷訓練。")
    print(f"       Windows PowerShell 可用 `Start-Process` 背景跑。")
    if sys.stdin.isatty():
        input("       Press Enter to start, Ctrl-C to cancel ... ")


def main() -> None:
    # Windows console 預設 cp950,中文 print 會炸,強制 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="NSVB-ZH 一鍵 Phase 0 → Phase 3 orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA),
                        help="dataset 目錄 (預設 data/)")
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT),
                        help="ckpt 目錄 (預設 checkpoints/)")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE),
                        help="marker / log 目錄 (預設 run_state/)")

    parser.add_argument("--from", dest="from_", default=None,
                        help="從某 stage 開始 (e.g. --from stage1)")
    parser.add_argument("--until", default=None,
                        help="跑到某 stage 結束(含)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="只跑指定 stage(可多個)")

    parser.add_argument("--force", action="store_true",
                        help="清掉所有 marker,從頭重跑")
    parser.add_argument("--force-from", default=None,
                        help="清掉指定 stage(含)之後的 marker,從那邊重跑")

    parser.add_argument("--stage2-variant", default="default",
                        choices=["default", "b", "d", "freeze_d_mel"],
                        help="Stage 2 變體:default / b (f0_support) / d (pro_match) "
                             "/ freeze_d_mel (救火配方)")
    parser.add_argument("--stage1-max-steps", type=int, default=STAGE1_MAX_STEPS)
    parser.add_argument("--stage2-max-steps", type=int, default=STAGE2_MAX_STEPS)

    parser.add_argument("--skip-gates", action="store_true",
                        help="完全略過 Phase 0 兩個 gate(risk §10 知情者用)")
    parser.add_argument("--continue-on-gate-fail", action="store_true",
                        help="Gate fail 印警告但繼續(預設是 abort)")

    parser.add_argument("--infer-input", default=None,
                        help="Phase 3 推理用的 amateur wav 路徑")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跑完訓練不做 Phase 3 推理")

    parser.add_argument("--dry-run", action="store_true",
                        help="只印每個 stage 會跑什麼,不真的執行")
    parser.add_argument("--no-confirm", action="store_true",
                        help="長跑提示時不要 prompt(非互動環境用)")

    args = parser.parse_args()

    cfg = RunConfig(
        data_root=Path(args.data_root).resolve(),
        ckpt_root=Path(args.ckpt_root).resolve(),
        state_dir=Path(args.state_dir).resolve(),
        stage2_variant=args.stage2_variant,
        stage1_max_steps=args.stage1_max_steps,
        stage2_max_steps=args.stage2_max_steps,
        skip_inference=args.skip_inference,
        extra_infer_input=Path(args.infer_input).resolve() if args.infer_input else None,
    )
    cfg.state_dir.mkdir(parents=True, exist_ok=True)

    # 計算 plan
    plan = plan_stages(args)
    if args.skip_gates:
        plan = [s for s in plan if s.name not in GATE_STAGES]
    if cfg.skip_inference:
        plan = [s for s in plan if s.name != "infer"]

    # marker 清理
    if args.force:
        for s in PIPELINE:
            clear_marker(s, cfg)
        print("[reset] cleared all markers")
    if args.force_from:
        names = [s.name for s in PIPELINE]
        if args.force_from not in names:
            raise SystemExit(f"未知 --force-from stage: {args.force_from}")
        idx = names.index(args.force_from)
        for s in PIPELINE[idx:]:
            clear_marker(s, cfg)
        print(f"[reset] cleared markers from '{args.force_from}' onward")

    # 印 plan
    print(f"\n{'='*64}")
    print(f"  NSVB-ZH pipeline · {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'='*64}")
    print(f"  data_root = {cfg.data_root}")
    print(f"  ckpt_root = {cfg.ckpt_root}")
    print(f"  state_dir = {cfg.state_dir}")
    print(f"  python    = {sys.executable}")
    print(f"\n  plan ({len(plan)} stages):")
    for s in plan:
        done = stage_done(s, cfg)
        flag = "[done]" if done else "     "
        eta = f" ~{human_duration(timedelta(minutes=s.est_min))}" if s.est_min else ""
        print(f"    {flag}  {s.name:20s}  {s.desc}{eta}")
    print(f"{'='*64}\n")

    if args.dry_run:
        print("[dry-run] cmds:")
        for s in plan:
            print(f"\n## {s.name}")
            print("  " + " ".join(s.cmd_fn(cfg)))
        return

    if not args.no_confirm:
        confirm_long_run(cfg, [s for s in plan if not stage_done(s, cfg)])

    # 開跑
    overall_start = datetime.now()
    failures: list[tuple[str, int]] = []
    for s in plan:
        if stage_done(s, cfg):
            print(f"\n[skip]  {s.name:20s}  already done")
            continue

        print(f"\n{'─'*64}")
        print(f"[start] {s.name}  · {s.desc}")
        print(f"        ETA ~{human_duration(timedelta(minutes=s.est_min))}"
              if s.est_min else "")
        print(f"{'─'*64}")
        cmd = s.cmd_fn(cfg)
        log_path = cfg.state_dir / f"{s.name}.log"
        t0 = datetime.now()
        rc = run_cmd(cmd, log_path)
        elapsed = datetime.now() - t0

        if rc == 0:
            mark_done(s, cfg)
            print(f"\n[done]  {s.name}  · {human_duration(elapsed)}")
            continue

        # 失敗處理
        print(f"\n[FAIL]  {s.name}  · exit={rc}  · {human_duration(elapsed)}")
        print(f"        log: {log_path}")
        print(f"        --- last 60 lines ---")
        print(tail_log(log_path, 60))
        print(f"        --- end of log ---")
        failures.append((s.name, rc))

        # gate 失敗:依 flag 決定要不要繼續
        if s.name in GATE_STAGES and not args.continue_on_gate_fail:
            print(f"\n[abort] gate '{s.name}' failed. 修好或加 --continue-on-gate-fail 繼續。")
            sys.exit(rc)
        # 訓練 stage 失敗則直接 abort,避免後續用爛 ckpt
        if s.name in {"stage1", "stage2"}:
            print(f"\n[abort] training stage '{s.name}' failed. 看 log 後重跑同指令續訓。")
            sys.exit(rc)
        # 其餘 stage 失敗也 abort,讓 user 看 log
        print(f"\n[abort] stage '{s.name}' failed.")
        sys.exit(rc)

    overall = datetime.now() - overall_start
    print(f"\n{'='*64}")
    if failures:
        print(f"[pipeline] DONE with failures: {failures}")
        sys.exit(1)
    print(f"[pipeline] ALL DONE ✓  · 總耗時 {human_duration(overall)}")
    print(f"{'='*64}")
    print(f"\n下一步:")
    print(f"  - listen demo: {cfg.ckpt_root / 'stage2' / 'generated' / 'demo_modeA.wav'}")
    print(f"  - mel eval log: {cfg.state_dir / 'mel_eval.log'}")
    print(f"  - 跨 step 聽測 dump: python scripts/dump_listening_set.py "
          f"--ckpt-dir {cfg.ckpt_root / 'stage2'}")


if __name__ == "__main__":
    main()
