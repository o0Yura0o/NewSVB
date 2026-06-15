"""Summarize Stage 2 training log into a readable markdown report.

Usage:
    python scripts/summarize_stage2_log.py runs/stage2_v2_20260517_151527.log
    # → 同目錄產出 stage2_v2_20260517_151527.summary.md

    # 指定輸出路徑:
    python scripts/summarize_stage2_log.py -o my_summary.md path/to/training.log

讀進 tqdm/tee 後行很長的 log,抽出:
  - 每個 [step N rate] 訓練 log 列
  - [stage2-monitor step N] 音質監控列
  - Failure mode A/B 警告
產出 markdown summary:
  - 關鍵 milestone(50, 500, warmup 結束, 每 10K)
  - 健康判定(對照 training_flow.md §3.6.1)
  - 各 metric 末尾 10K 步穩態值
  - first-crossing 時間軸(Δ/z 何時破 0.30、tdr 何時破 0.3、l_adv_mel 何時轉負)
"""
import argparse
import re
from pathlib import Path
from statistics import mean
from typing import Optional

# ── 訓練 log 列(由 stage2.py train_step log 印出) ──
# format: [step  12345 5.4it/s] m_total=.. d_z=.. d_mel=.. l_nce=.. l_adv_z=..
#         l_adv_mel=.. [l_pro_match=..] l_id_pro=.. delta_over_z=.. temporal_diff_ratio=..
# 為什麼 l_pro_match 用 (?:...)?:此欄是後加入的(predicate v3 後),舊 log 沒這欄,
# 用 optional non-capturing group 確保新舊 log 都能 parse;v2-style config 此值恆 0.0
STEP_RE = re.compile(
    r"\[step\s+(\d+)\s+([\d.]+)it/s\]\s+"
    r"m_total=(-?[\d.]+)\s+"
    r"d_z=(-?[\d.]+)\s+"
    r"d_mel=(-?[\d.]+)\s+"
    r"l_nce=(-?[\d.]+)\s+"
    r"l_adv_z=(-?[\d.]+)\s+"
    r"l_adv_mel=(-?[\d.]+)\s+"
    r"(?:l_pro_match=-?[\d.]+\s+)?"
    r"l_id_pro=(-?[\d.]+)\s+"
    r"delta_over_z=(-?[\d.]+)\s+"
    r"temporal_diff_ratio=(-?[\d.]+)"
)

# ── 音質監控列 ──
# [stage2-monitor step N] Δ_voiced_E=.. Δ_unvoiced_E=.. unvoiced_concentration=.. (...) voiced_spectral_ratio=.. (...)
MONITOR_RE = re.compile(
    r"\[stage2-monitor step\s+(\d+)\][^|]*?"
    r"Δ_voiced_E=([\d.eE+\-]+)\s+"
    r"Δ_unvoiced_E=([\d.eE+\-]+)\s+"
    r"unvoiced_concentration=([\d.]+)[^v]*"
    r"voiced_spectral_ratio=([\d.]+)"
)

# ── 健康範圍(來自 training_flow.md §3.6.1)──
HEALTH = {
    "delta_over_z":          {"healthy_lo": 0.03, "healthy_hi": 0.30,
                              "fail_a": 0.03, "fail_b": 0.30,
                              "desc": "M 對 latent 的修飾量"},
    "temporal_diff_ratio":   {"healthy_lo": 0.0,  "healthy_hi": 0.30,
                              "fail_b": 0.30,
                              "desc": "M 改動時間導數 / z 自身時間導數"},
    "unvoiced_concentration":{"healthy_lo": 0.0,  "healthy_hi": 0.55,
                              "fail_b": 0.65,
                              "desc": "Δ 集中在 unvoiced 段的比例(去殘響/去呼吸聲警訊)"},
    "voiced_spectral_ratio": {"healthy_lo": 0.70, "healthy_hi": 1.0,
                              "fail_b_low": 0.40,
                              "desc": "voiced 段 Δ 低時間頻率成分占比(envelope shift 主導)"},
}


def parse_steps(text: str) -> list[dict]:
    rows = []
    for m in STEP_RE.finditer(text):
        rows.append({
            "step":              int(m.group(1)),
            "it_per_s":          float(m.group(2)),
            "m_total":           float(m.group(3)),
            "d_z":               float(m.group(4)),
            "d_mel":             float(m.group(5)),
            "l_nce":             float(m.group(6)),
            "l_adv_z":           float(m.group(7)),
            "l_adv_mel":         float(m.group(8)),
            "l_id_pro":          float(m.group(9)),
            "delta_over_z":      float(m.group(10)),
            "temporal_diff_ratio": float(m.group(11)),
        })
    return rows


def parse_monitor(text: str) -> list[dict]:
    rows = []
    for m in MONITOR_RE.finditer(text):
        rows.append({
            "step":                  int(m.group(1)),
            "delta_voiced_e":        float(m.group(2)),
            "delta_unvoiced_e":      float(m.group(3)),
            "unvoiced_concentration": float(m.group(4)),
            "voiced_spectral_ratio": float(m.group(5)),
        })
    return rows


def parse_warnings(text: str) -> list[str]:
    """抓 Failure mode A/B 一次性警告。"""
    out = []
    for line in text.splitlines():
        if "Failure mode" in line and "⚠" in line:
            out.append(line.strip())
    return out


def parse_config_hints(text: str) -> dict:
    """從 [stage2] init log 推回本次 config 重點。"""
    out = {}
    for line in text.splitlines():
        if "lambda(adv_z/adv_mel/patchnce)" in line:
            m = re.search(r"=([\d./e\-]+)", line)
            if m:
                out["lambda_str"] = m.group(1)
        if "lr(M/Dz/Dmel)" in line:
            m = re.search(r"lr\(M/Dz/Dmel\)=([\w./e\-]+)", line)
            if m:
                out["lr_str"] = m.group(1)
        if "D_mel real source" in line:
            out["d_mel_source"] = line.split("D_mel real source:", 1)[1].strip()
        if line.startswith("[stage2] init done"):
            out["init_line"] = line.strip()
    return out


def verdict(value: float, metric: str) -> str:
    """根據健康範圍給單值評語。"""
    spec = HEALTH.get(metric)
    if not spec:
        return ""
    if metric == "delta_over_z":
        if value < spec["fail_a"]:
            return f"❌ Failure mode A (< {spec['fail_a']}, M 太保守)"
        if value > spec["fail_b"]:
            return f"❌ 過度激進 (> {spec['fail_b']}, 健康上限)"
        return f"✅ healthy ({spec['healthy_lo']}-{spec['healthy_hi']})"
    if metric == "temporal_diff_ratio":
        if value > spec["fail_b"]:
            return f"❌ 時間結構受損 (> {spec['fail_b']}, 健康上限)"
        return f"✅ healthy (< {spec['fail_b']})"
    if metric == "unvoiced_concentration":
        if value > spec["fail_b"]:
            return f"❌ Risk 2 警訊 (> {spec['fail_b']})"
        if value > spec["healthy_hi"]:
            return f"⚠️  邊界 ({spec['healthy_hi']}-{spec['fail_b']})"
        return f"✅ healthy (< {spec['healthy_hi']})"
    if metric == "voiced_spectral_ratio":
        if value < spec["fail_b_low"]:
            return f"❌ 改高頻時間振盪 (< {spec['fail_b_low']}, M 改 F0 trajectory 警訊)"
        if value < spec["healthy_lo"]:
            return f"⚠️  marginal ({spec['fail_b_low']}-{spec['healthy_lo']})"
        return f"✅ envelope-dominated (≥ {spec['healthy_lo']})"
    return ""


def first_crossing(rows: list[dict], metric: str, threshold: float,
                   direction: str = "above") -> Optional[int]:
    """找 metric 首次跨過 threshold 的 step。direction = "above" 或 "below"。"""
    for r in rows:
        v = r[metric]
        if direction == "above" and v > threshold:
            return r["step"]
        if direction == "below" and v < threshold:
            return r["step"]
    return None


def pick_milestones(rows: list[dict], warmup_step: int = 5000) -> list[dict]:
    """挑出有意義的 milestone:warmup 前細抽,warmup 後每 10K。"""
    if not rows:
        return []
    targets = [50, 100, 250, 500, 1000, 2500]
    targets.append(warmup_step)
    last_step = rows[-1]["step"]
    t = ((warmup_step // 1000) + 5) * 1000  # warmup_step + 5K
    while t <= last_step:
        targets.append(t)
        t += 10_000
    if last_step not in targets:
        targets.append(last_step)

    # 為每個 target 找最接近的實際 step
    out = []
    seen = set()
    for tgt in targets:
        best = min(rows, key=lambda r: abs(r["step"] - tgt))
        if best["step"] not in seen:
            out.append(best)
            seen.add(best["step"])
    return sorted(out, key=lambda r: r["step"])


def tail_steady_state(rows: list[dict], frac: float = 0.10) -> dict:
    """取最後 frac 比例步數的平均 + min/max,代表穩態值。"""
    if not rows:
        return {}
    n_tail = max(1, int(len(rows) * frac))
    tail = rows[-n_tail:]
    keys = ["m_total", "d_z", "l_nce", "l_adv_z", "l_adv_mel",
            "delta_over_z", "temporal_diff_ratio"]
    return {
        k: {"mean": mean(r[k] for r in tail),
            "min":  min(r[k] for r in tail),
            "max":  max(r[k] for r in tail)}
        for k in keys
    }


def overall_verdict(steady: dict) -> str:
    """根據穩態值給整體訓練評語。"""
    if not steady:
        return "❓ 無足夠資料"
    dz = steady["delta_over_z"]["mean"]
    tdr = steady["temporal_diff_ratio"]["mean"]
    flags = []
    if dz > 0.30:
        flags.append(f"Δ/z={dz:.2f} 超出健康範圍(M 過度修飾 latent)")
    if tdr > 0.30:
        flags.append(f"tdr={tdr:.2f} 超出健康範圍(時間軌跡被改寫)")
    if not flags:
        return "✅ 兩條主指標都落在健康範圍 — M 學到健康的 mapping"
    return "❌ " + "; ".join(flags)


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join([head, sep, *body])


def build_report(log_path: Path, output_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    steps = parse_steps(text)
    monitors = parse_monitor(text)
    warnings = parse_warnings(text)
    cfg = parse_config_hints(text)

    if not steps:
        return f"# {log_path.name}\n\n⚠️  log 內沒抽到任何 `[step N rate]` 訓練紀錄。\n"

    last = steps[-1]
    steady = tail_steady_state(steps, frac=0.10)
    milestones = pick_milestones(steps)

    lines = []
    lines.append(f"# Stage 2 訓練 log summary")
    lines.append("")
    lines.append(f"- **Log**: `{log_path}`")
    lines.append(f"- **總 step 數**: {last['step']:,}")
    lines.append(f"- **log 內紀錄筆數**: {len(steps):,}(每 50 步一筆,間隔由 `log_interval` 決定)")
    lines.append(f"- **平均訓練速度**: {mean(r['it_per_s'] for r in steps[-50:]):.2f} it/s(末尾 50 筆均值)")
    if cfg:
        lines.append("")
        lines.append("**Config 印出**:")
        for k, v in cfg.items():
            lines.append(f"- `{k}` = {v}")
    lines.append("")

    # ── 整體判定 ──
    lines.append("## 整體判定")
    lines.append("")
    lines.append(overall_verdict(steady))
    lines.append("")

    # ── 末段穩態 ──
    lines.append("## 末段 10% 穩態(整體訓練成果)")
    lines.append("")
    keys_order = [
        ("m_total",            "總 G loss"),
        ("d_z",                "D_z loss"),
        ("l_nce",              "PatchNCE"),
        ("l_adv_z",            "L_adv_z"),
        ("l_adv_mel",          "L_adv_mel"),
        ("delta_over_z",       "‖Δ‖/‖z‖"),
        ("temporal_diff_ratio","tdr"),
    ]
    table_rows = []
    for k, label in keys_order:
        s = steady[k]
        v = verdict(s["mean"], k) if k in HEALTH else ""
        table_rows.append([
            label,
            f"{s['mean']:.4f}",
            f"{s['min']:.4f}",
            f"{s['max']:.4f}",
            v,
        ])
    lines.append(fmt_table(
        ["指標", "mean", "min", "max", "健康判定"],
        table_rows,
    ))
    lines.append("")

    # ── milestone trajectory ──
    lines.append("## Milestone trajectory")
    lines.append("")
    ms_rows = [[
        f"{r['step']:>7,}",
        f"{r['m_total']:+.3f}",
        f"{r['l_nce']:.3f}",
        f"{r['l_adv_z']:+.3f}",
        f"{r['l_adv_mel']:+.3f}",
        f"{r['delta_over_z']:+.3f}",
        f"{r['temporal_diff_ratio']:+.3f}",
    ] for r in milestones]
    lines.append(fmt_table(
        ["step", "m_total", "l_nce", "l_adv_z", "l_adv_mel", "Δ/z", "tdr"],
        ms_rows,
    ))
    lines.append("")

    # ── first-crossing 時間軸 ──
    lines.append("## 關鍵 first-crossing 事件")
    lines.append("")
    events = [
        ("Δ/z 首次破 0.30(健康上限)",  "delta_over_z",         0.30,  "above"),
        ("Δ/z 首次破 1.00(嚴重失控)",  "delta_over_z",         1.00,  "above"),
        ("tdr 首次破 0.30(時間結構警訊)", "temporal_diff_ratio",  0.30,  "above"),
        ("tdr 首次破 1.00(嚴重重寫)",   "temporal_diff_ratio",  1.00,  "above"),
        ("l_adv_mel 首次 < 0(M fool D_mel)", "l_adv_mel",       0.0,   "below"),
        ("l_adv_z 首次 > 0(D_z warmup 結束)", "l_adv_z",         0.001, "above"),
    ]
    ev_rows = []
    for label, metric, thr, direction in events:
        step = first_crossing(steps, metric, thr, direction)
        ev_rows.append([label, f"step {step:,}" if step is not None else "(從未發生)"])
    lines.append(fmt_table(["事件", "step"], ev_rows))
    lines.append("")

    # ── monitor 音質追蹤 ──
    if monitors:
        lines.append("## 音質監控(每 audio_quality_monitor_interval 步抽樣)")
        lines.append("")
        last_mon = monitors[-1]
        lines.append(f"**末次採樣 (step {last_mon['step']:,})**")
        lines.append(
            f"- unvoiced_concentration = {last_mon['unvoiced_concentration']:.3f}  "
            f"{verdict(last_mon['unvoiced_concentration'], 'unvoiced_concentration')}"
        )
        lines.append(
            f"- voiced_spectral_ratio  = {last_mon['voiced_spectral_ratio']:.3f}  "
            f"{verdict(last_mon['voiced_spectral_ratio'], 'voiced_spectral_ratio')}"
        )
        lines.append("")
        lines.append("**全程取樣**(每 ~5K 步一筆):")
        lines.append("")
        mon_rows = [[
            f"{r['step']:>7,}",
            f"{r['unvoiced_concentration']:.3f}",
            f"{r['voiced_spectral_ratio']:.3f}",
            f"{r['delta_voiced_e']:.3e}",
            f"{r['delta_unvoiced_e']:.3e}",
        ] for r in monitors]
        lines.append(fmt_table(
            ["step", "unvoiced_conc", "voiced_spec_ratio", "Δ_voiced_E", "Δ_unvoiced_E"],
            mon_rows,
        ))
        lines.append("")

    # ── warnings ──
    if warnings:
        lines.append("## 訓中 Failure mode 警告")
        lines.append("")
        for w in warnings:
            lines.append(f"- ⚠ `{w}`")
        lines.append("")
    else:
        lines.append("## 訓中 Failure mode 警告")
        lines.append("")
        lines.append("(無 — Stage2Trainer.fit 沒觸發 Failure A/B 警告)")
        lines.append("")

    # ── 早期(warmup 內)狀況 ──
    early = [r for r in steps if r["step"] <= 5000]
    if early:
        max_dz = max(early, key=lambda r: r["delta_over_z"])
        min_mel = min(early, key=lambda r: r["l_adv_mel"])
        lines.append("## Warmup 期(step ≤ 5000)狀況")
        lines.append("")
        lines.append("> warmup 期 L_adv_z=0,M 只受 PatchNCE + L_adv_mel + L_id_pro 約束。")
        lines.append("> 觀察 M 在 D_z 缺席時的「自由失控」程度可判斷 PatchNCE 是否足以拉住 M。")
        lines.append("")
        lines.append(f"- 最大 Δ/z = **{max_dz['delta_over_z']:.3f}** @ step {max_dz['step']:,}")
        lines.append(f"- 最低 l_adv_mel = **{min_mel['l_adv_mel']:.3f}** @ step {min_mel['step']:,}")
        lines.append("")
        if max_dz["delta_over_z"] > 1.0:
            lines.append("> ⚠ warmup 期 Δ/z 飆破 1.0 → PatchNCE alone 不足以制住 M;")
            lines.append("> 整個訓練的「穩態 Δ/z」高機率是 D_z 啟動後把 M 鎖到的均衡點,非健康點。")
            lines.append("")

    return "\n".join(lines)


def main():
    # Windows cp950 console 不能印 emoji,強制 stdout 用 utf-8
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Stage 2 訓練 log → markdown summary")
    ap.add_argument("log_path", type=Path, help="path to stage2 training log")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output markdown path(預設:同目錄 .summary.md)")
    args = ap.parse_args()

    if not args.log_path.exists():
        raise SystemExit(f"log 不存在: {args.log_path}")

    out = args.output or args.log_path.with_suffix(".summary.md")
    report = build_report(args.log_path, out)
    out.write_text(report, encoding="utf-8")
    print(f"[OK] summary 寫入: {out}")
    print(f"     ({len(report.splitlines())} lines, {len(report)} chars)")


if __name__ == "__main__":
    main()