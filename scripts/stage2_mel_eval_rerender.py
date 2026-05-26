"""從現有 metrics_aggregate.csv 重新生報告(不重跑 inference)。

【為什麼存在】
舊版 stage2_mel_eval.py 的 report.md 把 M4 + VV 混合平均算 verdict,失去意義。新版
script 已改成 VV-only 主表 + M4 control 分表 + best step 推薦。

但 CSV 內已有完整 per-sample × per-step × 全 metric 資料,**不需要重跑 ~3h inference**,
直接從 CSV reconstruct per_sample dict,呼叫新版 aggregate_and_report 即可。

【用法】
    python scripts/stage2_mel_eval_rerender.py \\
        --csv outputs/stage2_v2_eval_fulltest/metrics_aggregate.csv \\
        --out-dir outputs/stage2_v2_eval_fulltest

寫入該目錄 `report.md`(會覆寫舊版)。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_mel_eval import aggregate_and_report


def main():
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="從 metrics_aggregate.csv 重生新版 report.md")
    ap.add_argument("--csv", type=Path, required=True,
                    help="metrics_aggregate.csv 路徑(舊版 stage2_mel_eval 產出)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="新報告寫到這個目錄(同 CSV 所在目錄就會覆寫舊 report.md)")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV 不存在: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 從 CSV 重建 per_sample dict ──
    # 結構: per_sample[s_key][label] = {metric_name: float}
    #
    # ⚠ 舊版 CSV 沒做 escape,M4 部份歌名含逗號(如「想你,零点零一分」)會讓 row
    # 多出一欄。用 csv.reader 自帶 quoting 解析,可正確處理新版 escape;舊版壞
    # row 用「**從右側 parse**」邏輯救:固定欄位數 = 2(sample, label) + N(metrics),
    # 從右端取 N 個是 metrics,再倒一個是 label,前面剩下的全部 join 回 sample。
    per_sample: dict = {}
    n_rows = 0
    metric_cols: list = []
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        metric_cols = header[2:]
        n_metrics = len(metric_cols)
        print(f"[rerender] CSV columns: {n_metrics} metrics, header={header[:3]}...")
        for row in reader:
            if len(row) < 2 + n_metrics:
                print(f"[rerender] WARN: skip malformed row (cols={len(row)}): {row[:3]}")
                continue
            # 從右側 parse:最後 n_metrics 欄是 metric values
            metric_vals = row[-n_metrics:]
            label = row[-(n_metrics + 1)]
            # sample = 剩下的全部(可能因逗號被切成多段)join 回去
            sample_parts = row[: -(n_metrics + 1)]
            s_key = ",".join(sample_parts)
            try:
                metrics_dict = {c: float(v) for c, v in zip(metric_cols, metric_vals)}
            except ValueError as e:
                print(f"[rerender] WARN: skip row, float parse failed ({e}): {s_key=}, {label=}")
                continue
            per_sample.setdefault(s_key, {})[label] = metrics_dict
            n_rows += 1

    n_samples = len(per_sample)
    print(f"[rerender] loaded {n_rows} rows = {n_samples} samples × ~{n_rows//max(n_samples,1)} labels each")

    # _0_orig 在原 script 用空 dict({}),CSV 沒這 row → 補上避免 aggregate 邏輯抱怨
    for s_key, labels_d in per_sample.items():
        if "_0_orig" not in labels_d:
            labels_d["_0_orig"] = {}

    # ── 呼叫新版 aggregate_and_report ──
    aggregate_and_report(per_sample, args.out_dir)
    print(f"[OK] {args.out_dir / 'report.md'} 已重生")


if __name__ == "__main__":
    main()