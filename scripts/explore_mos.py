import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

label_dir = "c:/Users/neo29/workspace/SVC/NSVB-ZH/data/VocalVerse/VocalVerse_Datasets-human_labels"

# ── Amateur MOS ──────────────────────────────────────────────
print("=" * 60)
print("Amateur MOS (Amateur_overall_mos_avg5.xlsx)")
print("=" * 60)
ama = pd.read_excel(f"{label_dir}/Amateur_overall_mos_avg5.xlsx")
print(f"Shape: {ama.shape}  (rows = recordings)")
print(f"Columns: {list(ama.columns)}\n")

# 总分 = sum of 5 reviewers (scale 1-5 each) => MOS = 总分 / 5
ama['MOS'] = ama['总分'] / 5.0

mos = ama['MOS']
print(f"MOS stats (总分 / 5):")
print(f"  Count  : {len(mos)}")
print(f"  Mean   : {mos.mean():.3f}")
print(f"  Std    : {mos.std():.3f}")
print(f"  Min    : {mos.min():.3f}")
print(f"  Max    : {mos.max():.3f}")
print(f"  Median : {mos.median():.3f}")
print(f"  25th % : {mos.quantile(0.25):.3f}")
print(f"  75th % : {mos.quantile(0.75):.3f}")

print("\nDistribution by 0.5 bins:")
edges = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]
for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i+1]
    label = f"[{lo:.1f},{min(hi,5.0):.1f}]" if i == len(edges)-2 else f"[{lo:.1f},{hi:.1f})"
    count = ((mos >= lo) & (mos < hi)).sum()
    bar = "#" * (count // 5)
    print(f"  {label} : {count:4d}  {bar}")

print("\n業餘樣本篩選（MOS <= threshold）：")
for thresh in [2.0, 2.5, 3.0, 3.5, 4.0]:
    n = (mos <= thresh).sum()
    pct = n / len(mos) * 100
    print(f"  MOS <= {thresh}: {n:4d} 筆 ({pct:.1f}%)")

# Unique singers
print(f"\n歌手數（歌曲id）: {ama['歌曲id'].nunique()}")
print(f"歌曲數（歌曲名称）: {ama['歌曲名称'].nunique()}")
print(f"\nMOS 分布 by 歌手（per-singer stats）:")
singer_stats = ama.groupby('歌曲id')['MOS'].agg(['count','mean','min','max'])
singer_stats.columns = ['錄音數', 'mean MOS', 'min MOS', 'max MOS']
print(singer_stats.to_string())

# ── Professional annotations ──────────────────────────────────
print("\n" + "=" * 60)
print("Professional annotations (multi-dim)")
print("=" * 60)
pro = pd.read_excel(
    f"{label_dir}/Professional_multidim_annotations_raw_Timbre_Breath_Emotion_Technique.xlsx"
)
print(f"Shape: {pro.shape}")
print(f"Columns: {list(pro.columns)}\n")

score_cols = [c for c in pro.columns if '打分' in c]
print(f"Score columns: {score_cols}")
for col in score_cols:
    s = pd.to_numeric(pro[col], errors='coerce').dropna()
    print(f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, min={s.min()}, max={s.max()}, n={len(s)}")
