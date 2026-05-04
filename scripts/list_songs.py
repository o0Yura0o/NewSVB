import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

label_dir = "c:/Users/neo29/workspace/SVC/NSVB-ZH/data/VocalVerse/VocalVerse_Datasets-human_labels"
ama = pd.read_excel(f"{label_dir}/Amateur_overall_mos_avg5.xlsx")
ama['MOS'] = ama['总分'] / 5.0

songs = ama.groupby(['歌曲id', '歌曲名称']).agg(
    錄音數=('录音id', 'count'),
    MOS均值=('MOS', 'mean'),
    MOS最小=('MOS', 'min'),
    MOS最大=('MOS', 'max'),
).reset_index()

print(f"VocalVerse 33 首歌曲清單：\n")
print(f"{'No':>3}  {'歌曲id':<10} {'歌曲名称':<20} {'錄音數':>6} {'MOS均值':>8} {'MOS最小':>8} {'MOS最大':>8}")
print("-" * 70)
for i, row in songs.iterrows():
    print(f"{i+1:>3}  {row['歌曲id']:<10} {row['歌曲名称']:<20} {row['錄音數']:>6} {row['MOS均值']:>8.2f} {row['MOS最小']:>8.1f} {row['MOS最大']:>8.1f}")

# Output just song names for M4Singer cross-reference
print("\n--- Song names only ---")
for name in songs['歌曲名称'].tolist():
    print(name)
