import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import wave
import struct

data_dir = "c:/Users/neo29/workspace/SVC/NSVB-ZH/data/VocalVerse"

total_duration = 0.0
total_files = 0
errors = 0
folder_stats = {}

for singer_id in sorted(os.listdir(data_dir)):
    singer_dir = os.path.join(data_dir, singer_id)
    if not os.path.isdir(singer_dir) or singer_id.startswith('.') or singer_id == 'VocalVerse_Datasets-human_labels':
        continue
    folder_dur = 0.0
    folder_count = 0
    for fname in os.listdir(singer_dir):
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(singer_dir, fname)
        try:
            with wave.open(fpath, 'r') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                dur = frames / float(rate)
                folder_dur += dur
                folder_count += 1
                total_duration += dur
                total_files += 1
        except Exception as e:
            errors += 1
    folder_stats[singer_id] = (folder_count, folder_dur)

print(f"Downloaded so far: {total_files} WAV files, {errors} errors")
print(f"Total duration: {total_duration:.1f}s = {total_duration/60:.1f}min = {total_duration/3600:.2f}h")
if total_files > 0:
    print(f"Average per file: {total_duration/total_files:.1f}s")
print()
print(f"{'Singer':<12} {'Files':>6} {'Duration(min)':>14}")
print("-" * 36)
for sid, (cnt, dur) in sorted(folder_stats.items()):
    print(f"{sid:<12} {cnt:>6} {dur/60:>14.1f}")
