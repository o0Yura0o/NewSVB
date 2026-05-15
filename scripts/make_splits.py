"""
scripts/make_splits.py
=========================

【這支腳本做什麼】
從 binarized .npz 切出 train / val / test 三個 item_id 列表，寫成 .txt 給
`nsvb/data/dataset.py` 過濾用。**不動 .npz 內容**，只產生列表檔。

【切割原則(SVC/SVB 慣例)】
1. 以「歌手」為單位 hold out test:同一位歌手不能同時出現在 train 跟 test，
   否則模型可能記住 spk_emb 而不是學泛化能力。
2. val 從 training 歌手裡再 hold out 「整首歌」(M4)或「整個來源錄音」(VV)，
   避免 random crop 讓 val/train 看到同一段 phrase。
3. 種子固定 → 重跑同種子產生**完全一樣**的切割,確保可重現。

【M4Singer 切割】
- item_id 結構:`{speaker}#{song}__{idx}`,例 `Alto-1#newboy__0000`
- test = 整個指定歌手(預設 2 位)的所有 phrase
- val = training 歌手裡每人 hold out N 首歌(預設 2 首)的所有 phrase
- train = 剩下

【VocalVerse 切割】
- item_id 結構(chunked):`{user_id}__{wav_id}__c{NNN}`
- test = 隨機選 X% 的 user_id(預設 10%)整人 hold out
- val = training user 中,隨機 hold out Y% 的「來源錄音」(整個 wav_id 的所有 chunk)
- train = 剩下

【為什麼 val 用「整首歌 / 整個來源錄音」而非單一 chunk】
- dataset.py 訓練時 random crop 同一首歌的不同 phrase → 若 train 與 val 共享同
  首歌,val 等於看 train 過的內容,評估失真
- 把整首歌(或整個來源錄音)整批分到 val 才是乾淨的「unseen」評估

【用法】
  python scripts/make_splits.py \
      --binarized-root /content/local_binarized
  # 預設輸出到 {binarized-root}/splits/{train,val,test}.txt + report.json

  # 鎖定特定歌手(推薦,讓 holdout 在 git 內可見):
  python scripts/make_splits.py \
      --binarized-root /content/local_binarized \
      --m4-test-singers Alto-2 Tenor-3

  # 自定比例:
  python scripts/make_splits.py \
      --binarized-root /content/local_binarized \
      --vv-test-singer-frac 0.15 --vv-val-utterance-frac 0.05

【可重現性】
seed 固定(預設 42)→ 同 seed + 同 dataset 永遠產生相同切割。
要驗證:看 `{out-dir}/report.json`,內含 seed + 實際被 hold out 的歌手 list。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── item_id parsers(必須跟 binarizer.py 一致)─────────
def parse_m4_item(item_id: str) -> tuple:
    """
    M4 item_id 結構:`{speaker}#{song}__{idx}`
    例:`Alto-1#newboy__0000` → ('Alto-1', 'newboy', '0000')

    用 rsplit 而非 split:歌名中**可能**含 '__'(雖然 M4 實測未見)
    → 用 rsplit('__', 1) 保證只把「最後一個 __」當分隔符。
    """
    speaker, rest = item_id.split('#', 1)
    parts = rest.rsplit('__', 1)
    if len(parts) == 2:
        song, idx = parts
    else:
        song, idx = parts[0], ''
    return speaker, song, idx


def parse_vv_item(item_id: str) -> tuple:
    """
    VV item_id 結構(chunked):`{user_id}__{wav_id}__c{NNN}`
    例:`443212__340406604__c003` → ('443212', '340406604', '003')

    user_id 一定是首段(VocalVerse 目錄結構保證);
    chunk 後綴一定是 `__c{NNN}` 結尾(binarizer.chunk_sample 寫法)。
    """
    if '__c' in item_id:
        rec_id, chunk_idx = item_id.rsplit('__c', 1)
    else:
        rec_id, chunk_idx = item_id, ''
    user_id, wav_id = rec_id.split('__', 1)
    return user_id, wav_id, chunk_idx


# ── 切割邏輯 ────────────────────────────────────────────
def make_m4_splits(
    m4_dir: Path,
    test_singers: list,
    val_songs_per_singer: int,
    rng: np.random.Generator,
):
    """Returns (train_items, val_items, test_items, train_singers, test_singers_used)."""
    items_by_singer = defaultdict(list)  # singer → [(item_id, song), ...]
    for p in sorted(m4_dir.glob('*.npz')):
        item_id = p.stem
        speaker, song, _ = parse_m4_item(item_id)
        items_by_singer[speaker].append((item_id, song))

    all_singers = sorted(items_by_singer.keys())
    missing = [s for s in test_singers if s not in items_by_singer]
    if missing:
        raise ValueError(
            f"M4 test singers not in data: {missing}\n"
            f"  available singers: {all_singers}"
        )

    test_items, val_items, train_items = [], [], []

    for singer in all_singers:
        items = items_by_singer[singer]
        if singer in test_singers:
            test_items.extend(i[0] for i in items)
            continue
        # training singer:從這位歌手的 song 中 hold out N 首給 val
        songs = sorted(set(s for _, s in items))
        rng.shuffle(songs)
        val_songs = set(songs[:val_songs_per_singer])
        for item_id, song in items:
            if song in val_songs:
                val_items.append(item_id)
            else:
                train_items.append(item_id)

    train_singers = [s for s in all_singers if s not in test_singers]
    return train_items, val_items, test_items, train_singers, list(test_singers)


def make_vv_splits(
    vv_dir: Path,
    test_singer_frac: float,
    val_utterance_frac: float,
    rng: np.random.Generator,
):
    """Returns (train_items, val_items, test_items, train_users, test_users)."""
    items_by_user = defaultdict(list)  # user → [(item_id, wav_id), ...]
    for p in sorted(vv_dir.glob('*.npz')):
        item_id = p.stem
        user_id, wav_id, _ = parse_vv_item(item_id)
        items_by_user[user_id].append((item_id, wav_id))

    all_users = sorted(items_by_user.keys())
    n_test = max(1, int(round(len(all_users) * test_singer_frac)))

    # 隨機選 test users(seeded → 可重現)
    shuffled = list(all_users)
    rng.shuffle(shuffled)
    test_users = sorted(shuffled[:n_test])

    test_items, val_items, train_items = [], [], []

    for user in all_users:
        items = items_by_user[user]
        if user in test_users:
            test_items.extend(i[0] for i in items)
            continue
        # training user:hold out val_utterance_frac 的 source 錄音(整 wav_id 的所有 chunk)
        wavs = sorted(set(w for _, w in items))
        rng.shuffle(wavs)
        n_val = max(0, int(round(len(wavs) * val_utterance_frac)))
        val_wavs = set(wavs[:n_val])
        for item_id, wav_id in items:
            if wav_id in val_wavs:
                val_items.append(item_id)
            else:
                train_items.append(item_id)

    train_users = [u for u in all_users if u not in test_users]
    return train_items, val_items, test_items, train_users, test_users


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for NSVB-ZH binarized data",
    )
    parser.add_argument("--binarized-root", default="data/binarized",
                        help="包含 m4singer/ 與 vocalverse/ 的根目錄")
    parser.add_argument("--out-dir", default=None,
                        help="輸出 train.txt/val.txt/test.txt 的目錄。"
                             "預設 {binarized-root}/splits/")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子(同 seed → 完全相同切割,可重現)")
    parser.add_argument("--m4-test-singers", nargs='*', default=None,
                        help="M4 hold-out 歌手 list,空白分隔。例:Alto-2 Tenor-3。"
                             "未指定則用 seed 自動挑 2 位(推薦明確指定,讓 holdout 在 git 內可見)")
    parser.add_argument("--m4-val-songs-per-singer", type=int, default=2,
                        help="M4 每位 training 歌手 hold out 幾首歌給 val(預設 2)")
    parser.add_argument("--vv-test-singer-frac", type=float, default=0.10,
                        help="VV hold out 幾成 user_id 給 test(預設 0.10)")
    parser.add_argument("--vv-val-utterance-frac", type=float, default=0.05,
                        help="VV training user 中,hold out 幾成「來源錄音」給 val(預設 0.05)")
    args = parser.parse_args()

    binarized_root = Path(args.binarized_root)
    out_dir = Path(args.out_dir) if args.out_dir else binarized_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    m4_dir = binarized_root / "m4singer"
    vv_dir = binarized_root / "vocalverse"
    if not m4_dir.is_dir():
        raise FileNotFoundError(f"missing {m4_dir}")
    if not vv_dir.is_dir():
        raise FileNotFoundError(f"missing {vv_dir}")

    # 決定 M4 test 歌手
    if args.m4_test_singers:
        m4_test_singers = sorted(args.m4_test_singers)
        print(f"[m4] test singers (explicit): {m4_test_singers}")
    else:
        m4_singers_all = sorted({parse_m4_item(p.stem)[0] for p in m4_dir.glob('*.npz')})
        idx = rng.choice(len(m4_singers_all), size=2, replace=False)
        m4_test_singers = sorted([m4_singers_all[i] for i in idx])
        print(f"[m4] test singers (auto, seed={args.seed}): {m4_test_singers}")

    print("\n=== M4Singer split ===")
    m4_train, m4_val, m4_test, m4_train_singers, m4_test_singers_used = make_m4_splits(
        m4_dir, m4_test_singers, args.m4_val_songs_per_singer, rng,
    )
    total_m4 = len(m4_train) + len(m4_val) + len(m4_test)
    print(f"  train: {len(m4_train):>6}  ({100*len(m4_train)/total_m4:.1f}%)")
    print(f"  val:   {len(m4_val):>6}  ({100*len(m4_val)/total_m4:.1f}%)")
    print(f"  test:  {len(m4_test):>6}  ({100*len(m4_test)/total_m4:.1f}%) from {m4_test_singers_used}")

    print("\n=== VocalVerse split ===")
    vv_train, vv_val, vv_test, vv_train_users, vv_test_users = make_vv_splits(
        vv_dir, args.vv_test_singer_frac, args.vv_val_utterance_frac, rng,
    )
    total_vv = len(vv_train) + len(vv_val) + len(vv_test)
    print(f"  train: {len(vv_train):>6}  ({100*len(vv_train)/total_vv:.1f}%)")
    print(f"  val:   {len(vv_val):>6}  ({100*len(vv_val)/total_vv:.1f}%)")
    print(f"  test:  {len(vv_test):>6}  ({100*len(vv_test)/total_vv:.1f}%) from {len(vv_test_users)} users")

    # 寫 split 檔(混合 M4 + VV,dataset.py 自然會依目錄分流)
    all_train = m4_train + vv_train
    all_val = m4_val + vv_val
    all_test = m4_test + vv_test
    total_all = len(all_train) + len(all_val) + len(all_test)

    print(f"\n=== Combined ===")
    print(f"  train: {len(all_train):>6}  ({100*len(all_train)/total_all:.1f}%)")
    print(f"  val:   {len(all_val):>6}  ({100*len(all_val)/total_all:.1f}%)")
    print(f"  test:  {len(all_test):>6}  ({100*len(all_test)/total_all:.1f}%)")

    for name, items in [('train', all_train), ('val', all_val), ('test', all_test)]:
        path = out_dir / f"{name}.txt"
        path.write_text('\n'.join(sorted(items)) + '\n', encoding='utf-8')
        print(f"  wrote {path}  ({len(items)} items)")

    # report 給 audit / 重現用
    report = {
        'seed': args.seed,
        'binarized_root': str(binarized_root),
        'counts': {
            'm4singer': {'train': len(m4_train), 'val': len(m4_val), 'test': len(m4_test)},
            'vocalverse': {'train': len(vv_train), 'val': len(vv_val), 'test': len(vv_test)},
            'total': {'train': len(all_train), 'val': len(all_val), 'test': len(all_test)},
        },
        'm4_test_singers': m4_test_singers_used,
        'm4_train_singers_count': len(m4_train_singers),
        'm4_val_songs_per_singer': args.m4_val_songs_per_singer,
        'vv_test_users_count': len(vv_test_users),
        'vv_train_users_count': len(vv_train_users),
        'vv_test_singer_frac': args.vv_test_singer_frac,
        'vv_val_utterance_frac': args.vv_val_utterance_frac,
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  wrote {report_path}")

    print(f"\n✅ Splits 寫到 {out_dir}/  (重跑同 seed 會產生相同切割)")


if __name__ == "__main__":
    main()