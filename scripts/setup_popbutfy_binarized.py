"""解壓 PopBuTFy 兩 side 之 tar.zst + 生成 train/test split。

【流程】
1. 解 binarized_popbutfy_amateur.tar.zst → data/binarized/popbutfy_amateur/
2. 解 binarized_popbutfy_pro.tar.zst     → data/binarized/popbutfy_pro/
3. 依 NSVB 原版之 test_prefixes(Male6#singing#, Female14#singing#)生 split
   → data/binarized/splits_popbutfy/{train,test}.txt

【設計取捨】
- 用 Python zstandard + tarfile streaming,不依賴本機 zstd CLI(未裝)
- tar 內結構是 `local_binarized/popbutfy_*/*.npz`,自動 strip `local_binarized/` 前綴
- centroids 之 `ppg_kmeans_centroids.npy` 重複包在兩 tar 內,內容相同(都源自 v2),
  解壓時跳過(目的 dir 已有 v2 原版)
- split 仿 NSVB 原版邏輯(`NSVB/data_gen/singing/binarize.py:split_train_test_set`):
  test_prefixes 之 substring 於 item_id 中即進 test 集,其餘進 train
"""
import sys
import time
from pathlib import Path

import tarfile
import zstandard


ROOT = Path(r'C:\Users\neo29\workspace\SVC\NSVB-ZH\data\binarized')
TARS = {
    'amateur': ROOT / 'binarized_popbutfy_amateur.tar.zst',
    'pro':     ROOT / 'binarized_popbutfy_pro.tar.zst',
}
# NSVB 原版 egs/datasets/audio/PopBuTFy/base.yaml 之 test_prefixes
TEST_PREFIXES = ['Male6#singing#', 'Female14#singing#']
SPLIT_DIR = ROOT / 'splits_popbutfy'


def extract_one(side: str):
    tar_path = TARS[side]
    target_subdir = f'popbutfy_{side}'
    print(f'\n[extract {side}] {tar_path.name}')
    print(f'  目標: {ROOT / target_subdir}/')
    if not tar_path.exists():
        print(f'  [SKIP] tar 不存在')
        return 0

    target_dir = ROOT / target_subdir
    if target_dir.exists() and any(target_dir.glob('*.npz')):
        existing = len(list(target_dir.glob('*.npz')))
        print(f'  [SKIP] 目標已存在 {existing} 個 .npz')
        return existing

    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()
    with open(tar_path, 'rb') as f:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                for m in tar:
                    # 預期 tar 內結構: local_binarized/popbutfy_{side}/foo.npz
                    #               + local_binarized/ppg_kmeans_centroids.npy
                    if not m.isfile():
                        continue
                    parts = Path(m.name).parts
                    # strip 第一層 'local_binarized'
                    if parts and parts[0] == 'local_binarized':
                        rel = Path(*parts[1:])
                    else:
                        rel = Path(m.name)
                    # 跳過 centroids(tar 內重複包,目的 dir 已有 v2 原版)
                    if rel.name == 'ppg_kmeans_centroids.npy':
                        continue
                    # 只接受對應 side 之 .npz
                    if not rel.parts or rel.parts[0] != target_subdir:
                        print(f'  [WARN] unexpected member: {m.name}')
                        continue
                    if rel.suffix != '.npz':
                        continue

                    m.name = str(rel)  # 重寫 member name 為相對 ROOT 之路徑
                    tar.extract(m, path=ROOT)
                    n += 1
                    if n % 1000 == 0:
                        elapsed = time.time() - t0
                        rate = n / elapsed
                        print(f'  ... {n} 檔  ({rate:.0f} files/s)')
    elapsed = time.time() - t0
    print(f'  [DONE] {n} 個 .npz 解出,{elapsed/60:.1f} min')
    return n


def gen_splits(n_amateur_expect: int, n_pro_expect: int):
    print(f'\n[split] 掃 .npz + 依 NSVB test_prefixes 切 train/test')
    print(f'  test_prefixes = {TEST_PREFIXES}')

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = []
    test_ids = []

    for side in ('amateur', 'pro'):
        dir_ = ROOT / f'popbutfy_{side}'
        if not dir_.exists():
            print(f'  [WARN] {dir_} 不存在,跳過')
            continue
        items = sorted(p.stem for p in dir_.glob('*.npz'))
        n_train = n_test = 0
        for item_id in items:
            if any(prefix in item_id for prefix in TEST_PREFIXES):
                test_ids.append(item_id)
                n_test += 1
            else:
                train_ids.append(item_id)
                n_train += 1
        print(f'  {side}: total={len(items)}  train={n_train}  test={n_test}')

    train_path = SPLIT_DIR / 'train.txt'
    test_path  = SPLIT_DIR / 'test.txt'
    train_path.write_text('\n'.join(train_ids) + '\n', encoding='utf-8')
    test_path.write_text('\n'.join(test_ids) + '\n', encoding='utf-8')

    print(f'\n  寫入:')
    print(f'    {train_path}  ({len(train_ids)} item_ids)')
    print(f'    {test_path}   ({len(test_ids)} item_ids)')
    return train_ids, test_ids


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print('=' * 60)
    print('PopBuTFy 解壓 + split 生成')
    print('=' * 60)

    n_am = extract_one('amateur')
    n_pr = extract_one('pro')

    print(f'\n[summary] amateur={n_am}, pro={n_pr}')
    if n_am != 14746:
        print(f'  [WARN] amateur 預期 14746,實際 {n_am}')
    if n_pr != 14219:
        print(f'  [WARN] pro 預期 14219,實際 {n_pr}')

    train_ids, test_ids = gen_splits(n_am, n_pr)

    # 抽樣顯示 test set 內容(確認 prefix 配對正確)
    print('\n[sanity] test 集前 5 個 item_id:')
    for x in test_ids[:5]:
        print(f'    {x}')
    print('[sanity] train 集前 3 個 item_id:')
    for x in train_ids[:3]:
        print(f'    {x}')


if __name__ == '__main__':
    main()