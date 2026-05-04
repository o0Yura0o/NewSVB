"""
scripts/download_vocalverse.py
================================

下載 VocalVerse 業餘歌聲 dataset 到本地（從 HuggingFace）。

用法：
    python scripts/download_vocalverse.py [--out-dir data/VocalVerse]

【為什麼用 snapshot_download 而非 git clone】
- HF dataset 用 LFS，git clone 慢且要先裝 git-lfs；snapshot_download 直接走 HTTP
- 支援續傳：跑到一半中斷，重跑只下載未完成的檔案
- 用 ignore_patterns 過濾 .DS_Store 等垃圾檔
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default="data/VocalVerse",
        help="本機儲存路徑（相對於目前工作目錄；預設 data/VocalVerse）",
    )
    ap.add_argument(
        "--repo-id", default="karl-wang/VocalVerse-dataset",
        help="HuggingFace dataset repo id",
    )
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} → {out}", flush=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out),
        ignore_patterns=[".DS_Store"],
    )
    print(f"\nDone. Downloaded to: {path}")


if __name__ == "__main__":
    sys.exit(main())
