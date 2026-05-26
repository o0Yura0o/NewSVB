"""
scripts/bootstrap_downloads.py
================================

【這支腳本做什麼】
NSVB-ZH 開新機後的「一鍵下載」:把 Phase 0 跑得起來所需的全部外部資源拉到位:
    1. NSVB pretrained vocoder ckpt (`1012_hifigan_all_songs_nsf`)         — GitHub release
    2. NSVB pretrained Stage 1 init ckpt (`1030_vae_mle`)                  — GitHub release
    3. M4Singer dataset                                                    — Google Drive
    4. VocalVerse dataset (a cappella wav)                                 — HuggingFace
    5. VocalVerse 標籤 xlsx (amateur MOS + pro 4-dim)                       — HuggingFace

【設計選擇】

- **每一步可獨立跳過**:`--skip vocoder vae_mle m4 vv vv_labels` 任意組合。已存在
  的目標檔自動 skip(用 marker file / 目錄存在判斷),不會無謂重下。
- **進度顯示**:大檔下載用 `tqdm` 包 urllib stream,連線中斷後重跑會從頭開始
  下載 partial 檔(GitHub release 不支援 Range header 續傳,M4Singer Google Drive
  經 gdown 自身續傳機制)。
- **解壓自動處理**:GitHub release 是 `.zip`,下完用 `zipfile` 解到 `checkpoints/`
  正確子目錄,刪掉中間 .zip。
- **可選依賴**:gdown(M4Singer)只在 `--skip m4` 沒給時才 import,
  避免裝環境的人若不需要 M4 就不用裝 gdown。

【為什麼 GitHub release 而非 git clone NSVB repo】
- NSVB repo 本體 ~150 MB,ckpt 是大 LFS 檔(vocoder ~50 MB + VAE ~30 MB);
  git clone 整 repo 比直接抓 release zip 慢且占空間。
- release URL 是穩定 endpoint,不需要設 GH_TOKEN(public release)。

【為什麼 M4Singer 用 gdown 而非 HuggingFace mirror】
- HuggingFace 上 `umoubuton/m4singer` 不完整(label class 都是同一個,
  原始目錄結構與 wav-per-singer 對應不上)
- 官方分發是 Google Drive(M4Singer README 註明),只能 gdown
- M4Singer 整包 ~9 GB,下載 ~30-60 min,要有耐心

【為什麼 VocalVerse 走 huggingface_hub.snapshot_download】
- 與既有 `scripts/download_vocalverse.py` 邏輯一致(走 LFS 對等,支援續傳)
- HuggingFace dataset endpoint 穩定且免 token

【用法】
    # 全下(預設)
    python scripts/bootstrap_downloads.py

    # 跳過已下載的部分(e.g. 重灌時只缺 M4)
    python scripts/bootstrap_downloads.py --skip vocoder vae_mle vv vv_labels

    # 自訂目錄
    python scripts/bootstrap_downloads.py \\
        --data-root D:\\NSVB-ZH-data \\
        --ckpt-root D:\\NSVB-ZH-ckpts

    # 只看要做哪些事(不真的下載)
    python scripts/bootstrap_downloads.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request
from urllib.error import URLError


# ── 來源 URL(public,皆可直接 GET)─────────────────────────
NSVB_RELEASE_BASE = (
    "https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release"
)
URL_VOCODER = f"{NSVB_RELEASE_BASE}/1012_hifigan_all_songs_nsf.zip"
URL_VAE_MLE = f"{NSVB_RELEASE_BASE}/1030_vae_mle.zip"
# M4Singer 官方 Google Drive(從 M4Singer README 取)
M4SINGER_GDRIVE_ID = "1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW"
# VocalVerse HuggingFace
VOCALVERSE_HF_REPO = "karl-wang/VocalVerse-dataset"
# VocalVerse 標籤檔在 HF repo 內的相對路徑
VOCALVERSE_LABEL_DIR = "VocalVerse_Datasets-human_labels"
VOCALVERSE_LABEL_FILES = [
    "Amateur_overall_mos_avg5.xlsx",
    "Professional_multidim_annotations_raw_Timbre_Breath_Emotion_Technique.xlsx",
    "Professional_scoring_rubric.xlsx",
]


# ── 下載工具 ─────────────────────────────────────────
def _human_readable(n_bytes: int) -> str:
    """6 種單位精簡顯示(下載速度 / 檔案大小用)。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def download_http(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    """
    下載 HTTP URL 到 dst,顯示進度。

    為什麼用 urllib.request 而非 requests:
        urllib 是 stdlib,Windows 預設 Python 即有;
        requests 雖然好用但需要 pip install,降低 bootstrap 腳本的「環境依賴」。
        tqdm 若沒裝就 fallback 到純 stderr 進度列。

    為什麼用 stream + chunk:
        ckpt 動輒 30-50 MB,whole-file read 會占 RAM;chunk 下載對所有大小都安全。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[download] {url}\n           → {dst}", flush=True)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    try:
        with request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            if use_tqdm:
                bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                           desc=dst.name, ncols=100, ascii=True)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if use_tqdm:
                        bar.update(len(buf))
                    else:
                        pct = (done / total * 100) if total else 0
                        sys.stderr.write(
                            f"\r  {_human_readable(done)}"
                            f"{' / ' + _human_readable(total) if total else ''}"
                            f" ({pct:.1f}%)"
                        )
                        sys.stderr.flush()
            if use_tqdm:
                bar.close()
            else:
                sys.stderr.write("\n")
        tmp.rename(dst)
    except URLError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"下載失敗: {url}\n  {e}") from e


def unzip_to(zip_path: Path, dst_dir: Path, remove_zip: bool = True) -> None:
    """解壓 zip 到 dst_dir(必要時建立),預設刪除中間 zip 檔以省空間。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip]    {zip_path.name} → {dst_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst_dir)
    if remove_zip:
        zip_path.unlink()
        print(f"[unzip]    cleaned up {zip_path.name}", flush=True)


# ── 每一步 ─────────────────────────────────────────
def fetch_nsvb_vocoder(ckpt_root: Path, dry_run: bool) -> None:
    """
    NSVB pretrained HifiGAN-NSF vocoder。下完解壓到
        {ckpt_root}/nsvb_pretrained/1012_hifigan_all_songs_nsf/
    內含 config.yaml + model_ckpt_steps_1170000.ckpt。
    """
    target_dir = ckpt_root / "nsvb_pretrained" / "1012_hifigan_all_songs_nsf"
    marker = target_dir / "config.yaml"
    if marker.exists():
        print(f"[skip]     vocoder ckpt 已存在 ({target_dir}/)\n", flush=True)
        return
    if dry_run:
        print(f"[dry-run]  would download vocoder → {target_dir}\n", flush=True)
        return

    zip_path = ckpt_root / "nsvb_pretrained" / "1012_hifigan_all_songs_nsf.zip"
    download_http(URL_VOCODER, zip_path)
    unzip_to(zip_path, ckpt_root / "nsvb_pretrained")
    print(f"[done]     vocoder → {target_dir}\n", flush=True)


def fetch_nsvb_vae_mle(ckpt_root: Path, dry_run: bool) -> None:
    """
    NSVB pretrained CVAE ckpt(1030_vae_mle),Stage 1 init 用。
    解壓到 {ckpt_root}/nsvb_pretrained/1030_vae_mle/。
    """
    target_dir = ckpt_root / "nsvb_pretrained" / "1030_vae_mle"
    # 這個 ckpt 解壓出來是 model_ckpt_steps_200000.ckpt + 可能的 config;
    # 我們檢查 .ckpt 是否存在當 marker
    marker_glob = list(target_dir.glob("model_ckpt_steps_*.ckpt")) if target_dir.exists() else []
    if marker_glob:
        print(f"[skip]     vae_mle ckpt 已存在 ({marker_glob[0].name})\n", flush=True)
        return
    if dry_run:
        print(f"[dry-run]  would download vae_mle → {target_dir}\n", flush=True)
        return

    zip_path = ckpt_root / "nsvb_pretrained" / "1030_vae_mle.zip"
    download_http(URL_VAE_MLE, zip_path)
    unzip_to(zip_path, ckpt_root / "nsvb_pretrained")
    print(f"[done]     vae_mle → {target_dir}\n", flush=True)


def fetch_m4singer(data_root: Path, dry_run: bool) -> None:
    """
    M4Singer 官方 Google Drive zip,gdown 下載 + 解壓到 {data_root}/m4singer/。

    為什麼 marker 用「目錄存在 + 至少一個 #-命名子目錄」:
        M4Singer 結構是 `{歌手}#{歌名}/{idx}.wav`,zip 解開即是這種目錄;
        檢查任一 `*#*` 目錄存在表示前次下載解壓成功。
    """
    target_dir = data_root / "m4singer"
    has_data = target_dir.exists() and any(target_dir.glob("*#*"))
    if has_data:
        print(f"[skip]     M4Singer 已存在 ({target_dir}/)\n", flush=True)
        return
    if dry_run:
        print(f"[dry-run]  would download M4Singer → {target_dir} (~9 GB)\n", flush=True)
        return

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "下載 M4Singer 需要 gdown:\n  pip install gdown\n"
            "或加 --skip m4 跳過 M4Singer"
        )

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "m4singer.zip"
    print(f"[download] M4Singer Google Drive (id={M4SINGER_GDRIVE_ID})\n"
          f"           → {zip_path}\n"
          f"           ⚠ ~9 GB,下載 ~30-60 min,請耐心等待\n", flush=True)
    # gdown 自己有進度條;quiet=False 顯示
    gdown.download(id=M4SINGER_GDRIVE_ID, output=str(zip_path), quiet=False)

    unzip_to(zip_path, data_root)
    # M4Singer zip 可能解壓成 m4singer/ 或 M4Singer/(看版本);統一為 m4singer/
    cand = data_root / "M4Singer"
    if cand.exists() and not target_dir.exists():
        cand.rename(target_dir)
    print(f"[done]     M4Singer → {target_dir}\n", flush=True)


def fetch_vocalverse(data_root: Path, dry_run: bool) -> None:
    """
    VocalVerse a cappella wav,走 huggingface_hub.snapshot_download。
    下載到 {data_root}/VocalVerse/。

    為什麼 marker 用「目錄存在 + 至少一個非 label 子目錄」:
        VocalVerse 結構是 `{user_id}/{wav_id}.wav`;檢查除了
        `VocalVerse_Datasets-human_labels` 之外還有別的子目錄。

    為什麼需要 hf_xet:
        VocalVerse repo 用 HuggingFace Xet Storage backend(content-addressed
        大檔優化)。沒裝 hf_xet 會 fallback 到單連線 HTTP,50 GB 從台灣拉
        cas-bridge.xethub.hf.co 會反覆 timeout。snapshot_download 內建 resume,
        中斷後重跑會接續,不會白下。
    """
    target_dir = data_root / "VocalVerse"

    def has_data() -> bool:
        if not target_dir.exists():
            return False
        for child in target_dir.iterdir():
            if child.is_dir() and child.name != VOCALVERSE_LABEL_DIR:
                # 檢查內含至少一個 wav
                if any(child.glob("*.wav")):
                    return True
        return False

    if has_data():
        print(f"[skip]     VocalVerse 已存在 ({target_dir}/)\n", flush=True)
        return
    if dry_run:
        print(f"[dry-run]  would download VocalVerse → {target_dir} (~50 GB)\n", flush=True)
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "下載 VocalVerse 需要 huggingface_hub(environment.yml 已含):\n"
            "  pip install huggingface_hub\n或加 --skip vv 跳過"
        )

    # 主動檢查 hf_xet,沒裝就大聲警告(50 GB 用普通 HTTP 拉會反覆 timeout)
    try:
        import hf_xet  # noqa: F401
        print(f"[ok]       hf_xet 已安裝(Xet Storage 加速)", flush=True)
    except ImportError:
        print(f"\n{'⚠'*32}", flush=True)
        print(f"⚠  hf_xet 沒裝。VocalVerse 用 Xet Storage,fallback HTTP 會慢且容易 timeout。", flush=True)
        print(f"⚠  強烈建議先裝再跑:  pip install hf_xet", flush=True)
        print(f"⚠  繼續用普通 HTTP 下載中…(中斷可重跑,snapshot_download 自動 resume)\n", flush=True)

    # 提高 timeout 上限,預設 10s 對跨海連線太短
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    print(f"[download] VocalVerse HF dataset ({VOCALVERSE_HF_REPO})\n"
          f"           → {target_dir}\n"
          f"           ⚠ ~50 GB,需要顯著磁碟空間\n", flush=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Retry loop:跨海連線 timeout 是常態,重試到自然完成
    # snapshot_download 內建單檔 resume,所以 retry 不會重下已完成的檔
    import time as _time
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id=VOCALVERSE_HF_REPO,
                repo_type="dataset",
                local_dir=str(target_dir),
                ignore_patterns=[".DS_Store"],
                max_workers=4,  # 並行太多反而 timeout 機率高
            )
            break
        except Exception as e:
            print(f"\n[retry {attempt}/{max_attempts}] snapshot_download 失敗: {type(e).__name__}: {e}",
                  flush=True)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"VocalVerse 下載失敗 {max_attempts} 次,放棄。\n"
                    f"建議:\n"
                    f"  1. 確認網路穩定(VPN / 換線路)\n"
                    f"  2. pip install hf_xet 後重跑\n"
                    f"  3. 重跑 `python scripts/bootstrap_downloads.py --only vv` 可從斷點續傳"
                )
            wait = 10 * attempt
            print(f"           {wait} 秒後重試…", flush=True)
            _time.sleep(wait)

    print(f"[done]     VocalVerse → {target_dir}\n", flush=True)


def fetch_vocalverse_labels(data_root: Path, dry_run: bool) -> None:
    """
    VocalVerse 標籤 xlsx(amateur MOS + pro 4-dim)。

    為什麼跟 wav 分開:
        標籤檔小(< 5 MB)但對 MOS filter 必要;有些情境只要拿標籤先看分布,
        不需先下整 50 GB wav。snapshot_download 走 allow_patterns 只抓 label 目錄。
    """
    label_dir = data_root / "VocalVerse" / VOCALVERSE_LABEL_DIR
    expected = [label_dir / f for f in VOCALVERSE_LABEL_FILES]
    if all(p.exists() for p in expected):
        print(f"[skip]     VocalVerse labels 已存在 ({label_dir}/)\n", flush=True)
        return
    if dry_run:
        print(f"[dry-run]  would download VocalVerse labels → {label_dir}\n", flush=True)
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "下載 VocalVerse labels 需要 huggingface_hub:\n  pip install huggingface_hub"
        )

    label_dir.mkdir(parents=True, exist_ok=True)
    for fname in VOCALVERSE_LABEL_FILES:
        rel = f"{VOCALVERSE_LABEL_DIR}/{fname}"
        print(f"[download] {rel}", flush=True)
        # 為什麼用 hf_hub_download 而非 snapshot_download with allow_patterns:
        #   allow_patterns 在 snapshot 內部仍會掃描整個 repo manifest;
        #   單檔下載對 5 個小 xlsx 效率更高
        hf_hub_download(
            repo_id=VOCALVERSE_HF_REPO,
            repo_type="dataset",
            filename=rel,
            local_dir=str(data_root / "VocalVerse"),
        )
    print(f"[done]     VocalVerse labels → {label_dir}\n", flush=True)


# ── CLI ───────────────────────────────────────────────
TASKS = {
    "vocoder":   fetch_nsvb_vocoder,
    "vae_mle":   fetch_nsvb_vae_mle,
    "m4":        fetch_m4singer,
    "vv":        fetch_vocalverse,
    "vv_labels": fetch_vocalverse_labels,
}


def main() -> None:
    # Windows console cp950 對 emoji / 中文 print 可能炸,強制 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="NSVB-ZH bootstrap: 一鍵下載 dataset + pretrained ckpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root", default="data",
        help="dataset 根目錄 (預設 data/);會建立 m4singer/ 與 VocalVerse/",
    )
    parser.add_argument(
        "--ckpt-root", default="checkpoints",
        help="ckpt 根目錄 (預設 checkpoints/);會建立 nsvb_pretrained/ 子目錄",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[], choices=list(TASKS.keys()),
        help="跳過指定任務,可多選 (例:--skip m4 vv 只下 ckpt 不下 wav)",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, choices=list(TASKS.keys()),
        help="只跑指定任務(與 --skip 互斥),可多選",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只印「會做什麼」不真的下載,debug 用",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    ckpt_root = Path(args.ckpt_root).resolve()
    print(f"[bootstrap] data_root = {data_root}")
    print(f"[bootstrap] ckpt_root = {ckpt_root}")
    if args.dry_run:
        print(f"[bootstrap] DRY RUN — no files will be downloaded\n")
    else:
        print()

    # 決定要跑哪些
    if args.only:
        if args.skip:
            print("ERROR: --only 與 --skip 互斥")
            sys.exit(1)
        plan = list(args.only)
    else:
        plan = [k for k in TASKS.keys() if k not in args.skip]
    print(f"[bootstrap] plan = {plan}\n")

    # 對應 root
    root_for = {
        "vocoder":   ckpt_root,
        "vae_mle":   ckpt_root,
        "m4":        data_root,
        "vv":        data_root,
        "vv_labels": data_root,
    }
    failed: list[str] = []
    for task in plan:
        fn = TASKS[task]
        try:
            fn(root_for[task], args.dry_run)
        except Exception as e:
            print(f"\n[ERROR]    task '{task}' failed: {e}\n", flush=True)
            failed.append(task)

    print(f"{'='*64}")
    if failed:
        print(f"[bootstrap] DONE with {len(failed)} failed: {failed}")
        print(f"            → 修好後重跑同指令,已成功的會自動 skip")
        sys.exit(1)
    print(f"[bootstrap] ALL DONE ✓")
    print(f"\n下一步:")
    print(f"  python -m nsvb.data.binarizer --dataset m4singer")
    print(f"  python -m nsvb.data.binarizer --dataset vocalverse "
          f"--vocalverse-amateur-score-max 3.0 --vocalverse-chunk-sec 5.0")


if __name__ == "__main__":
    main()
