"""PopBuTFy 跨語言 dataset 適配器(thesis §4.6 cross-lingual validation 用)。

【為什麼存在】
NSVB-ZH 架構設計為 language-agnostic;PopBuTFy 是 NSVB 原版用的英文 **paired**
amateur+pro singing dataset。跑同 pipeline 可:
1. 驗證跨語言通用性(中文 M4+VV → 英文 PopBuTFy)
2. 用 paired ground truth 補上 unpaired pipeline 缺的 paired metric
   (MCD/SSIM/F0 RMSE vs paired_pro,§4.6 paired eval 用)
3. NSVB 原版 ckpt 推理同 PopBuTFy → 直接 baseline 對照

【PopBuTFy 結構(本機:c:\\Users\\neo29\\workspace\\SVC\\NSVB\\data\\processed\\PopBuTFy_new\\data)】
{root}/
├── Female1#singing#Almost_lover_Amateur/
│   ├── Female1#singing#Almost_lover_Amateur_0.mp3
│   ├── Female1#singing#Almost_lover_Amateur_1.mp3
│   └── ... (chunks)
├── Female1#singing#Almost_lover_Professional/
│   └── ... (chunks)
└── ... (~904 folders = ~452 paired (singer, song) pairs)

【映射到 SampleSpec】
- `speaker_id` = folder name 第一個 `#` 之前(Female1 / Male2 / ...)
- `dataset`    = "popbutfy_pro" 或 "popbutfy_amateur"
- `item_id`    = `{folder_name}__{chunk_idx}` — 保留 `#` 跟既有 M4-style filter
  (`"#" in v`)相容,在 stage2 eval / listening 等下游可被自動分組

【amateur↔pro 配對】
filename swap Amateur ↔ Professional(同 NSVB 原版 PopBuTFyENBinarizer 邏輯)。
配對 dict 可 dump JSON,§4.6 paired eval 直接讀。

【限制】
- 假設 amateur chunk N 對應 pro chunk N(NSVB 原版同假設)。若 chunk 時間軸不齊,
  下游 eval 階段再用 DTW 做 chunk-level alignment。
- 檔案是 MP3,本 pipeline `load_wav` 走 librosa 理論可吃。smoke test 驗證。
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nsvb.data.binarizer import SampleSpec


CHUNK_RE = re.compile(r"_(\d+)\.mp3$")
SIDE_SUFFIX = {"pro": "_Professional", "amateur": "_Amateur"}


def list_popbutfy(root: Path, side: str) -> List[SampleSpec]:
    """掃 root 下所有 `*_{Amateur|Professional}` folders,回 SampleSpec list。

    Args:
        root: PopBuTFy 資料根目錄
        side: "pro" 或 "amateur"
    """
    if side not in SIDE_SUFFIX:
        raise ValueError(f"side must be 'pro' or 'amateur', got {side!r}")
    suffix = SIDE_SUFFIX[side]
    dataset_name = f"popbutfy_{side}"

    samples = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith(suffix):
            continue
        # speaker_id 取 first `#` 之前(Female1 / Male1)
        speaker_id = folder.name.split("#", 1)[0]
        for mp3 in sorted(folder.glob("*.mp3")):
            m = CHUNK_RE.search(mp3.name)
            if not m:
                continue
            chunk_idx = m.group(1)
            item_id = f"{folder.name}__{chunk_idx}"
            samples.append(SampleSpec(
                wav_path=str(mp3.resolve()),
                speaker_id=speaker_id,
                item_id=item_id,
                dataset=dataset_name,
            ))
    return samples


def build_pairing(root: Path) -> Dict[str, str]:
    """掃 root 下所有 Amateur folders,建 amateur item_id → pro item_id 配對 dict。

    用 filename swap Amateur → Professional(NSVB 原版同邏輯)。
    跳過沒有對應 pro folder 或對應 chunk 的 amateur 樣本。
    Returns: {amateur_item_id: pro_item_id}
    """
    pairs: Dict[str, str] = {}
    skipped_folders = 0
    skipped_chunks = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith("_Amateur"):
            continue
        pro_folder_name = folder.name.replace("_Amateur", "_Professional")
        pro_folder = root / pro_folder_name
        if not pro_folder.exists():
            skipped_folders += 1
            continue
        for mp3 in sorted(folder.glob("*.mp3")):
            m = CHUNK_RE.search(mp3.name)
            if not m:
                continue
            chunk_idx = m.group(1)
            am_item = f"{folder.name}__{chunk_idx}"
            pro_mp3 = pro_folder / mp3.name.replace("_Amateur_", "_Professional_")
            if not pro_mp3.exists():
                skipped_chunks += 1
                continue
            pairs[am_item] = f"{pro_folder_name}__{chunk_idx}"
    print(f"[popbutfy] {len(pairs)} amateur→pro pairs "
          f"(skipped: {skipped_folders} folders / {skipped_chunks} chunks)")
    return pairs


def main():
    """Smoke test CLI:確認 folder 掃描 + mp3 載入 + 配對映射都 OK。"""
    # Windows cp950 console 不能印 emoji,改 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="PopBuTFy adapter smoke test")
    ap.add_argument("--root", required=True, type=Path,
                    help="PopBuTFy data root, e.g. .../PopBuTFy_new/data")
    ap.add_argument("--n", type=int, default=5,
                    help="每 side 印幾個 sample 給人眼檢查(預設 5)")
    ap.add_argument("--pairing-json", type=Path, default=None,
                    help="若給,dump 配對 dict 到該檔(JSON)")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"root not found: {args.root}")
    print(f"[popbutfy] root: {args.root}\n")

    counts = {}
    for side in ("pro", "amateur"):
        specs = list_popbutfy(args.root, side)
        counts[side] = specs
        n_spk = len(set(s.speaker_id for s in specs))
        n_song = len(set(s.item_id.rsplit("__", 1)[0] for s in specs))
        print(f"[popbutfy] {side}: {len(specs)} chunks / {n_song} (singer,song) folders / {n_spk} speakers")
        for spec in specs[:args.n]:
            print(f"  {spec.dataset}/{spec.item_id}")
            print(f"    speaker={spec.speaker_id}, wav={Path(spec.wav_path).name}")
        print()

    pairs = build_pairing(args.root)
    if args.pairing_json:
        args.pairing_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pairing_json, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[popbutfy] pairing dumped to {args.pairing_json}")

    # 確認 librosa 能讀 mp3
    print("\n[popbutfy] librosa MP3 load smoke test:")
    try:
        import librosa
        for side in ("amateur", "pro"):
            if counts.get(side):
                sample = counts[side][0]
                wav, sr = librosa.load(sample.wav_path, sr=None)
                print(f"  [OK] {Path(sample.wav_path).name}: sr={sr}, "
                      f"len={len(wav)}, duration={len(wav)/sr:.2f}s")
    except Exception as e:
        print(f"  [FAIL] librosa load: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()