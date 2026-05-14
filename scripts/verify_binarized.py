"""
scripts/verify_binarized.py
=============================

【這支腳本做什麼】
Phase 0 binarize 完成後的完整性驗證。逐一打開 {root}/{dataset} 下每個 .npz，
檢查能否載入、key 是否齊、shape 是否一致、數值是否健康，最後印出 summary +
壞檔清單。

【為什麼需要這支】
- Colab local-write + 背景 rsync 流程：rsync 中途被打斷可能留下「截斷的 .npz」
  （檔案存在但 zip central directory 不完整 → np.load 報 BadZipFile）
- binarize 過程曾出現單首歌 crash（例如 import 錯誤）→ 該檔可能完全沒寫，
  也可能寫到一半
- numpy ABI 不一致曾經發生 → 極端情況下可能寫出數值異常的 array
- dataset.py 的 __init__ 雖然會 try/except 丟掉壞檔，但「靜默丟掉」=訓練樣本
  數悄悄變少卻不報錯。這支把問題顯式攤開。

【檢查項目】
  1. 可載入       np.load 不報 BadZipFile / OSError / ValueError / EOFError
  2. key 齊全     必要 key 全在（wav/mel/f0/voicing/register_soft/register_id/
                  ppg/spk_emb + meta_*）
  3. shape 正確   mel[T,80] f0[T] voicing[T] register_soft[T,5] register_id[T]
                  ppg[T,1280] spk_emb[256]；所有 framewise key 的 T 軸一致
  4. 數值健康     mel/f0/ppg/spk_emb 無 NaN/Inf；f0 落在 [0, F0_FMAX]；
                  mel 非全零；ppg 非全零；T >= 最低門檻
  5. phoneme_id   選擇性回報（cluster_ppg 未跑時不存在 → 不算錯，只統計覆蓋率）

【用法】
  python scripts/verify_binarized.py --root data/binarized --dataset m4singer
  python scripts/verify_binarized.py --root data/binarized --dataset vocalverse
  # 一次驗兩個：
  python scripts/verify_binarized.py --root data/binarized --dataset m4singer vocalverse
  # 嚴格模式：任一壞檔 → exit code 1（給 CI / 自動化流程用）
  python scripts/verify_binarized.py --root data/binarized --dataset m4singer --strict

【退出碼】
  0 = 全部健康（或非 --strict 模式）
  1 = --strict 且有壞檔
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 讓 `python scripts/verify_binarized.py` 不論從哪個 cwd 跑都能 import nsvb。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# F0 上限（與 nsvb/utils/audio_config.F0_FMAX 對齊；import 失敗時 fallback）
try:
    from nsvb.utils.audio_config import F0_FMAX
except Exception:
    F0_FMAX = 1400.0

# ── 每個 .npz 應有的 key 與其期望 shape ──────────────────
# None 維度 = 該軸長度視樣本而定（T = mel frame 數，N = wav 取樣點數）
REQUIRED_FRAMEWISE = {
    # key:            (期望維度數, 最後一軸固定長度 or None)
    "mel": (2, 80),
    "f0": (1, None),
    "voicing": (1, None),
    "register_soft": (2, 5),
    "register_id": (1, None),
    "ppg": (2, 1280),
}
REQUIRED_OTHER = {
    "wav": (1, None),
    "spk_emb": (1, 256),
}
REQUIRED_META = [
    "meta_dataset", "meta_speaker_id", "meta_item_id",
    "meta_sample_rate", "meta_hop_size",
]
MIN_FRAMES = 4  # T < 4 時 PatchNCE contrastive 訊號過弱（見 losses.py 註解）


def check_one(path: Path) -> tuple[bool, list[str], dict]:
    """
    驗證單一 .npz。

    Returns:
        ok:     是否全項通過
        errors: 失敗原因清單（空 = 通過）
        info:   附帶資訊（T, has_phoneme_id, ...），即使失敗也盡量填
    """
    errors: list[str] = []
    info: dict = {"T": None, "has_phoneme_id": False}

    # 1. 可載入
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:  # BadZipFile / OSError / ValueError / EOFError ...
        return False, [f"load failed: {type(e).__name__}: {e}"], info

    try:
        files = set(data.files)

        # 2. key 齊全
        missing = [
            k for k in
            (*REQUIRED_FRAMEWISE, *REQUIRED_OTHER, *REQUIRED_META)
            if k not in files
        ]
        if missing:
            errors.append(f"missing keys: {missing}")
        info["has_phoneme_id"] = "phoneme_id" in files

        # 3. shape 正確 + T 軸一致
        t_axis = None
        for key, (ndim, last_dim) in REQUIRED_FRAMEWISE.items():
            if key not in files:
                continue
            arr = data[key]
            if arr.ndim != ndim:
                errors.append(f"{key}: ndim {arr.ndim} != {ndim}")
                continue
            if last_dim is not None and arr.shape[-1] != last_dim:
                errors.append(f"{key}: last dim {arr.shape[-1]} != {last_dim}")
            # T 軸（framewise key 的第 0 軸）必須一致
            this_t = arr.shape[0]
            if t_axis is None:
                t_axis = this_t
            elif this_t != t_axis:
                errors.append(f"{key}: T axis {this_t} != {t_axis} (其他 framewise key)")
        info["T"] = t_axis

        for key, (ndim, last_dim) in REQUIRED_OTHER.items():
            if key not in files:
                continue
            arr = data[key]
            if arr.ndim != ndim:
                errors.append(f"{key}: ndim {arr.ndim} != {ndim}")
            elif last_dim is not None and arr.shape[-1] != last_dim:
                errors.append(f"{key}: shape {arr.shape} != (...,{last_dim})")

        # phoneme_id 若存在，T 軸也要對齊
        if info["has_phoneme_id"] and t_axis is not None:
            pid = data["phoneme_id"]
            if pid.ndim != 1 or pid.shape[0] != t_axis:
                errors.append(f"phoneme_id: shape {pid.shape} != ({t_axis},)")

        # T 下限
        if t_axis is not None and t_axis < MIN_FRAMES:
            errors.append(f"T={t_axis} < MIN_FRAMES={MIN_FRAMES} (PatchNCE 訊號過弱)")

        # 4. 數值健康
        for key in ("mel", "f0", "ppg", "spk_emb"):
            if key not in files:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                errors.append(f"{key}: {n_bad} NaN/Inf 值")
        # mel / ppg 不該全零（全零 = 抽取失敗的徵兆）
        for key in ("mel", "ppg"):
            if key in files:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.size and not np.any(arr):
                    errors.append(f"{key}: 全零 array")
        # f0 範圍
        if "f0" in files:
            f0 = np.asarray(data["f0"], dtype=np.float32)
            f0_voiced = f0[f0 > 0]
            if f0_voiced.size:
                lo, hi = float(f0_voiced.min()), float(f0_voiced.max())
                if hi > F0_FMAX * 1.01:  # 1% 容差
                    errors.append(f"f0: max {hi:.1f} > F0_FMAX {F0_FMAX}")
                if lo < 0:
                    errors.append(f"f0: 負值 {lo:.1f}")
        # spk_emb 應為單位向量附近（Resemblyzer L2-normed）→ 全零代表抽取失敗
        if "spk_emb" in files:
            se = np.asarray(data["spk_emb"], dtype=np.float32)
            if se.size and not np.any(se):
                errors.append("spk_emb: 全零 array")

    finally:
        data.close()

    return (len(errors) == 0), errors, info


def verify_dataset(root: Path, dataset: str, max_report: int) -> bool:
    """驗證單一 dataset 目錄，回傳是否全數通過。"""
    ds_dir = root / dataset
    if not ds_dir.is_dir():
        print(f"[verify] ✗ 目錄不存在: {ds_dir}")
        return False

    npz_paths = sorted(ds_dir.rglob("*.npz"))
    print(f"\n{'='*64}")
    print(f"[verify] {dataset}: 找到 {len(npz_paths)} 個 .npz  ({ds_dir})")
    if not npz_paths:
        print(f"[verify] ✗ {dataset} 沒有任何 .npz 檔")
        return False

    bad: list[tuple[Path, list[str]]] = []
    t_values: list[int] = []
    n_with_pid = 0
    # chunk 統計：把 {item_id}__c000.npz 還原回來源錄音 {item_id}，
    # 數「不同來源錄音」有幾首 + 每首切了幾個 chunk
    chunks_per_recording: dict[str, int] = {}

    for i, p in enumerate(npz_paths, 1):
        ok, errs, info = check_one(p)
        if info["T"] is not None:
            t_values.append(info["T"])
        if info["has_phoneme_id"]:
            n_with_pid += 1
        if not ok:
            bad.append((p, errs))
        # 來源錄音 stem：__c 之前的部分（未切 chunk 的檔沒有 __c → stem = 整個檔名）
        stem = p.stem.split("__c")[0]
        chunks_per_recording[stem] = chunks_per_recording.get(stem, 0) + 1
        if i % 2000 == 0:
            print(f"[verify]   ...{i}/{len(npz_paths)} 已檢查，{len(bad)} 壞檔")

    n_ok = len(npz_paths) - len(bad)
    n_recordings = len(chunks_per_recording)
    is_chunked = n_recordings < len(npz_paths)
    print(f"\n[verify] {dataset} 結果：")
    print(f"  .npz 總數  : {len(npz_paths)}")
    print(f"  健康       : {n_ok}/{len(npz_paths)}")
    print(f"  壞檔       : {len(bad)}")
    if is_chunked:
        cpr = np.array(list(chunks_per_recording.values()))
        print(f"  來源錄音   : {n_recordings} 首 → 切成 {len(npz_paths)} 個 chunk")
        print(f"  chunk/首   : min={cpr.min()} max={cpr.max()} "
              f"mean={cpr.mean():.1f} median={int(np.median(cpr))}")
        print(f"  ⚠️  請核對「來源錄音」數是否等於 binarize 預期的歌曲數；"
              f"少了 = 有整首在 binarize 中途 crash 沒產出")
    else:
        print(f"  來源錄音   : {n_recordings} 首（未切 chunk，1 首 = 1 .npz）")
    if t_values:
        t_arr = np.array(t_values)
        print(f"  T (frames) : min={t_arr.min()} max={t_arr.max()} "
              f"mean={t_arr.mean():.0f} median={int(np.median(t_arr))}")
    print(f"  phoneme_id : {n_with_pid}/{len(npz_paths)} 已有 "
          f"({'cluster_ppg 已跑' if n_with_pid == len(npz_paths) else 'cluster_ppg 尚未跑或未跑完' if n_with_pid else 'cluster_ppg 尚未跑'})")

    if bad:
        print(f"\n[verify] 壞檔清單（前 {min(max_report, len(bad))} 筆）：")
        for p, errs in bad[:max_report]:
            print(f"  ✗ {p.name}")
            for e in errs:
                print(f"      - {e}")
        if len(bad) > max_report:
            print(f"  ... 還有 {len(bad) - max_report} 筆未列出")
        # 把完整壞檔清單寫到檔案，方便後續針對性重 binarize
        bad_list_path = ds_dir.parent / f"{dataset}_bad_npz.txt"
        bad_list_path.write_text(
            "\n".join(str(p) for p, _ in bad), encoding="utf-8",
        )
        print(f"\n[verify] 完整壞檔路徑已寫到 {bad_list_path}")
        print(f"[verify]   → 刪掉這些檔後重跑 binarize（skip_existing 會補回缺檔）")
    else:
        print(f"\n[verify] ✓ {dataset} 全數通過")

    return len(bad) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 binarize 完整性驗證",
    )
    parser.add_argument("--root", default="data/binarized",
                        help="binarized 根目錄（{root}/{dataset}/*.npz）")
    parser.add_argument("--dataset", nargs="+", required=True,
                        help="要驗證的 dataset 名稱，可多個 "
                             "(e.g. --dataset m4singer vocalverse)")
    parser.add_argument("--max-report", type=int, default=30,
                        help="壞檔清單最多印幾筆（完整清單一律寫檔）")
    parser.add_argument("--strict", action="store_true",
                        help="任一 dataset 有壞檔就 exit 1（給自動化流程用）")
    args = parser.parse_args()

    root = Path(args.root)
    all_ok = True
    for ds in args.dataset:
        ok = verify_dataset(root, ds, args.max_report)
        all_ok = all_ok and ok

    print(f"\n{'='*64}")
    if all_ok:
        print("[verify] ✓ 全部 dataset 通過完整性檢查")
    else:
        print("[verify] ✗ 有 dataset 未通過——見上方壞檔清單")

    if args.strict and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()