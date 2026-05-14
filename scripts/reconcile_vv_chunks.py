"""
scripts/reconcile_vv_chunks.py
================================

【這支腳本做什麼】
偵測 VocalVerse 「整首被 skip 但 chunk 沒切完」的 binarize 缺料。
對每首來源錄音：從原始 wav 時長推算「應有幾個 chunk」，跟 binarized 目錄裡
「實際有幾個 __cNNN.npz」比對，列出缺尾的錄音。

【為什麼 verify_binarized.py 不夠、需要這支】
binarizer.py 的斷點續跑用 `{item_id}__c000.npz` 是否存在當「整首已處理過」的
代表（檢查所有 chunk 太貴）。但 Colab local-write + 背景 rsync 流程下：
  - 某 session 在 binarize / rsync 一首歌的途中被中斷
  - 該首 c000 已 sync 到 Drive，但 c020+ 還沒 sync；local 端 session 結束被清空
  - 下個 session 看到 Drive 有 c000 → 整首永久 skip → c020+ 永遠缺
verify_binarized.py 逐檔檢查「每個 .npz 本身健不健康」抓不到這種缺料——
每個檔案都是好的，錯的是「數量」。這支用「來源 wav 時長」當 ground truth 補上。

【T_mel 推算（與 audio_io.compute_mel 對齊）】
  compute_mel 用 librosa.stft(center=True) → T_mel = 1 + N // HOP_SIZE
  N = librosa.load(sr=22050) 後的取樣點數。
  dereverb（DeepFilterNet3）理論上保長度、loudness norm 保長度，故用
  soundfile 讀 header 算 N_resampled 已足夠精確（±1~2 frame，遠小於一個 chunk
  ~860 frame，不影響「缺一截」的判定）。

【用法】
  python scripts/reconcile_vv_chunks.py \
      --vv-source /content/drive/MyDrive/NSVB-ZH/data/VocalVerse \
      --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized

  # 參數要跟當初 binarize 時一致（才會挑出同一批來源錄音 + 同樣 chunk 切法）
  #   --amateur-score-max 3.0   （MOS filter，預設 3.0）
  #   --chunk-sec 5.0           （VV chunk 長度，預設 5.0）

【退出碼】
  0 = 全部錄音 chunk 數齊全（或非 --strict）
  1 = --strict 且有缺料錄音
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

# 讓 `python scripts/reconcile_vv_chunks.py` 不論從哪個 cwd 跑都能 import nsvb：
# 直接跑 script 時 sys.path[0] 是 scripts/，repo root 不在 path 上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.utils.audio_config import SAMPLE_RATE, HOP_SIZE
from nsvb.data.vocalverse_mos import (
    FilterCriteria, find_label_dir, load_vocalverse_labels,
    filter_samples, format_filter_stats,
)


# ── list_vocalverse 內聯版 ──────────────────────────────
# 為什麼不從 nsvb.data.binarizer import：binarizer.py 在 module 層就載入
# torch / Whisper / DeepFilterNet / Resemblyzer / torchcrepe 整套 GPU 抽取器，
# reconcile 完全用不到，卻會因環境裡任一重依賴沒裝好而「import 就炸」。
# list_vocalverse 本體只是「iterate 目錄 + glob wav」，內聯進來最穩。
@dataclass
class SampleSpec:
    wav_path: str
    speaker_id: str
    item_id: str
    dataset: str


def list_vocalverse(root: Path) -> list:
    """
    VocalVerse 結構：{root}/{user_id}/{wav_id}.wav
    item_id = "{user_id}__{wav_id}"（與 binarizer.list_vocalverse 完全一致，
    reconcile 才能用 item_id 去 glob 對應的 __c*.npz）。
    """
    samples = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        for wav in sorted(user_dir.glob("*.wav")):
            samples.append(SampleSpec(
                wav_path=str(wav.resolve()),
                speaker_id=user_id,
                item_id=f"{user_id}__{wav.stem}",
                dataset="vocalverse",
            ))
    return samples


def estimate_t_mel(wav_path: str) -> int:
    """
    從原始 wav header 推算 binarize 後的 mel frame 數 T_mel。
    用 soundfile 讀 header（不解碼整個檔，~毫秒級）。
    """
    info = sf.info(wav_path)
    # resample 到 SAMPLE_RATE 後的取樣點數
    n_resampled = round(info.frames * SAMPLE_RATE / info.samplerate)
    # librosa.stft(center=True): T = 1 + N // hop
    return 1 + n_resampled // HOP_SIZE


def expected_chunk_count(t_mel: int, chunk_sec: float,
                         min_remaining_sec: float) -> int:
    """
    複製 binarizer.chunk_sample 的切片邏輯，只算「會產出幾個 chunk」。
    必須與 chunk_sample 完全一致：
        for start_f in range(0, T_mel, chunk_frames):
            end_f = min(start_f + chunk_frames, T_mel)
            if (end_f - start_f) < min_frames: break
    """
    chunk_frames = int(chunk_sec * SAMPLE_RATE / HOP_SIZE)
    min_frames = int(min_remaining_sec * SAMPLE_RATE / HOP_SIZE)
    n = 0
    for start_f in range(0, t_mel, chunk_frames):
        end_f = min(start_f + chunk_frames, t_mel)
        if (end_f - start_f) < min_frames:
            break
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="偵測 VocalVerse binarize chunk 缺料（整首被 skip 但 chunk 沒切完）",
    )
    parser.add_argument("--vv-source", required=True,
                        help="VocalVerse 原始資料集根目錄（含 {user}/{wav} + human_labels）")
    parser.add_argument("--binarized-root", required=True,
                        help="binarized 根目錄（{root}/vocalverse/*.npz）")
    parser.add_argument("--dataset-name", default="vocalverse")
    parser.add_argument("--chunk-sec", type=float, default=5.0,
                        help="binarize 時用的 VV chunk 長度（要一致）")
    parser.add_argument("--min-remaining-sec", type=float, default=3.0,
                        help="binarize 時短尾段丟棄門檻（要一致）")
    parser.add_argument("--amateur-score-max", type=float, default=3.0,
                        help="binarize 時的 MOS filter 門檻（要一致，挑出同一批來源錄音）")
    parser.add_argument("--label-dir", default=None,
                        help="VocalVerse human_labels 目錄；預設自動從 --vv-source 找")
    parser.add_argument("--tolerance", type=int, default=1,
                        help="actual 與 expected 容許誤差（T_mel 推算 ±1~2 frame 可能"
                             "造成邊界 chunk ±1）；超過此值才算缺料")
    parser.add_argument("--strict", action="store_true",
                        help="有缺料錄音就 exit 1（給自動化流程用）")
    args = parser.parse_args()

    vv_source = Path(args.vv_source)
    bin_dir = Path(args.binarized_root) / args.dataset_name
    if not bin_dir.is_dir():
        print(f"[reconcile] ✗ binarized 目錄不存在: {bin_dir}")
        sys.exit(2)

    # 1. 列出來源錄音 + 套用與 binarize 一致的 MOS filter
    print(f"[reconcile] 列出 VocalVerse 來源錄音: {vv_source}")
    samples = list_vocalverse(vv_source)
    print(f"[reconcile] 原始 {len(samples)} 首")

    criteria = FilterCriteria(amateur_score_max=args.amateur_score_max)
    if criteria.is_active():
        label_dir = Path(args.label_dir) if args.label_dir else find_label_dir(vv_source)
        if label_dir is None or not label_dir.is_dir():
            print(f"[reconcile] ✗ 找不到 human_labels 目錄"
                  f"（預設找 {vv_source}/VocalVerse_Datasets-human_labels/）；"
                  f"請用 --label-dir 指定")
            sys.exit(2)
        labels_map = load_vocalverse_labels(label_dir, require_pro=False)
        samples, stats = filter_samples(
            samples, labels_map, criteria=criteria, on_missing_record="drop",
        )
        print(format_filter_stats(stats))
    print(f"[reconcile] MOS filter 後 {len(samples)} 首（這是 binarize 預期處理的數量）")

    # 2. 逐首比對 expected vs actual
    chunk_frames = int(args.chunk_sec * SAMPLE_RATE / HOP_SIZE)
    min_frames = int(args.min_remaining_sec * SAMPLE_RATE / HOP_SIZE)
    print(f"[reconcile] chunk_frames={chunk_frames}  min_frames={min_frames}  "
          f"tolerance=±{args.tolerance}")

    missing: list[tuple[str, int, int]] = []      # 完全沒有（連 c000 都沒有）
    truncated: list[tuple[str, int, int]] = []    # 有 c000 但 chunk 不足
    extra: list[tuple[str, int, int]] = []        # chunk 比預期多（異常，少見）
    total_expected = total_actual = 0
    n_err = 0

    for i, spec in enumerate(samples, 1):
        try:
            t_mel = estimate_t_mel(spec.wav_path)
        except Exception as e:
            print(f"[reconcile] ⚠️  讀不到來源 wav {spec.wav_path}: {e}")
            n_err += 1
            continue
        exp = expected_chunk_count(t_mel, args.chunk_sec, args.min_remaining_sec)
        act = len(list(bin_dir.glob(f"{spec.item_id}__c*.npz")))
        total_expected += exp
        total_actual += act

        if act == 0:
            missing.append((spec.item_id, exp, act))
        elif act < exp - args.tolerance:
            truncated.append((spec.item_id, exp, act))
        elif act > exp + args.tolerance:
            extra.append((spec.item_id, exp, act))

        if i % 100 == 0:
            print(f"[reconcile]   ...{i}/{len(samples)} 已比對")

    # 3. 報告
    n_bad = len(missing) + len(truncated) + len(extra)
    print(f"\n{'='*64}")
    print(f"[reconcile] 結果：")
    print(f"  來源錄音       : {len(samples)} 首")
    print(f"  chunk 預期總數 : {total_expected}")
    print(f"  chunk 實際總數 : {total_actual}  (差 {total_expected - total_actual})")
    print(f"  齊全           : {len(samples) - n_bad - n_err} 首")
    print(f"  完全沒有       : {len(missing)} 首")
    print(f"  缺尾 (truncated): {len(truncated)} 首")
    print(f"  異常多出       : {len(extra)} 首")
    if n_err:
        print(f"  來源讀取失敗   : {n_err} 首")

    def _dump(title, rows):
        if not rows:
            return
        print(f"\n[reconcile] {title}:")
        for item_id, exp, act in rows[:40]:
            print(f"  ✗ {item_id}  expected={exp}  actual={act}")
        if len(rows) > 40:
            print(f"  ... 還有 {len(rows) - 40} 首未列出")

    _dump("完全沒有產出的錄音（binarize 從沒成功跑過）", missing)
    _dump("缺尾的錄音（c000 在但後段 chunk 缺）", truncated)
    _dump("chunk 異常多出的錄音（請手動檢查）", extra)

    bad_ids = [r[0] for r in (missing + truncated + extra)]
    if bad_ids:
        out_path = Path(args.binarized_root) / f"{args.dataset_name}_incomplete.txt"
        out_path.write_text("\n".join(bad_ids), encoding="utf-8")
        print(f"\n[reconcile] 缺料錄音 item_id 已寫到 {out_path}")
        print(f"[reconcile] 修復步驟：")
        print(f"  1. 刪掉這些錄音在 binarized 目錄（含 Drive + local）的所有 __c*.npz")
        print(f"     （c000 還在的話 skip_existing 會繼續跳過 → 必須先刪 c000）")
        print(f"  2. 重跑 binarize：skip_existing 看不到 c000 → 自動補切完整 chunk")
        print(f"  例：在 binarized/{args.dataset_name}/ 下")
        print(f"     while read id; do rm -f \"${{id}}__c\"*.npz; done < {out_path.name}")
    else:
        print(f"\n[reconcile] ✓ 所有來源錄音 chunk 數齊全，binarize 完整")

    if args.strict and bad_ids:
        sys.exit(1)


if __name__ == "__main__":
    main()