"""
nsvb/data/vocalverse_mos.py
=============================

【這支檔案做什麼】
讀取 VocalVerse 自帶的人類標記 xlsx，提供「依 MOS / 專業多維評分過濾 SampleSpec」
的工具，讓 binarize 階段只保留「真正業餘」的樣本。

【兩份 xlsx】
1. `Amateur_overall_mos_avg5.xlsx`（"non-music-major"-業餘評審 5 位平均）
   - 欄位：歌曲id | 歌曲名称 | 录音id | 分数_1..5 | 总分
   - 总分=5 評審 1-5 分加總；MOS = 总分/5
   - 每筆 1 個 score；覆蓋率 100%

2. `Professional_multidim_annotations_raw_..._Timbre_Breath_Emotion_Technique.xlsx`
   （兩位專業聲樂教練評審）
   - 欄位：record_id | 评级打分者姓名/昵称 | "音色" | "情感" | "技巧" | "气息控制"
     維度打分（皆 1-5）
   - 注意：929 筆 record_id 各被「**單一**」評審（A 442 筆、B 487 筆，幾乎不重疊；
     僅 30 筆雙評估評審間一致性，見 paper Table 2）
   - 909 筆有完整 4 維分數（其餘有 NaN）

【為什麼分兩份】
原作者論文描述：先用業餘群眾標記 (165 評審) 嘗試大規模「整體好聽度」標記但因
forced distribution + 平均效應 → 多數樣本聚集 [2-4] 中間段；於是改用 2 位
professional 對 4 個維度（音色/情感/技巧/氣息）細分標記，建立後續訓練 dataset。

【為什麼採用 (技巧+氣息)/2 為主要 NSVB-ZH 過濾分數】
NSVB-ZH 訓練的 M 是「修飾技術」(把 amateur 的 vocal mechanics 推向 pro)；
四個維度中：
  - **技巧 / 氣息控制**：vocal mechanics（vibrato、portamento、support、phrasing），
    M 直接訓練的目標。論文 Table 1 註明這兩維「significantly improvable through training」
  - **音色**：voice timbre，論文 Table 1 與 §3.1.5 強調「largely related to physiological
    and acoustic characteristics」=> 不可改、也不該改。NSVB-ZH 用 spk_emb 鎖音色
    （Risk 4 防護），絕對不該以音色分數過濾 amateur
  - **情感**：emotion 是宏觀演出，部分被 breath 與 dynamic shaping 帶出，但 M 不
    直接訓練；做為次要 signal 可選

實測相關性（909 筆 spearman）：
    技巧 ↔ 氣息            : 0.69  ← 強相關，合理組合
    技巧 ↔ amateur_MOS    : 0.38  ← 業餘評分對技術的辨識力中等
    技巧 ↔ 音色            : (高，但 timbre 不該過濾)
    技巧 ↔ 情感            : (中)

定義：
    amateur_score = (技巧 + 氣息控制) / 2     # 兩者皆 1-5 整數，平均後 [1.0, 5.0]

【閾值推薦（929 筆中 909 筆有 pro 標記）】
依 amateur_score ≤ X 過濾後的覆蓋（per-singer median 是「同 song_id 留存樣本」中位數）：

| amateur_score ≤ X | 留 N | singers | 總時長 | per-singer median |
|---|---|---|---|---|
| 2.0 | 202 | 32/33 |  11.1 h | 6  | 太少，過擬合 |
| 2.5 | 371 | 33/33 |  20.6 h | 11 | 強過濾，省時間優先選 |
| **3.0** | **536** | **33/33** | **29.8 h** | **17** | ⭐ **預設**：與 M4Singer 30h 對齊 |
| 3.5 | 676 | 33/33 |  37.8 h | 21 | 含 average，不夠 amateur |

**預設推薦 amateur_score ≤ 3.0**：
  - 總時長 29.8h ≈ M4Singer 30h（pro/amateur 數量級平衡）
  - 33 singer 全保留，per-singer 中位 17 樣本（min ~12），無過擬合風險
  - 留下的樣本 mean amateur_score=2.40，即「技術明顯偏弱」區間，與 NSVB-ZH 設計一致

【次要：MOS / 其他維度作為 corroborator】
若想加嚴可以同時要求：
    --vocalverse-mos-max 3.5 (amateur agreement 不該太離譜)
    --vocalverse-technique-max 3.0 (單獨對技巧加 cap)
所有 set 的 max 用 AND 組合（樣本必須**全部滿足**才保留）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ── xlsx 路徑常數 ────────────────────────────────────────────
DEFAULT_LABEL_DIR_NAME = "VocalVerse_Datasets-human_labels"
AMATEUR_XLSX = "Amateur_overall_mos_avg5.xlsx"
PRO_XLSX = (
    "Professional_multidim_annotations_raw_Timbre_Breath_Emotion_Technique.xlsx"
)

# Pro xlsx 欄名（"中文 curly quotes"，與檔案完全一致；需小心字符）
COL_TIMBRE = "“音色”维度打分"
COL_EMOTION = "“情感”维度打分"
COL_TECHNIQUE = "“技巧”维度打分"
COL_BREATH = "“气息控制”维度打分"


# ── 標籤資料結構 ───────────────────────────────────────────
@dataclass
class VocalVerseLabels:
    """單一錄音的所有 label 資訊（amateur MOS + 4-dim professional）。

    為什麼分這麼細：
        binarize 過濾僅用 (technique, breath) 為主，但保留全部 5 個維度方便：
        1. 將來做更精細實驗（例如 emotion-aware training）
        2. JSD gate / debug 視覺化
        3. user 自訂多維過濾組合
    """
    record_id: str
    song_id: str
    amateur_mos: Optional[float] = None      # 1-5；None=無此 label
    pro_technique: Optional[float] = None    # 1-5；None=無此 label / NaN
    pro_breath: Optional[float] = None
    pro_timbre: Optional[float] = None
    pro_emotion: Optional[float] = None

    @property
    def amateur_score(self) -> Optional[float]:
        """NSVB-ZH 主要過濾分數：(技巧 + 氣息控制) / 2。

        為什麼是這個組合：見檔頭「為什麼採用 (技巧+氣息)/2」一節
        為什麼用 None 而非 NaN：caller 用 `is None` 判斷比 `math.isnan` 直觀
        """
        if self.pro_technique is None or self.pro_breath is None:
            return None
        return (self.pro_technique + self.pro_breath) / 2.0


def find_label_dir(vocalverse_root: Path) -> Optional[Path]:
    """自動找 label 目錄（VocalVerse 官方 zip 解壓後的標準位置）。"""
    candidate = vocalverse_root / DEFAULT_LABEL_DIR_NAME
    if candidate.is_dir():
        return candidate
    return None


def find_label_xlsx(vocalverse_root: Path) -> Optional[Path]:
    """
    向後相容：找 amateur MOS xlsx。

    為什麼保留：早期 binarizer 版本只用 amateur MOS；現有 caller 仍可呼叫此函式
    """
    label_dir = find_label_dir(vocalverse_root)
    if label_dir is None:
        return None
    candidate = label_dir / AMATEUR_XLSX
    if candidate.exists():
        return candidate
    return None


# ── 載入 ────────────────────────────────────────────────────
def load_amateur_mos(xlsx_path: Path) -> Dict[Tuple[str, str], float]:
    """
    讀 Amateur xlsx → (歌曲id, 录音id) → MOS dict。向後相容舊 API。
    """
    df = pd.read_excel(str(xlsx_path))
    required = {"歌曲id", "录音id", "总分"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"xlsx 缺欄位 {missing}；確認 {xlsx_path} 是 VocalVerse 官方 "
            f"{AMATEUR_XLSX}"
        )
    out: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        song_id = str(row["歌曲id"])
        rec_id = str(row["录音id"])
        out[(song_id, rec_id)] = float(row["总分"]) / 5.0
    return out


def load_vocalverse_labels(
    label_dir: Path,
    require_pro: bool = False,
) -> Dict[str, VocalVerseLabels]:
    """
    讀兩份 xlsx 合併成 record_id → VocalVerseLabels 的 dict。

    Args:
        label_dir:    {vocalverse_root}/VocalVerse_Datasets-human_labels/
        require_pro:  True 則只回傳同時有 amateur + pro 完整 4 維分數的紀錄；
                      False 則 amateur side 為主，pro 缺則該欄為 None

    Returns:
        Dict[record_id_str -> VocalVerseLabels]

    為什麼用 record_id 當 key（不是 (歌曲id, record_id)）：
        record_id 全 dataset 唯一（929 筆都不重複），單 key 簡單；
        歌曲id 從 amateur xlsx 拿，存進 VocalVerseLabels.song_id
    """
    ama_path = label_dir / AMATEUR_XLSX
    pro_path = label_dir / PRO_XLSX
    if not ama_path.exists():
        raise FileNotFoundError(f"Amateur xlsx not found: {ama_path}")

    ama = pd.read_excel(str(ama_path))
    ama["MOS"] = ama["总分"] / 5.0

    # Pro xlsx 是可選；若不存在僅用 amateur
    pro_map: Dict[str, dict] = {}
    if pro_path.exists():
        pro = pd.read_excel(str(pro_path))
        # numeric coerce（NaN 樣本將跳過）
        for c in (COL_TIMBRE, COL_EMOTION, COL_TECHNIQUE, COL_BREATH):
            pro[c] = pd.to_numeric(pro[c], errors="coerce")
        for _, r in pro.iterrows():
            rid = str(r["record_id"])
            pro_map[rid] = {
                "pro_technique": (None if pd.isna(r[COL_TECHNIQUE])
                                  else float(r[COL_TECHNIQUE])),
                "pro_breath":    (None if pd.isna(r[COL_BREATH])
                                  else float(r[COL_BREATH])),
                "pro_timbre":    (None if pd.isna(r[COL_TIMBRE])
                                  else float(r[COL_TIMBRE])),
                "pro_emotion":   (None if pd.isna(r[COL_EMOTION])
                                  else float(r[COL_EMOTION])),
            }
    else:
        print(f"[vocalverse_mos] WARNING: pro xlsx not found at {pro_path}; "
              f"only amateur MOS available", flush=True)

    # 合併
    out: Dict[str, VocalVerseLabels] = {}
    for _, r in ama.iterrows():
        rid = str(r["录音id"])
        sid = str(r["歌曲id"])
        pro_part = pro_map.get(rid, {})
        labels = VocalVerseLabels(
            record_id=rid,
            song_id=sid,
            amateur_mos=float(r["MOS"]),
            pro_technique=pro_part.get("pro_technique"),
            pro_breath=pro_part.get("pro_breath"),
            pro_timbre=pro_part.get("pro_timbre"),
            pro_emotion=pro_part.get("pro_emotion"),
        )
        if require_pro and labels.amateur_score is None:
            continue
        out[rid] = labels
    return out


# ── 多維過濾 ───────────────────────────────────────────────
@dataclass
class FilterCriteria:
    """
    多維過濾條件。任何 set 的 max 都會被 AND 起來（必須**全部滿足**）。

    為什麼預設都 None：
        None=該維度不參與過濾；user 只用想用的維度，不需把無意義的閾值設到 5.0
    """
    # 主要：(技巧 + 氣息) / 2，NSVB-ZH 推薦
    amateur_score_max: Optional[float] = None
    # 個別 pro 維度
    technique_max: Optional[float] = None
    breath_max: Optional[float] = None
    timbre_max: Optional[float] = None       # 不建議用（physiology-locked）
    emotion_max: Optional[float] = None
    # Amateur MOS（次要 corroborator）
    mos_max: Optional[float] = None

    def is_active(self) -> bool:
        return any(v is not None for v in [
            self.amateur_score_max, self.technique_max, self.breath_max,
            self.timbre_max, self.emotion_max, self.mos_max,
        ])

    def check(self, labels: VocalVerseLabels) -> Tuple[bool, str]:
        """
        判斷單一 record 是否通過所有 active 過濾條件。

        Returns:
            (kept, reject_reason): kept=True → keep；False → 根據 reason 統計被拒原因
        """
        if self.amateur_score_max is not None:
            s = labels.amateur_score
            if s is None:
                return False, "missing_pro"
            if s > self.amateur_score_max:
                return False, "high_amateur_score"
        if self.technique_max is not None:
            if labels.pro_technique is None:
                return False, "missing_technique"
            if labels.pro_technique > self.technique_max:
                return False, "high_technique"
        if self.breath_max is not None:
            if labels.pro_breath is None:
                return False, "missing_breath"
            if labels.pro_breath > self.breath_max:
                return False, "high_breath"
        if self.timbre_max is not None:
            if labels.pro_timbre is None:
                return False, "missing_timbre"
            if labels.pro_timbre > self.timbre_max:
                return False, "high_timbre"
        if self.emotion_max is not None:
            if labels.pro_emotion is None:
                return False, "missing_emotion"
            if labels.pro_emotion > self.emotion_max:
                return False, "high_emotion"
        if self.mos_max is not None:
            if labels.amateur_mos is None:
                return False, "missing_mos"
            if labels.amateur_mos > self.mos_max:
                return False, "high_mos"
        return True, ""


def filter_samples(
    samples: Iterable,                                     # List[SampleSpec]
    labels_map: Dict[str, VocalVerseLabels],
    criteria: FilterCriteria,
    on_missing_record: str = "drop",
) -> Tuple[List, dict]:
    """
    對 list_vocalverse 產出的 SampleSpec 列表做多維過濾。

    Args:
        samples:    SampleSpec 列表（item_id 為 "{歌曲id}__{录音id}"）
        labels_map: load_vocalverse_labels 的輸出（record_id → VocalVerseLabels）
        criteria:   FilterCriteria
        on_missing_record:
            樣本 record_id 不在 labels_map 時：
            "drop"  — 丟棄（保守）
            "keep"  — 保留（樂觀，假設未標記為 amateur）
            "raise" — 報錯

    Returns:
        (filtered_samples, stats_dict)

    為什麼 on_missing_record 預設 "drop"：
        參考舊版 mos filter 行為；保證輸出資料品質可預測（無未標記樣本悄悄混入）
    """
    if on_missing_record not in ("drop", "keep", "raise"):
        raise ValueError(f"on_missing_record must be drop|keep|raise, got {on_missing_record!r}")

    if not criteria.is_active():
        # 沒設任何過濾，直接回傳所有樣本
        return list(samples), {"kept": 0, "filter_active": False}

    kept: List = []
    reasons: Dict[str, int] = {}
    kept_amateur_scores: List[float] = []

    for s in samples:
        # item_id "{歌曲id}__{录音id}"
        try:
            _, rec_id = s.item_id.split("__", 1)
        except ValueError:
            if on_missing_record == "raise":
                raise ValueError(f"item_id {s.item_id!r} 不符 '{{歌曲id}}__{{录音id}}' 格式")
            reasons["bad_item_id"] = reasons.get("bad_item_id", 0) + 1
            if on_missing_record == "drop":
                continue
            kept.append(s)
            continue

        labels = labels_map.get(rec_id)
        if labels is None:
            if on_missing_record == "raise":
                raise KeyError(f"label missing for record_id={rec_id}")
            reasons["missing_record"] = reasons.get("missing_record", 0) + 1
            if on_missing_record == "drop":
                continue
            kept.append(s)
            continue

        ok, reject = criteria.check(labels)
        if ok:
            kept.append(s)
            if labels.amateur_score is not None:
                kept_amateur_scores.append(labels.amateur_score)
        else:
            reasons[reject] = reasons.get(reject, 0) + 1

    total = len(kept) + sum(reasons.values())
    stats = {
        "filter_active": True,
        "criteria": {k: v for k, v in vars(criteria).items() if v is not None},
        "total_input": total,
        "kept": len(kept),
        "rejected": dict(reasons),
        "on_missing_record": on_missing_record,
    }
    if kept_amateur_scores:
        import statistics
        stats["kept_amateur_score_mean"] = statistics.mean(kept_amateur_scores)
        stats["kept_amateur_score_min"] = min(kept_amateur_scores)
        stats["kept_amateur_score_max"] = max(kept_amateur_scores)
    return kept, stats


def format_filter_stats(stats: dict) -> str:
    """單行 print friendly 訊息。"""
    if not stats.get("filter_active", False):
        return "[vocalverse-filter] no filter active (kept all samples)"
    parts = [f"kept={stats['kept']}/{stats['total_input']}"]
    crit = stats.get("criteria", {})
    if crit:
        parts.append("criteria=" + ",".join(f"{k}={v}" for k, v in crit.items()))
    parts.append("rejected=" + ",".join(f"{k}={v}" for k, v in stats.get("rejected", {}).items()) or "rejected=0")
    if "kept_amateur_score_mean" in stats:
        parts.append(
            f"kept_score=mean{stats['kept_amateur_score_mean']:.2f} "
            f"range[{stats['kept_amateur_score_min']:.1f},{stats['kept_amateur_score_max']:.1f}]"
        )
    return "[vocalverse-filter] " + "  ".join(parts)


# ── 向後相容 wrapper ────────────────────────────────────
def filter_samples_by_mos(
    samples: Iterable,
    mos_map: Dict[Tuple[str, str], float],
    threshold: float,
    on_missing: str = "drop",
) -> Tuple[List, dict]:
    """
    舊 API（單純依 amateur MOS 過濾）；保留以向後相容。
    新程式請改用 filter_samples + FilterCriteria(mos_max=threshold)。
    """
    kept: List = []
    dropped_high = 0
    dropped_missing = 0
    kept_mos: List[float] = []
    for s in samples:
        try:
            song_id, rec_id = s.item_id.split("__", 1)
        except ValueError:
            if on_missing == "raise":
                raise ValueError(f"item_id {s.item_id!r}")
            if on_missing == "drop":
                dropped_missing += 1
                continue
            kept.append(s)
            continue
        key = (song_id, rec_id)
        if key not in mos_map:
            if on_missing == "raise":
                raise KeyError(key)
            if on_missing == "drop":
                dropped_missing += 1
                continue
            kept.append(s)
            continue
        mos = mos_map[key]
        if mos <= threshold:
            kept.append(s)
            kept_mos.append(mos)
        else:
            dropped_high += 1
    total = len(kept) + dropped_high + dropped_missing
    stats = {
        "threshold": threshold,
        "total_input": total,
        "kept": len(kept),
        "dropped_high_mos": dropped_high,
        "dropped_missing": dropped_missing,
        "on_missing": on_missing,
    }
    if kept_mos:
        stats["mos_mean"] = sum(kept_mos) / len(kept_mos)
        stats["mos_min"] = min(kept_mos)
        stats["mos_max"] = max(kept_mos)
    return kept, stats


# ── CLI sanity check ───────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="VocalVerse multi-criteria filter sanity check")
    ap.add_argument("--vocalverse-root", default="data/VocalVerse")
    ap.add_argument("--label-dir", default=None,
                    help="覆寫 label 目錄（預設 {root}/VocalVerse_Datasets-human_labels/）")
    args = ap.parse_args()

    root = Path(args.vocalverse_root)
    label_dir = Path(args.label_dir) if args.label_dir else find_label_dir(root)
    if label_dir is None or not label_dir.is_dir():
        print(f"ERROR: label dir not found (root={root})", file=sys.stderr)
        sys.exit(1)

    labels = load_vocalverse_labels(label_dir, require_pro=False)
    print(f"Loaded {len(labels)} records")

    # 統計分布
    import statistics as st
    amateur_scores = [l.amateur_score for l in labels.values() if l.amateur_score is not None]
    mos_scores = [l.amateur_mos for l in labels.values() if l.amateur_mos is not None]
    print(f"\namateur_score (技巧+氣息)/2: n={len(amateur_scores)}, "
          f"mean={st.mean(amateur_scores):.2f}, median={st.median(amateur_scores):.2f}")
    print(f"amateur_MOS:                n={len(mos_scores)}, "
          f"mean={st.mean(mos_scores):.2f}, median={st.median(mos_scores):.2f}")

    print("\n=== amateur_score (技巧+氣息)/2 threshold sweep ===")
    for thr in [2.0, 2.5, 3.0, 3.5]:
        n = sum(1 for s in amateur_scores if s <= thr)
        pct = n / len(amateur_scores) * 100
        print(f"  amateur_score ≤ {thr}: {n:4d} ({pct:.1f}%)")

    print("\n=== amateur_MOS threshold sweep（次要） ===")
    for thr in [2.0, 2.5, 3.0, 3.5]:
        n = sum(1 for s in mos_scores if s <= thr)
        pct = n / len(mos_scores) * 100
        print(f"  amateur_MOS ≤ {thr}: {n:4d} ({pct:.1f}%)")