"""
nsvb/data/txt_processors/zh.py
================================

【這支檔案做什麼】
中文文字 → 音素序列（聲母 / 韻母 + 聲調）轉換器。

輸入：中文歌詞字串（可含標點與半形/全形混用）
輸出：phoneme 序列（list[str]），如：
    "你好世界" → ['n', 'i3', 'h', 'ao3', 'sh', 'ix4', 'j', 'ie4']

【為什麼需要它】
NSVB 原本用英文（CMU dict 音素集）；NSVB-ZH 必須換成中文聲韻母系統。
這支處理器有兩個用途：
  1. **訓練端**（Phase 0 binarizer）：把歌詞文字轉成 phoneme id 序列，可選地做為
     輔助標註（例如未來若加 PPG-ground-truth 對齊損失時可用）。
  2. **記憶儲存格式**：所有歌詞統一用 INITIALS+FINALS_TONE3 編碼，便於跨資料集
     共享 phoneme 詞表（VocalVerse 與 M4Singer 共用同一張表）。

【為什麼不直接用 PPG 而還要算文字 phoneme】
NSVB-ZH 主架構確實是用 PPG（從音檔抽）作為 D_z 條件，**理論上不依賴文字**。
但保留文字 → phoneme 流程的價值在於：
  - **資料策展驗證**：JSD 檢查需要離散 phoneme ID，文字端 phoneme 是「真值」，
    可以用來校正 PPG argmax 的偏差（例如 Whisper 把某音素系統性誤判時）。
  - **歌詞對齊（未來）**：若 Phase 3 想實作 MFA-style 強制對齊（Mode C 用），
    就需要文字 phoneme 序列當對齊目標。
  - **除錯工具**：訓練不收斂時，比對「文字 phoneme 分布」vs「PPG argmax 分布」
    可以快速定位問題（PPG 抽錯 vs 真的有訓練問題）。

【設計選擇】
1. **聲母與韻母分開**：和原 NSVB zh.py 一致，輸出 token 為 [shengmu, yunmu+tone]，
   單音節若只有韻母（如「啊 a1」）則只輸出韻母。
2. **聲調用 5 表示輕聲**：原 NSVB 對「沒有聲調 = 輕聲」這類邊界情況也是用 5 標記，
   保持相容。
3. **使用 pypinyin 而非 g2pM**：
   - pypinyin 維護活躍、純 Python、無深度模型依賴
   - g2pM 在多音字上更準（用 BiLSTM），但對歌詞場景（多為現代流行歌詞詞彙）
     pypinyin 已足夠，且 g2pM 啟動慢、體積大
   - **未來可選升級**：若發現多音字錯誤率高，再切換到 g2pM
4. **不做文本正規化（NSW）**：歌詞通常乾淨，不像 TTS 文本需要把「3.14 → 三點一四」。
   若 VocalVerse 的歌詞 metadata 有數字，再加正規化。
"""

import re
from typing import List, Tuple

from pypinyin import pinyin, Style


# 句界標記：phoneme 序列中的「停頓 / 邊界」記號
# 為什麼用 "|"：和原 NSVB phone_set.json 一致，便於後續若需要 cross-language tokenizer
SIL_TOKEN = "|"

# 合法 pinyin 韻母字元集（小寫 ASCII + ü）；任何 yunmu 字串只能由這些字元組成
# 為什麼要這個檢查：pypinyin 對英文/數字/標點會原樣回傳（fallback behaviour），
# 若不過濾，"Hello"、"123" 會被誤當成 yunmu 加上聲調 → phoneme 序列被汙染
_PINYIN_CHAR_RE = re.compile(r"^[a-zü]+$")


def _is_valid_pinyin_unit(s: str) -> bool:
    """判斷一個字串是否為合法 pinyin 字段（聲母或韻母）。"""
    if not s:
        return False
    return bool(_PINYIN_CHAR_RE.match(s))

# 中文標點 → 半形對照（pypinyin 對全形標點處理可能不一致）
FULLWIDTH_PUNC_MAP = {
    ord(f): ord(t) for f, t in zip(
        "：，。！？【】（）％＃＠＆１２３４５６７８９０",
        ":,.!?[]()%#@&1234567890",
    )
}

# 允許保留的字元集合：中文 + 英數 + 空白 + 基本標點
# 為什麼要過濾：歌詞檔可能含特殊符號（★、♪ 等），pypinyin 遇到會回傳原字導致 phoneme 序列髒
ALLOWED_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9\s,.!?]")


def preprocess_text(text: str) -> str:
    """
    歌詞前處理：全形→半形、過濾特殊符號、標點正規化。

    為什麼這層存在：
      pypinyin 對「乾淨的中文 + 半形標點」表現最穩定。
      預先正規化能避免 phoneme 序列出現未知 token。
    """
    # 1. 全形標點轉半形
    text = text.translate(FULLWIDTH_PUNC_MAP)
    # 2. 過濾不在白名單內的字元
    text = ALLOWED_PATTERN.sub("", text)
    # 3. 折疊重複標點（"!!!" → "!"）
    text = re.sub(r"([,.!?])\1+", r"\1", text)
    # 4. 標點兩側加空格便於後續處理
    text = re.sub(r"([,.!?])", r" \1 ", text)
    # 5. 折疊多重空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_phonemes(text: str, use_tone: bool = True) -> Tuple[List[str], str]:
    """
    主入口：中文文字 → phoneme 序列 + 處理後的文字。

    Args:
        text:     原始中文歌詞字串
        use_tone: 是否在韻母附帶聲調（True → "ai3"，False → "ai"）
                  訓練時建議 True；歌唱中聲調與旋律有關，丟掉聲調會降低資訊量

    Returns:
        phs:        list[str]，phoneme tokens，已含開頭 SIL_TOKEN 與每個音節後的邊界
        clean_text: 前處理後的文字（便於 debug / 對齊原始字元位置）

    為什麼回傳 clean_text：
      上層 binarizer 若需要「字元-phoneme 對應」（例如 word-level alignment），
      需要知道實際被處理的文字是哪個版本。

    為什麼用 INITIALS + FINALS_TONE3 兩次呼叫：
      pypinyin 沒有單一 style 同時給聲母 + 帶聲調韻母。INITIALS 給聲母（"n"、"sh"），
      FINALS_TONE3 給韻母帶聲調數字（"i3"、"ao4"）。聲調寫在數字後而非附加號，
      與 NSVB 原版相容。

    為什麼處理「聲母==韻母」的情況：
      pypinyin 對單韻母字（如「啊」）會回傳 INITIALS=""、FINALS_TONE3="a"，
      或對某些字 INITIALS 與 FINALS 重複。我們檢測這個情況只輸出韻母，避免重複 token。
    """
    text = preprocess_text(text)

    if not text:
        # 空字串保險回傳
        return [SIL_TOKEN], text

    # pypinyin 對非中文字元會原樣保留（例如英文字母），這裡只處理中文部分的轉換
    # heteronym=False：多音字只取最常見讀音（如歌詞場景已足夠）
    shengmu = pinyin(text, style=Style.INITIALS, heteronym=False, strict=False)
    yunmu_with_tone = pinyin(
        text, style=Style.FINALS_TONE3, heteronym=False, strict=False
    )
    yunmu_no_tone = pinyin(
        text, style=Style.FINALS, heteronym=False, strict=False
    )

    # pypinyin 三次呼叫的長度應一致（每個輸入字元 → 一個 list）
    assert len(shengmu) == len(yunmu_with_tone) == len(yunmu_no_tone), (
        f"pypinyin output length mismatch: "
        f"{len(shengmu)} / {len(yunmu_with_tone)} / {len(yunmu_no_tone)}"
    )

    phs: List[str] = [SIL_TOKEN]

    for sm_list, ym_t_list, ym_list in zip(shengmu, yunmu_with_tone, yunmu_no_tone):
        sm = sm_list[0]
        ym_tone = ym_t_list[0]
        ym = ym_list[0]

        # 先驗證 ym 是合法 pinyin 韻母字串（純小寫拉丁字母 + ü）。
        # 若不是 → 整個音節是 pypinyin 的 fallback（英文/數字/標點），跳過 phoneme 化。
        # 為什麼這個檢查不可省：pypinyin 對非中文字元會原樣回傳，例如：
        #   "Hello" → INITIALS="" / FINALS="Hello" / FINALS_TONE3="Hello"
        #   "123"   → 同樣回傳原字
        # 若不過濾，後面會誤當韻母加聲調 → phoneme 序列被汙染（Risk: D_z 詞表錯亂）
        if not _is_valid_pinyin_unit(ym):
            # 非中文字元：當作邊界，不產生 phoneme token
            continue

        # 聲母也要驗證：sm 可能是空字串（合法，代表零聲母如「啊」），
        # 或是合法聲母（"n", "sh"），但不該是其他奇怪內容
        if sm and not _is_valid_pinyin_unit(sm):
            continue

        # 處理「沒有聲調」的情況：pypinyin 對輕聲字 ym_tone 等於 ym（無數字結尾）
        # 為了維持「韻母都有聲調數字」的不變條件，補上 5 表示輕聲
        # 為什麼用 5：原 NSVB 慣例，常見中文 NLP 標準（1=陰平 2=陽平 3=上 4=去 5=輕聲）
        if use_tone and ym_tone == ym:
            ym_tone = ym_tone + "5"
        elif not use_tone:
            ym_tone = ym  # 不要聲調

        # 兩種主要 case：
        #   (1) 有聲母 + 有韻母（正常字，如「你 ni3」→ ['n', 'i3']）
        #   (2) 無聲母 + 有韻母（單韻母字，如「啊 a1」→ ['a1']）
        if sm and sm != ym:
            phs.append(sm)
            phs.append(ym_tone)
            phs.append(SIL_TOKEN)
        else:
            phs.append(ym_tone)
            phs.append(SIL_TOKEN)

    return phs, text


def build_phoneme_vocab() -> List[str]:
    """
    回傳 NSVB-ZH 所有可能的 phoneme token 列表（用於建 phoneme_id 詞表）。

    為什麼需要固定詞表：
      D_z 的 phoneme 條件是 discrete ID（argmax PPG）。為了讓 PPG 抽出來的 phoneme
      （Whisper 詞表）與文字 phoneme（pypinyin 詞表）能對齊或互查，需要一個明確的
      ID-to-string 映射。這裡定義文字端的 vocab；PPG 端會在 Phase 0 抽特徵時建另一張，
      兩張表的對應關係留給 binarizer 處理。

    內容：
      - 句界 SIL
      - 23 聲母（含空聲母情況下的單獨韻母由韻母覆蓋）
      - 韻母 × 聲調（5 個聲調）

    為什麼把所有聲調都展開成獨立 token：
      D_z 看到的是 discrete phoneme ID。如果只用韻母（無聲調）共 35 個，會丟掉聲調
      在歌唱中與旋律對應的訊息。展開後雖然 vocab 變大（~180），但對 D_z 不增成本
      （embedding lookup 是 O(1)）。
    """
    initials = [
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h",
        "j", "q", "x",
        "zh", "ch", "sh", "r",
        "z", "c", "s",
        "y", "w",
    ]

    # 韻母單體（與 pypinyin FINALS 輸出對齊）
    finals = [
        "a", "o", "e", "i", "u", "ü",
        "ai", "ei", "ao", "ou",
        "an", "en", "ang", "eng", "ong",
        "ia", "ie", "iao", "iou", "iu", "ian", "in", "iang", "ing", "iong",
        "ua", "uo", "uai", "uei", "ui", "uan", "un", "uang", "ueng",
        "üe", "üan", "ün",
        "er",
        "ix", "iy",  # pypinyin 對「資 zi → zix1」「之 zhi → zhiy1」會用這種韻母
    ]

    tones = ["1", "2", "3", "4", "5"]

    vocab = [SIL_TOKEN]
    vocab.extend(initials)
    for f in finals:
        for t in tones:
            vocab.append(f + t)

    return vocab


if __name__ == "__main__":
    # 自我測試：覆蓋 case 1/2/3、聲調、空字串、混合英數
    samples = [
        "你好世界",                # 標準中文
        "啊我愛你",                 # 含單韻母字（啊、愛）
        "Hello 世界123",            # 中英數混合
        "今天天氣真好！！！",        # 含全形標點
        "",                         # 空字串
        "雨下整夜 我的愛溢出就像雨水",  # 流行歌歌詞片段
    ]

    print(f"Phoneme vocab size = {len(build_phoneme_vocab())}\n")

    for s in samples:
        phs, clean = text_to_phonemes(s, use_tone=True)
        print(f"Input : {s!r}")
        print(f"Clean : {clean!r}")
        print(f"Phs   : {phs}")
        print()
