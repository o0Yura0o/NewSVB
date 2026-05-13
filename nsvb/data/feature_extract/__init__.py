"""
nsvb/data/feature_extract/
============================

Phase 0 特徵抽取管線。每個檔案專責一種特徵：

  f0_torchcrepe.py    F0 (Hz) per frame，輸出對齊 mel frame rate
  ppg_whisper.py      Whisper hidden state（continuous content）+ 對齊 mel
  spk_resemblyzer.py  Speaker embedding（固定 256 維 / utterance）

這些抽取器共用 nsvb/utils/audio_config.py 的常數，確保所有特徵 frame-rate 對齊。
"""
