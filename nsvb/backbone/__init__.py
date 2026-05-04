"""
nsvb/backbone/
================

【這個 package 做什麼】
從 NSVB 原 repo 移植過來的「骨幹」神經網路模組，全部 self-contained
（不依賴 NSVB 的 utils.hparams 全域 state、不依賴 FastSpeech2 文字管線）。
我們的 Stage 1 / Stage 2 model 直接拿這些 module 當 building block 使用。

【為什麼移植而非 git submodule】
- NSVB repo 有大量我們不用的東西（asr/ glow/ tts/ a2p alignment 等）
- 直接 import 會被迫拖入 hparams 全域 state，與我們乾淨的 audio_config 衝突
- 移植時順便把參數從 hparams 改成 explicit constructor args，未來修改不會
  跨檔污染

【目錄結構】
    fvae.py                    ← CVAE 骨幹（WN / FVAEEncoder / FVAEDecoder / FVAE）
    multi_window_disc.py       ← Multi-window mel 判別器（D_mel）
    vocoder/
        parallel_wavegan.py    ← Vocoder generator (NSVB 1012_hifigan_all_songs_nsf 對應架構)

【為什麼把 vocoder 跟其他 backbone 區隔】
- vocoder 推理時是 frozen pretrained，與訓練端解耦
- 訓練時不需要 import vocoder（避免拖長 import 時間）
- 推理 / vocoder identity test 才會 import
"""