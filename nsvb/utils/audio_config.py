"""
nsvb/utils/audio_config.py
============================

【這支檔案做什麼】
集中管理 NSVB-ZH 全專案共用的音訊參數常數：取樣率、mel 參數、frame rate。
所有特徵抽取器（F0、PPG、mel、register）與訓練配置都從這裡 import。

【為什麼需要它】
特徵之間必須 frame-rate 一致，否則 D_z 的 register/phoneme 條件對不齊 latent z 的 frame：
    mel:        sample_rate / hop_size = 22050 / 256 = 86.13 fps
    F0:         必須同 fps 才能 frame-wise 對應
    PPG:        Whisper 原生 50 fps，需 resample 到 mel fps
    register:   F0 → soft bucket，自然繼承 F0 fps
若各模組各寫一份常數，未來改參數時極易遺漏某處導致對齊偏移 → 訓練出現 ghost 特徵。
集中一份做 single source of truth。

【為什麼選 22050 / 128 / 512 / 80 / 0 / 8000】
這套參數**完全對齊 NSVB 原作者提供的 vocoder ckpt 實際訓練設定**
（`checkpoints/1012_hifigan_all_songs_nsf/config.yaml`，
  訓練自 100+ 小時中英文混合歌聲）：

  sample_rate=22050  hop=128  fft=512  win=512  num_mels=80  fmin=0  fmax=8000

- **22050 Hz**：中文歌聲頻寬主要 < 8 kHz，22050 Nyquist=11025 已涵蓋
- **hop_size 128**（~5.8 ms frame, 172.27 fps）：歌聲 vibrato 約 5-7 Hz，
  vibrato 一週期 ~150 ms 對應 ~26 frame，足以保留細節
- **fft_size 512 / win 512**：~23 ms 視窗，足以解析最低 F0（~65 Hz, C2）
  的 1.5 週期；窄 fft 換來時間解析度高，符合歌聲快速 portamento 特性
- **fmin=0, fmax=8000**：與 ckpt 訓練設定一致；雖然 8 kHz 損失高頻 sibilance，
  但 mel basis 必須匹配 vocoder 訓練時看到的分布

【為什麼必須完全對齊 ckpt config 而非 yaml 預設】
- NSVB repo 有兩份 yaml：`egs/egs_bases/singing/base.yaml` (fmin=50, fmax=11025)
  與 `checkpoints/1012_hifigan_all_songs_nsf/config.yaml` (fmin=0, fmax=8000)
- **以 ckpt config 為準**：vocoder 訓練時看到的 mel 分布是後者；前者是「acoustic model
  訓練 yaml」，與 vocoder 訓練分屬不同 pipeline
- 任一 mel param 不同 → mel basis 矩陣不同 → vocoder 重建品質崩壞

【Vocoder 介面：(mel, F0) → wav】
- 1012 ckpt 的 generator config: `use_pitch_embed: true`（F0 內嵌做為 conditioner）
- 注意 `use_nsf: false`（不是真正 source-filter；只是用 pitch embedding）
- 架構實際是 ParallelWaveGAN-style（layers=30, residual=64, stacks=3,
  upsample [2,4,4,4]），雖然資料夾名稱有 "hifigan"
- 推理 / vocoder identity test 都要把 F0 一併餵進去

【未來若要升級到 BigVGAN-v2】
- 改 sample_rate 24000、hop=256、fft=1024、num_mels 100、fmax=12000；
  屆時只需改這支檔案的常數，所有下游模組自動跟上；但要重新訓 vocoder 或找新 ckpt
"""

# ── 取樣率 ─────────────────────────────────────────────────
# 為什麼 22050：與 NSVB 原版 HifiGAN-NSF ckpt 訓練設定一致
SAMPLE_RATE = 22050

# ── STFT / Mel ────────────────────────────────────────────
# 為什麼 hop=128, fft=512, win=512：完全對齊
# checkpoints/1012_hifigan_all_songs_nsf/config.yaml，讓我們能直接用作者的 ckpt
HOP_SIZE = 128          # frame shift (samples), ~5.8 ms / frame
FFT_SIZE = 512          # FFT bins, ~23 ms 窗
WIN_SIZE = 512          # window length (samples)
NUM_MELS = 80           # mel band 數
# 為什麼 fmin=50, fmax=11025：對齊 NSVB inference 時實際使用的 mel 公式
# （NSVB egs/egs_bases/singing/base.yaml + tts/base.yaml）。
#
# 【重要陷阱記錄】
# vocoder ckpt 1012_hifigan_all_songs_nsf/config.yaml 標 fmin=0, fmax=8000，
# 看起來像是 vocoder 訓練時的 mel 設定，但實際上：
#   - NSVB inference path 載入 acoustic model config（singing/base.yaml）做為 hparams
#   - hparams['fmin']=50, hparams['fmax']=11025 主導 mel 計算
#   - vocoder 訓練時看到的 mel = inference 時看到的 mel = fmin=50 / fmax=11025
# 我們先前對齊 ckpt config (fmin=0, fmax=8000) → mel 分布偏移 → 電音
# 改成 inference 實際用的 fmin=50, fmax=11025 → mel 分布對齊 → 應該 clean
MEL_FMIN = 50
MEL_FMAX = 11025

# Mel log 計算的數值穩定下限
# 為什麼 1e-10：對齊 NSVB vocoder 訓練實際使用的 mel 公式
# （NSVB/data_gen/tts/data_gen_utils.py:process_utterance + pwg.py 的
#  wav2spec_eps 預設 1e-10），配合 log10：silent 段 mel ≈ log10(1e-10) = -10
# 注意：先前一度看到 modules/hifigan/mel_utils.py 用 1e-5 + ln，但那不是訓練
# vocoder 時實際走的程式碼路徑——實際訓練走 process_utterance（log10 + 1e-10）
MEL_EPS = 1e-10

# 響度正規化目標（LUFS, ITU BS.1770）
# 為什麼 -22：NSVB 原版預設；接近串流平台（Spotify -14, YouTube -14）下方一格，
#            保留動態空間給歌唱強弱對比
LOUDNESS_TARGET_LUFS = -22.0

# ── 衍生量（read-only，由上面參數決定）─────────────────────
FRAME_RATE_HZ = SAMPLE_RATE / HOP_SIZE   # 86.13 fps
FRAME_DURATION_MS = 1000.0 / FRAME_RATE_HZ  # ~11.6 ms

# ── F0 抽取（torchcrepe）──────────────────────────────────
# 為什麼 fmin=50, fmax=1400：人聲 F0 範圍
#   - 男低音底 ~ 65 Hz (C2)，留 50 為下限緩衝
#   - 流行女歌手 belt 高潮段常觸碰 C6 (1047 Hz) ~ E6 (1319 Hz)；
#     fmax=1400 覆蓋到 F6 (1397 Hz)，遠超流行歌 tessitura 上限
#   - **為什麼不更高**（如 1800/2000）：CREPE bin 範圍越大、false positive 越多；
#     1400 是「能容納所有現實高音 + 不引入過多噪聲」的甜蜜點
#   - **過往陷阱**：先前設 1100 會在 D6 (1175 Hz) 處被 CREPE 截到 octave 錯誤
#     （D5=587），給 D_z 一個「胸聲 register 條件」對「實際頭聲 mel」的錯誤對齊
#     訊號——比直接 unvoiced 還糟糕
F0_FMIN = 50.0
F0_FMAX = 1400.0
# 為什麼用 'full' model：~80MB，比 'tiny' 慢但對歌聲（vibrato、glide）準確度
#                      明顯較好；Phase 0 一次性抽取，速度不重要
CREPE_MODEL = "full"
# 為什麼 viterbi=False：歌聲有快速半音 portamento，viterbi smoothing 會把細節
#                     拉平；我們寧可保留原始 F0 細節（vibrato 進入 z，bucket 條件
#                     再做粗化）
CREPE_VITERBI = False
# 信心度低於此值的 frame 視為 unvoiced（F0 設為 0）
# 為什麼 0.21：torchcrepe 對歌聲 voicing 邊界的經驗值；過高會把氣聲段誤判 unvoiced，
#              過低會把無聲段誤判 voiced
CREPE_CONFIDENCE_THRESHOLD = 0.21

# ── Whisper PPG 抽取 ──────────────────────────────────────
# 為什麼 large-v3：中文 ASR 表現最佳，hidden state phonetic 細節豐富
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
# 為什麼取 layer 8：Whisper-large-v3 共 32 層，層 6-12 為「phonetic」層
#                  （早期更接近聲學特徵，晚期更接近語意 token）
#                  layer 8 經驗上對 cross-speaker phoneme 表徵最穩
WHISPER_HIDDEN_LAYER = 8
# Whisper 內部固定 16 kHz、30 秒視窗、50 fps 輸出
WHISPER_INPUT_SR = 16000
WHISPER_FRAME_RATE_HZ = 50.0
# 為什麼 1280：whisper-large-v3 encoder hidden state 維度（model.config.d_model）
#              下游 binarizer / cluster_ppg 需要這個常數來預配置陣列形狀
WHISPER_HIDDEN_DIM = 1280

# ── Resemblyzer Speaker Embedding ─────────────────────────
# Resemblyzer 內部 16 kHz，輸出固定 256 維
RESEMBLYZER_INPUT_SR = 16000
RESEMBLYZER_EMBED_DIM = 256

# ── VAE / Latent ──────────────────────────────────────────
# 為什麼 latent_size 128 / down_factor 4：對齊 NSVB 1030_vae_mle ckpt
#   - NSVB FVAE 真實設定：strides=[4]（從 ckpt encoder.pre_net.0 weight (192,80,8) 推得
#     kernel=8, stride=4；config.yaml 的 strides 欄是 None 但 ckpt weight 為準）
#   - down=4：mel 172.27 fps → latent ~43 fps，~23 ms per latent frame
#     雖然比 hop=256 / down=4 給的 46ms 細，但 23 ms 仍 << 中文音素平均 80-150 ms，
#     z 仍夠粗化；且維持與 NSVB 一致是 transfer learning 的前提
#   - latent_size=128：每個 latent frame 128 維，與 NSVB 一致
LATENT_SIZE = 128
LATENT_DOWN_FACTOR = 4
LATENT_FRAME_RATE_HZ = FRAME_RATE_HZ / LATENT_DOWN_FACTOR  # ~43.07 fps

# ── 訓練 batch 內樣本長度上限 ──────────────────────────────
# 為什麼 1500：
#   - VocalVerse binarize 切 5s chunks（~860 frames）→ 1500 容納無需 random crop
#   - M4Singer 5-sec snippets（~860 frames，少數 9s outlier ~1500 frames）→ 1500 涵蓋
#   - latent T_z = 1500 / 4 = 375 ≥ PatchNCE num_patches=128 ✓
#   - 對齊 NSVB 設計「每樣本作為完整 phrase 訓練單位」哲學
# 為什麼不更高（如 NSVB 原版 2400）：M4Singer 最長 8.8s ≈ 1500 frames，1500 已涵蓋
#   所有 M4 snippets；用 2400 多 60% VRAM/算力卻無實質訊息增加
DEFAULT_MAX_FRAMES = 1500
