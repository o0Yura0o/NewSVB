"""
scripts/stage2_ckpts_listening.py
===================================

【這支腳本做什麼】
跨多個 Stage 2 step ckpts 跑 Mode A 推理,把每個 ckpt 對同一批 val 樣本的輸出 wav
並排存到一個資料夾,方便 A/B/.../H 主觀聽測對比。

【為什麼這樣設計】
Stage 2 是 GAN-style 對抗訓練,單一 scalar metric 無法決定「最佳 ckpt」(d_z/d_mel 是相對訊號,
l_adv 絕對值會 saturate)。**主觀聽測才是真正的 ground truth**,且 GAN 訓練常見「best ckpt
不是最後一個」的現象 — 訓得太久反而 over-modification。

【輸出結構】
{out_dir}/
├── {dataset}_{safe_item_id}/
│   ├── _0_orig.wav            原音(.npz wav field,已 dereverb+loudness norm)
│   ├── _1_stage1_recon.wav    Stage 1 only(無 M),測 VAE 重建 baseline
│   ├── step005000.wav         Stage 2 + M from step 5000
│   ├── step015000.wav
│   ├── step030000.wav
│   ├── ...
│   └── step120000.wav
└── ...

下載整個 {out_dir} 後,每個 sample 內檔名排序對應「越往下越多訓練」,直接連著聽
就能感受到 M 隨訓練進度的修飾強度。

【判讀指引】
- _0_orig vs _1_stage1_recon → VAE 重建損失(應該幾乎沒差,Stage 1 訓很好)
- _1_stage1_recon vs step005000 → M 早期(剛過 warmup)的初步修飾
- step005000 → step120000 → M 修飾強度逐步升級
- 找出「最 pro-sounding」「最少 artifact」「咬字 / vibrato 保留最好」的 step,
  那個 ckpt 就是 Phase 3 該用的

【用法】
  python scripts/stage2_ckpts_listening.py \\
      --stage1-ckpt /content/stage1_ckpts/stage1_best.pt \\
      --stage2-ckpt-dir /content/stage2_ckpts \\
      --vocoder-ckpt /content/drive/MyDrive/nsvb_ckpts/1012_hifigan_all_songs_nsf/model_ckpt_steps_1170000.ckpt \\
      --binarized-root /content/drive/MyDrive/NSVB-ZH/data/binarized \\
      --val-split /content/local_binarized/splits/val.txt \\
      --out-dir /content/drive/MyDrive/NSVB-ZH/outputs/stage2_listening \\
      --n-samples 6 \\
      --steps "5000,15000,30000,50000,70000,90000,110000,120000"
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.model.svb_vae_zh import SVBVAEZh
from nsvb.model.m_mapping import ResidualM
from nsvb.backbone.vocoder import HifiGanNSFGenerator
from nsvb.utils.audio_config import SAMPLE_RATE, HOP_SIZE, NUM_MELS, LATENT_DOWN_FACTOR


def interp_f0_unvoiced(f0_np: np.ndarray) -> np.ndarray:
    """vocoder 需要連續 F0 (unvoiced log-space 內插),避免 SineGen 在邊界產生電音 artifact。
    跟 scripts/vocoder_identity_test.py 用同樣的內插策略。
    """
    f0 = f0_np.copy().astype(np.float32)
    uv = f0 == 0
    if uv.all() or not uv.any():
        return f0
    log_f0 = np.log2(f0 + 1e-5)
    idx = np.arange(len(f0))
    log_f0[uv] = np.interp(idx[uv], idx[~uv], log_f0[~uv])
    return np.exp2(log_f0).astype(np.float32)


def pad_to_multiple(arr: np.ndarray, multiple: int, pad_value=0.0):
    """沿 axis 0 補到 multiple 的倍數。FVAE encoder stride=4 → T 必須 % 4 == 0,
    跟 collate_fn 的 padding 邏輯一致。"""
    T = arr.shape[0]
    pad = (-T) % multiple
    if pad == 0:
        return arr
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[0] = (0, pad)
    return np.pad(arr, pad_widths, constant_values=pad_value)


def load_svbvae(stage1_ckpt_path: str, device: str) -> SVBVAEZh:
    """載入 Stage 1 best ckpt 的 SVBVAEZh 整套(encoder + decoder + condition),
    Stage 2 用同一個 backbone(凍結)。"""
    state = torch.load(stage1_ckpt_path, map_location='cpu', weights_only=False)
    cfg = state['config']
    model = SVBVAEZh(
        num_mels=cfg['num_mels'], ppg_dim=cfg['ppg_dim'], spk_emb_dim=cfg['spk_emb_dim'],
        latent_size=cfg['latent_size'], hidden_size=cfg['hidden_size'],
        kernel_size=cfg['fvae_kernel_size'],
        enc_n_layers=cfg['enc_n_layers'], dec_n_layers=cfg['dec_n_layers'],
    )
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    return model


def load_M(stage2_ckpt_path: str, device: str) -> ResidualM:
    """從 Stage 2 step ckpt 載入 ResidualM(只取 M state,其餘的 D_z / D_mel / opt 都跳過)。"""
    state = torch.load(stage2_ckpt_path, map_location='cpu', weights_only=False)
    cfg = state['config']
    m = ResidualM(
        latent_dim=cfg['latent_size'],
        hidden_dim=cfg['m_hidden_dim'],
        kernel_size=cfg['m_kernel_size'],
        num_layers=cfg['m_num_layers'],
        init_delta_scale=cfg['m_init_delta_scale'],
    )
    m.load_state_dict(state['M'], strict=True)
    m.to(device).eval()
    return m


def load_vocoder(vocoder_ckpt_path: str, device: str) -> HifiGanNSFGenerator:
    """載 NSVB 1012 HifiGAN-NSF(同 scripts/vocoder_identity_test.py 的 load_vocoder)。"""
    v_state = torch.load(vocoder_ckpt_path, map_location='cpu', weights_only=False)
    sd = v_state.get('state_dict', v_state)
    if 'model_gen' in sd:
        sd = sd['model_gen']
    g = HifiGanNSFGenerator(c_out=1, num_mels=NUM_MELS, audio_sample_rate=SAMPLE_RATE)
    g.load_state_dict(sd, strict=True)
    g.to(device).eval()
    g.remove_weight_norm()
    return g


def find_npz(item_id: str, binarized_root: Path):
    """找 item_id 對應的 .npz(可能在 m4singer/ 或 vocalverse/)。"""
    for ds in ('m4singer', 'vocalverse'):
        p = binarized_root / ds / f'{item_id}.npz'
        if p.exists():
            return p, ds
    return None, None


@torch.no_grad()
def run_mode_a(svbvae: SVBVAEZh, M, vocoder, npz_path: Path, device: str,
               skip_vocoder: bool = False):
    """單筆 Mode A 推理:.npz → mel/ppg/f0/spk → φ → (M) → θ → vocoder → wav。
    M=None 時走 Stage 1 only baseline(沒有 M 的純重建)。

    為什麼用 m_q 而非採樣 z:跟 stage2 _encode_and_downsample 一致,deterministic
    inference 較穩;採樣會引入 noise,不利對比聽測。

    Args:
        skip_vocoder: True 時跳過 vocoder forward(vocoder 可傳 None)。給 stage2_mel_eval
                      在全 test set 上跑(vocoder 是 bottleneck;不要 wav 時可省 ~85% 時間)。

    Returns:
        wav   : np.ndarray [T_audio]  或 None (skip_vocoder=True)
        mel_out: np.ndarray [T_orig, NUM_MELS]  decoder 輸出的 mel(送 vocoder 前)
                 給 stage2_mel_eval 共用,不重跑 inference
    """
    d = np.load(npz_path, allow_pickle=True)
    mel_np = d['mel'].astype(np.float32)
    ppg_np = d['ppg'].astype(np.float32)
    f0_np = d['f0'].astype(np.float32)
    spk_np = d['spk_emb'].astype(np.float32)
    T_orig = mel_np.shape[0]

    # T 補到 4 倍數(FVAE stride=4 要求)
    mel_pad = pad_to_multiple(mel_np, LATENT_DOWN_FACTOR, pad_value=-10.0)
    ppg_pad = pad_to_multiple(ppg_np, LATENT_DOWN_FACTOR, pad_value=0.0)
    f0_pad = pad_to_multiple(f0_np, LATENT_DOWN_FACTOR, pad_value=0.0)

    mel = torch.from_numpy(mel_pad).unsqueeze(0).to(device)
    ppg = torch.from_numpy(ppg_pad).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0_pad).unsqueeze(0).to(device)
    spk = torch.from_numpy(spk_np).unsqueeze(0).to(device)
    mel_mask = torch.zeros_like(f0_t)
    mel_mask[:, :T_orig] = 1.0

    # 走 fvae 內部(這樣能在 encoder/decoder 中間插 M)
    g = svbvae.condition(ppg, f0_t, spk)             # [1, gin, T]
    mel_chfirst = mel.transpose(1, 2)                # [1, NUM_MELS, T]
    mask_exp = mel_mask.unsqueeze(1)                 # [1, 1, T]
    g_sqz = svbvae.fvae.g_pre_net(g)
    z_q, m_q, logs_q, _ = svbvae.fvae.encoder(mel_chfirst, mask_exp, g_sqz)
    # 走 m_q deterministic
    z_in = m_q if M is None else M(m_q)
    mel_out_chfirst = svbvae.fvae.decoder(z_in, mask_exp, g)   # [1, NUM_MELS, T_pad]
    # 砍回 T_orig
    mel_out_chfirst = mel_out_chfirst[:, :, :T_orig]
    # mel 在 eval 用(time-major,跟 .npz['mel'] 同 layout)
    mel_out_np = mel_out_chfirst.squeeze(0).transpose(0, 1).cpu().numpy()  # [T_orig, NUM_MELS]

    if skip_vocoder:
        return None, mel_out_np

    # vocoder 端:F0 unvoiced log-space 內插
    f0_interp = interp_f0_unvoiced(f0_np[:T_orig])
    f0_v = torch.from_numpy(f0_interp).unsqueeze(0).to(device)
    wav = vocoder(mel_out_chfirst, f0_v).squeeze(1).squeeze(0).cpu().numpy()
    return wav, mel_out_np


def main():
    parser = argparse.ArgumentParser(
        description='Batch Mode A inference across Stage 2 ckpts for subjective listening test',
    )
    parser.add_argument('--stage1-ckpt', required=True,
                        help='Path to stage1_best.pt(SVBVAEZh backbone)')
    parser.add_argument('--stage2-ckpt-dir', required=True,
                        help='Folder containing stage2_step*.pt files')
    parser.add_argument('--vocoder-ckpt', default=None,
                        help='Path to NSVB 1012 HifiGAN ckpt(--skip-vocoder 啟用時可省略)')
    parser.add_argument('--binarized-root', required=True,
                        help='Folder with m4singer/ and vocalverse/ subdirs of .npz')
    parser.add_argument('--val-split', required=True,
                        help='Path to splits/val.txt(make_splits.py 輸出)')
    parser.add_argument('--out-dir', required=True,
                        help='輸出 wav 的根目錄(會建 per-sample 子資料夾)')
    parser.add_argument('--n-samples', type=int, default=6,
                        help='取幾個 val 樣本聽(預設 6,bias 偏 VocalVerse)')
    parser.add_argument('--steps', default='5000,15000,30000,50000,70000,90000,110000,120000',
                        help='comma-separated step numbers 對應 stage2_step{N}.pt')
    parser.add_argument('--seed', type=int, default=42,
                        help='樣本選擇 random seed(同 seed → 跨 ckpt 樣本完全一致)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dump-mel', action='store_true',
                        help='同時 dump 每組 mel 為 .npy (orig/stage1_recon/step{N}.mel.npy), '
                             '供 scripts/stage2_mel_eval.py 直接讀,不用重跑 inference')
    parser.add_argument('--skip-vocoder', action='store_true',
                        help='跳過 vocoder forward(節省 85% 時間,但只能生 mel 不生 wav)。'
                             '配 --dump-mel 用於全 test set 跑 eval。--vocoder-ckpt 可省略。')
    parser.add_argument('--all-samples', action='store_true',
                        help='忽略 --n-samples,跑 val_split 內全部樣本。'
                             '搭配 --skip-vocoder --dump-mel 用於全 test set eval。')
    args = parser.parse_args()

    if not args.skip_vocoder and not args.vocoder_ckpt:
        parser.error('--vocoder-ckpt 必填(除非 --skip-vocoder)')

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binarized_root = Path(args.binarized_root)

    # 抽樣:bias 偏 VocalVerse(amateur 是聽測重點);--all-samples 時改取全部
    val_list = [s.strip() for s in Path(args.val_split).read_text().splitlines() if s.strip()]
    val_vv = [v for v in val_list if '__c' in v]
    val_m4 = [v for v in val_list if '#' in v and v not in val_vv]

    random.seed(args.seed)
    if args.all_samples:
        picks = val_vv + val_m4
        random.shuffle(picks)  # deterministic shuffle 給 eval 跑順序
        print(f'[picks] ALL {len(picks)} samples (M4={len(val_m4)}, VV={len(val_vv)}, '
              f'seed={args.seed} for shuffle)')
    else:
        n_vv = max(args.n_samples - 1, 1)
        picks = random.sample(val_vv, min(n_vv, len(val_vv)))
        if val_m4 and len(picks) < args.n_samples:
            picks += random.sample(val_m4, args.n_samples - len(picks))
        print(f'[picks] {len(picks)} samples (seed={args.seed}):')
        for p in picks:
            print(f'  - {p}')

    # 載入共用模型(只載一次);--skip-vocoder 時跳過 vocoder 載入
    print(f'\n[load] SVBVAEZh backbone from {args.stage1_ckpt}')
    svbvae = load_svbvae(args.stage1_ckpt, device)
    if args.skip_vocoder:
        vocoder = None
        print(f'[load] vocoder SKIPPED (--skip-vocoder)')
    else:
        print(f'[load] vocoder from {args.vocoder_ckpt}')
        vocoder = load_vocoder(args.vocoder_ckpt, device)

    step_list = [int(s.strip()) for s in args.steps.split(',') if s.strip()]
    print(f'\n[steps] {len(step_list)} stage2 ckpts to evaluate: {step_list}')

    # 主迴圈
    for item_id in picks:
        npz_path, ds = find_npz(item_id, binarized_root)
        if npz_path is None:
            print(f'\n  ✗ {item_id}: .npz not found under {binarized_root}/{{m4singer,vocalverse}}/')
            continue
        safe = item_id.replace('#', '_').replace('__', '-')
        sample_dir = out_dir / f'{ds}_{safe}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n[sample] {ds}/{item_id}  →  {sample_dir.name}/')

        # _0_orig wav + mel(mel 取自 .npz['mel'],即 binarize 的 GT)
        # skip_vocoder 時不讀 wav 欄位(.npz 內 wav 是 ~5MB 大頭,跳過省 IO + Drive FUSE)
        with np.load(npz_path, allow_pickle=True) as d:
            mel_orig = d['mel'].astype(np.float32)            # [T, NUM_MELS]
            f0_orig = d['f0'].astype(np.float32)
            wav_orig = None if args.skip_vocoder else d['wav'].astype(np.float32)
        if not args.skip_vocoder:
            sf.write(str(sample_dir / '_0_orig.wav'), wav_orig, SAMPLE_RATE)
            print(f'  _0_orig.wav  (T_audio={len(wav_orig)} ≈ {len(wav_orig)/SAMPLE_RATE:.2f}s)')
        else:
            print(f'  (skip-vocoder, T_mel={mel_orig.shape[0]} ≈ {mel_orig.shape[0]/172:.2f}s)')
        if args.dump_mel:
            np.save(str(sample_dir / '_0_orig.mel.npy'), mel_orig)
            np.save(str(sample_dir / 'f0.npy'), f0_orig)

        # _1_stage1_recon(M=None)
        wav_s1, mel_s1 = run_mode_a(svbvae, None, vocoder, npz_path, device,
                                     skip_vocoder=args.skip_vocoder)
        if not args.skip_vocoder:
            sf.write(str(sample_dir / '_1_stage1_recon.wav'), wav_s1, SAMPLE_RATE)
            print(f'  _1_stage1_recon.wav')
        if args.dump_mel:
            np.save(str(sample_dir / '_1_stage1_recon.mel.npy'), mel_s1)

        # Stage 2 each step
        for step in step_list:
            ckpt_path = Path(args.stage2_ckpt_dir) / f'stage2_step{step}.pt'
            if not ckpt_path.exists():
                print(f'  ✗ step{step}: ckpt not at {ckpt_path}')
                continue
            M = load_M(str(ckpt_path), device)
            wav_s2, mel_s2 = run_mode_a(svbvae, M, vocoder, npz_path, device,
                                         skip_vocoder=args.skip_vocoder)
            if not args.skip_vocoder:
                sf.write(str(sample_dir / f'step{step:06d}.wav'), wav_s2, SAMPLE_RATE)
                print(f'  step{step:06d}.wav')
            if args.dump_mel:
                np.save(str(sample_dir / f'step{step:06d}.mel.npy'), mel_s2)
            del M  # 釋放 GPU 記憶體

    print(f'\n✅ done. 下載 {out_dir}/ 來聽。')
    print('   每個 sample 子資料夾內檔名 sorted 即時間順序:')
    print('     _0_orig < _1_stage1_recon < step005000 < ... < step120000')


if __name__ == '__main__':
    main()
