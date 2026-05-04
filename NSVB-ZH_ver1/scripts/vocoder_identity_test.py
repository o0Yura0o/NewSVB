# -*- coding: utf-8 -*-
"""
Phase 0 vocoder identity test.

Question
--------
The pretrained 1012_hifigan_all_songs_nsf vocoder was trained on English
singing (PopBuTFy). Does it still reconstruct Chinese singing (M4Singer /
OpenSinger) faithfully when fed the ground-truth mel? If not, any Stage 2
improvement will be invisible at listening time.

Method
------
For each dataset, pick K clips (default 20 each).
For every clip:
  1. Load ground-truth waveform, compute mel with the *exact same*
     mel extraction parameters as the vocoder was trained with
     (hop=128, fft=512, win=512, sr=22050, fmin=50, fmax=11025).
  2. Feed mel + F0 through HifiGAN NSF to get y_recon.
  3. Recompute mel from y_recon.
  4. Compute:
        mel SSIM   (structural similarity of mel spectrograms)
        F0 RMSE    (Hz, computed by DIO + StoneMask)
     Optionally dump wav pairs for blind listening.

Thresholds
----------
Pass if:   mel SSIM >= 0.90  AND  F0 RMSE <= 10 Hz
Warning:   mel SSIM 0.85-0.90 or F0 RMSE 10-20 Hz   (marginal; test on humans)
Fail:      mel SSIM < 0.85  OR  F0 RMSE > 20 Hz
           -> retrain or fine-tune the vocoder on Chinese singing before Stage 2.

Usage
-----
python scripts/vocoder_identity_test.py \\
    --vocoder_ckpt checkpoints/0109_hifigan_bigpopcs_hop128/model_ckpt_steps_1512000.ckpt \\
    --wav_dirs m4=data/raw/m4singer open=data/raw/opensinger \\
    --n_per_dir 20 \\
    --out_dir outputs/phase0_vocoder
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile


def parse_wav_dirs(spec: list) -> dict:
    """Parse 'name=path' pairs from CLI."""
    out = {}
    for s in spec:
        name, path = s.split('=', 1)
        out[name] = path
    return out


def find_wavs(root: str, n: int, seed: int = 0, exts=('.wav',)) -> list:
    root = Path(root)
    candidates = []
    for ext in exts:
        candidates.extend(root.rglob(f'*{ext}'))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return [str(p) for p in candidates[:n]]


def wav_to_mel(wav: np.ndarray, sr: int, hparams: dict) -> np.ndarray:
    """Compute log-mel with matching hyperparameters.

    Uses the NSVB-style mel extraction. Change this if your vocoder was
    trained with librosa or torchaudio with different conventions.
    """
    # Delegate to the repo's own mel function for bit-exact compatibility.
    from utils.audio import librosa_wav2spec
    spec = librosa_wav2spec(
        wav, fft_size=hparams.get('fft_size', 512),
        hop_size=hparams.get('hop_size', 128),
        win_length=hparams.get('win_size', 512),
        num_mels=hparams.get('audio_num_mel_bins', 80),
        fmin=hparams.get('fmin', 50),
        fmax=hparams.get('fmax', 11025),
        sample_rate=hparams.get('audio_sample_rate', 22050),
    )
    return spec['mel']


def extract_f0(wav: np.ndarray, sr: int, hop_size: int = 128) -> np.ndarray:
    """DIO + StoneMask F0, frame-aligned to hop_size."""
    import pyworld as pw
    f0, t = pw.dio(wav.astype(np.float64), sr,
                   frame_period=1000.0 * hop_size / sr)
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sr)
    return f0.astype(np.float32)


def mel_ssim(mel_a: np.ndarray, mel_b: np.ndarray) -> float:
    """Structural similarity between two mel spectrograms.

    Trims to common length, treats mel as a 2D grayscale image.
    """
    from skimage.metrics import structural_similarity as ssim
    T = min(mel_a.shape[0], mel_b.shape[0])
    a = mel_a[:T]; b = mel_b[:T]
    data_range = max(a.max() - a.min(), b.max() - b.min(), 1e-6)
    return float(ssim(a, b, data_range=data_range))


def f0_rmse(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Voiced-only F0 RMSE in Hz."""
    T = min(len(f0_a), len(f0_b))
    a = f0_a[:T]; b = f0_b[:T]
    mask = (a > 0) & (b > 0)
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def load_vocoder(ckpt_path: str, device: torch.device):
    """Load HifiGAN NSF vocoder from NSVB-style checkpoint."""
    from modules.hifigan.hifigan import HifiGanGenerator
    from utils.hparams import set_hparams, hparams as HP
    # Assumes hparams.yaml sits next to the ckpt
    cfg_yaml = Path(ckpt_path).parent / 'config.yaml'
    set_hparams(str(cfg_yaml))
    g = HifiGanGenerator(HP)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('model_gen.', '', 1) if k.startswith('model_gen.') else k: v
          for k, v in sd.items()}
    g.load_state_dict(sd, strict=False)
    g.eval().to(device)
    return g, dict(HP)


@torch.no_grad()
def vocode(g, mel: np.ndarray, f0: np.ndarray, device) -> np.ndarray:
    """mel (T, n_mels), f0 (T,) -> wav (samples,)"""
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)   # (1, T, M)
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(device)     # (1, T)
    wav = g(mel_t.transpose(1, 2), f0=f0_t).squeeze().cpu().numpy()
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocoder_ckpt', required=True)
    ap.add_argument('--wav_dirs', nargs='+', required=True,
                    help='name=path pairs, e.g. m4=/path/to/m4 open=/path/to/open')
    ap.add_argument('--n_per_dir', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='outputs/phase0_vocoder')
    ap.add_argument('--save_wavs', action='store_true',
                    help='Dump original and reconstructed wavs for listening.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[vocoder-test] Loading vocoder: {args.vocoder_ckpt}")
    g, hp = load_vocoder(args.vocoder_ckpt, device)
    sr = hp.get('audio_sample_rate', 22050)

    dir_map = parse_wav_dirs(args.wav_dirs)
    all_results = {}
    summary = {}

    for name, root in dir_map.items():
        print(f"\n[vocoder-test] Dataset={name}  root={root}")
        wavs = find_wavs(root, args.n_per_dir, seed=args.seed)
        per_clip = []
        for idx, wp in enumerate(wavs):
            sr_src, wav_i16 = wavfile.read(wp)
            if wav_i16.ndim > 1:
                wav_i16 = wav_i16[:, 0]
            wav = wav_i16.astype(np.float32) / 32768.0
            if sr_src != sr:
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                wav = torchaudio.functional.resample(wav_t, sr_src, sr).squeeze(0).numpy()

            mel_gt = wav_to_mel(wav, sr, hp)
            f0_gt = extract_f0(wav, sr, hop_size=hp.get('hop_size', 128))
            # Align lengths
            T = min(mel_gt.shape[0], len(f0_gt))
            mel_gt = mel_gt[:T]; f0_gt = f0_gt[:T]

            wav_recon = vocode(g, mel_gt, f0_gt, device)
            mel_recon = wav_to_mel(wav_recon, sr, hp)
            f0_recon = extract_f0(wav_recon, sr, hop_size=hp.get('hop_size', 128))

            ssim_v = mel_ssim(mel_gt, mel_recon)
            rmse_v = f0_rmse(f0_gt, f0_recon)
            per_clip.append({'path': wp, 'mel_ssim': ssim_v, 'f0_rmse_hz': rmse_v})
            print(f"  [{idx+1:02d}/{len(wavs)}]  SSIM={ssim_v:.3f}  F0_RMSE={rmse_v:.2f} Hz  {Path(wp).name}")

            if args.save_wavs:
                stem = Path(wp).stem
                out_sub = Path(args.out_dir) / name
                out_sub.mkdir(parents=True, exist_ok=True)
                wavfile.write(out_sub / f'{stem}_orig.wav', sr,
                              (wav * 32767).astype(np.int16))
                wavfile.write(out_sub / f'{stem}_recon.wav', sr,
                              (wav_recon * 32767).astype(np.int16))

        ssim_arr = np.array([c['mel_ssim'] for c in per_clip])
        rmse_arr = np.array([c['f0_rmse_hz'] for c in per_clip if not np.isnan(c['f0_rmse_hz'])])
        summary[name] = {
            'n_clips': len(per_clip),
            'mel_ssim_mean': float(ssim_arr.mean()),
            'mel_ssim_min': float(ssim_arr.min()),
            'f0_rmse_mean': float(rmse_arr.mean()) if len(rmse_arr) else float('nan'),
            'f0_rmse_max': float(rmse_arr.max()) if len(rmse_arr) else float('nan'),
        }
        all_results[name] = per_clip

    # Decision
    def verdict(s):
        if s['mel_ssim_mean'] >= 0.90 and s['f0_rmse_mean'] <= 10:
            return 'PASS'
        if s['mel_ssim_mean'] >= 0.85 and s['f0_rmse_mean'] <= 20:
            return 'MARGINAL (listen carefully)'
        return 'FAIL (fine-tune vocoder on Chinese singing before Stage 2)'

    verdicts = {name: verdict(s) for name, s in summary.items()}

    out = {
        'vocoder_ckpt': args.vocoder_ckpt,
        'summary': summary,
        'verdicts': verdicts,
        'per_clip': all_results,
    }
    out_json = Path(args.out_dir) / 'vocoder_report.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("VOCODER IDENTITY TEST RESULT")
    print("=" * 60)
    for name, v in verdicts.items():
        s = summary[name]
        print(f"{name:12s}  SSIM={s['mel_ssim_mean']:.3f}  "
              f"F0_RMSE={s['f0_rmse_mean']:.2f} Hz  -> {v}")
    print(f"\nFull report: {out_json}")


if __name__ == '__main__':
    main()
