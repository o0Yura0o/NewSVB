# -*- coding: utf-8 -*-
"""
NSVB-ZH inference.

Three modes correspond to the ablation described in rebuild_checklist.md:

Mode A (default, end-to-end):
    audio_a -> encoder -> z_a -> M -> z_p' -> decoder -> mel_p' -> vocoder -> audio_p'
    Trained-inference matched. F0 contour is amateur's.

Mode B (pro F0 ref):
    audio_a -> encoder -> z_a -> M -> z_p' -> decoder(z_p', f0=f0_REF) -> vocoder
    Requires a reference F0 contour (typically from an aligned pro performance
    or from an F0 editor). Tests whether M improved latent quality indepen-
    dent of F0 conditioning.

Mode C (z_a replay with pro decoder only):
    audio_a -> encoder -> z_a -> decoder(z_a, f0=f0_REF) -> vocoder
    Baseline that skips M entirely. Useful to attribute quality gains to M
    vs to F0 contour substitution.

Usage
-----
python inference/nsvb_zh_inference.py \\
    --stage2_ckpt checkpoints/nsvb_zh_stage2/model_best.ckpt \\
    --vocoder_ckpt checkpoints/0109_hifigan_bigpopcs_hop128/model_best.ckpt \\
    --config egs/datasets/audio/M4OpenSinger/stage2_mapping.yaml \\
    --mode A \\
    --src_wav path/to/amateur.wav \\
    --out_wav out/enhanced.wav
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile


def load_task_and_vocoder(config_path: str, stage2_ckpt: str,
                          vocoder_ckpt: str, device: torch.device):
    from utils.hparams import set_hparams, hparams
    set_hparams(config_path)

    from tasks.singing.nsvb_zh_task import NsvbZhTask
    task = NsvbZhTask()
    task.build_model()
    state = torch.load(stage2_ckpt, map_location='cpu')
    sd = state.get('state_dict', state)
    sd = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
          for k, v in sd.items()}
    missing, unexpected = task.load_state_dict(sd, strict=False)
    print(f"[infer] loaded stage2 ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    task.eval().to(device)

    from modules.hifigan.hifigan import HifiGanGenerator
    g = HifiGanGenerator(hparams)
    vstate = torch.load(vocoder_ckpt, map_location='cpu')
    vsd = vstate.get('state_dict', vstate)
    vsd = {k.replace('model_gen.', '', 1) if k.startswith('model_gen.') else k: v
           for k, v in vsd.items()}
    g.load_state_dict(vsd, strict=False)
    g.eval().to(device)

    return task, g, dict(hparams)


def wav_to_mel_f0(wav: np.ndarray, hp: dict):
    from utils.audio import librosa_wav2spec
    import pyworld as pw
    sr = hp['audio_sample_rate']
    hop = hp['hop_size']
    spec = librosa_wav2spec(
        wav, fft_size=hp['fft_size'], hop_size=hop,
        win_length=hp['win_size'], num_mels=hp['audio_num_mel_bins'],
        fmin=hp['fmin'], fmax=hp['fmax'], sample_rate=sr,
    )
    mel = spec['mel']                    # (T, n_mels)
    f0, t = pw.dio(wav.astype(np.float64), sr,
                   frame_period=1000.0 * hop / sr)
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sr)
    T = min(mel.shape[0], len(f0))
    return mel[:T], f0[:T].astype(np.float32)


@torch.no_grad()
def infer(task, vocoder, hp: dict, mode: str,
          src_wav: np.ndarray, ref_f0: np.ndarray = None,
          device: torch.device = 'cpu') -> np.ndarray:
    mel, f0_src = wav_to_mel_f0(src_wav, hp)
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)        # (1, T, M)
    f0_src_t = torch.from_numpy(f0_src).float().unsqueeze(0).to(device)  # (1, T)

    z = task.run_encoder(task.model, {
        'mels': mel_t, 'f0': f0_src_t,
    })['z']                              # (1, C, T)

    if mode.upper() == 'A':
        z_out = task.m(z)
        f0_use = f0_src_t
    elif mode.upper() == 'B':
        if ref_f0 is None:
            raise ValueError("Mode B requires --ref_f0_wav (pro reference).")
        z_out = task.m(z)
        _, f0_ref = wav_to_mel_f0(ref_f0, hp)
        T = min(z_out.size(-1), len(f0_ref))
        z_out = z_out[..., :T]
        f0_use = torch.from_numpy(f0_ref[:T]).float().unsqueeze(0).to(device)
    elif mode.upper() == 'C':
        if ref_f0 is None:
            raise ValueError("Mode C requires --ref_f0_wav (F0 only source).")
        z_out = z                        # bypass M
        _, f0_ref = wav_to_mel_f0(ref_f0, hp)
        T = min(z_out.size(-1), len(f0_ref))
        z_out = z_out[..., :T]
        f0_use = torch.from_numpy(f0_ref[:T]).float().unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mel_out = task.decode_mel(z_out, f0_use)    # (1, T, M)
    wav = vocoder(mel_out.transpose(1, 2), f0=f0_use).squeeze().cpu().numpy()
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage2_ckpt', required=True)
    ap.add_argument('--vocoder_ckpt', required=True)
    ap.add_argument('--mode', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--src_wav', required=True)
    ap.add_argument('--ref_f0_wav', default=None,
                    help='Reference wav for F0 (required for Mode B/C).')
    ap.add_argument('--out_wav', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    task, vocoder, hp = load_task_and_vocoder(
        args.config, args.stage2_ckpt, args.vocoder_ckpt, device
    )

    sr = hp['audio_sample_rate']
    sr_src, src_i16 = wavfile.read(args.src_wav)
    src = (src_i16.astype(np.float32) / 32768.0) if src_i16.dtype == np.int16 else src_i16
    if src.ndim > 1: src = src[:, 0]
    assert sr_src == sr, f"Source wav sr={sr_src} but config sr={sr}"

    ref_arr = None
    if args.ref_f0_wav:
        sr_r, ref_i16 = wavfile.read(args.ref_f0_wav)
        assert sr_r == sr
        ref_arr = (ref_i16.astype(np.float32) / 32768.0) if ref_i16.dtype == np.int16 else ref_i16
        if ref_arr.ndim > 1: ref_arr = ref_arr[:, 0]

    wav_out = infer(task, vocoder, hp, args.mode, src, ref_f0=ref_arr, device=device)

    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.out_wav, sr, (wav_out * 32767).clip(-32768, 32767).astype(np.int16))
    print(f"[infer] wrote {args.out_wav}  (mode={args.mode})")


if __name__ == '__main__':
    main()
