# Phase 0 — Probe Interpretation

This is the go/no-go gate before Stage 2.

## Two questions the probe answers

1. **Is "M4 vs OpenSinger" = "amateur vs pro"?**
   If a simple classifier can distinguish them with high accuracy from CVAE
   latents, the gap is probably dominated by dataset identity (singer, room,
   mic, mixing), *not* singing quality. D_z will learn that identity, M will
   learn to imitate it, and the resulting "enhancement" will be unlistenable
   or just different-sounding — not better.

2. **Does the pretrained HifiGAN NSF vocoder handle Chinese singing?**
   If mel reconstruction is already broken at the vocoder level, any
   improvement M learns becomes invisible downstream.

## Probe accuracy interpretation

| Accuracy | Case | Interpretation | Action |
|---|---|---|---|
| `< 0.55` | — | Near chance. z is near-domain-invariant. | Double-check sample count and label integrity. |
| `0.55 – 0.60` | **A** | Some residual bias, within noise. | Run base architecture (`use_grl: false`). Expect clean Stage 2 run. |
| `0.60 – 0.75` | **B** | Moderate dataset bias. Recoverable via domain adversarial training. | Enable GRL in Stage 1. Plan extra 20 % compute budget. |
| `0.75 – 0.90` | **C (hard)** | Dataset dominates latent. GRL alone will not fix it. | Do not run Stage 2 as-is. Use C1 / C2 / C3 below. |
| `> 0.90` | **C (catastrophic)** | z is basically a dataset id. | Reconsider the project framing. |

## If Case C — fallback options

### C1. MOSNet re-labeling (recommended)

Re-label the *combined* M4+OpenSinger pool by predicted MOS quartile:
top quartile → "pro" (1), bottom quartile → "amateur" (0), middle → discard
(or keep as unlabeled for semi-supervised extension).

Advantages:
- Breaks the dataset correlation with the label.
- Provides an objective evaluation metric (ΔMOS_predicted).
- Keeps the entire Stage 2 architecture unchanged.

Risks:
- MOSNet is itself noisy, especially on singing voice (most MOSNets were
  trained on TTS). Consider ensembling UTMOS + MOSNet22 + DNSMOS.
- Need to verify the re-labeling makes acoustic sense on a small human sample.

### C2. DTW pseudo-pairs

Falls back to the original NSVB paired architecture. Reuse
`preprocess_align.py` on songs that appear in both datasets, set a similarity
threshold (e.g. chroma cosine > 0.7) to keep only reliable pairs.

Advantages:
- Uses well-tested code paths from upstream NSVB.
- Paired supervision is the strongest signal available.

Risks:
- Very few song overlaps likely; pair count may be too small.
- Reframes the paper contribution (not an "unpaired" method anymore).

### C3. Chinese karaoke dataset

Switch to a naturally-paired Chinese dataset (原唱 / 翻唱). Candidates:
- 酷狗 karaoke crawls (licensing unclear).
- 歡歌 (Changba) datasets used by some academic groups.
- Internal collections if available.

Risks:
- Licensing / legality.
- Distribution shift from open singing datasets.

## Vocoder test interpretation

| Metric | Pass | Marginal | Fail |
|---|---|---|---|
| mel SSIM | ≥ 0.90 | 0.85 – 0.90 | < 0.85 |
| F0 RMSE (Hz) | ≤ 10 | 10 – 20 | > 20 |

Both metrics must be in the same column to assign that verdict.

**If marginal**: dump the reconstructed wavs (`--save_wavs`) and listen.
Chinese tones are carried by F0 contour, so F0 RMSE > 15 Hz usually hurts
perceived intelligibility before it hurts mel SSIM.

**If fail on only one dataset**: the vocoder likely needs fine-tuning on
that dataset's acoustic profile. Fine-tune for 20–50k steps before Stage 2.

**If fail on both**: retrain / fine-tune the vocoder on a balanced Chinese
singing mix before doing anything else.

## Notes

- The probe uses **utterance-level** pooling (mean + std over valid frames)
  rather than frame-level, so short clips and long clips contribute equally.
  A frame-level probe would be even more discriminative but less stable.
- The probe is run on a **half-cooked** Stage 1 ckpt (50–80k steps) so that
  z has converged enough to be meaningful but has not overfit. Running it
  on a fully-converged ckpt biases upward.
- Stratified split ensures the test set has both datasets in equal parts,
  so 0.5 = chance and accuracy is directly interpretable.
