"""NSVB-ZH two-stage model flow diagram (paper Fig.1 style)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# palette matching the pptx deck
ACCENT  = '#1F3A68'
TRAIN   = '#BFD7ED'
FROZEN  = '#E5E5E5'
DATA    = '#FFF5E1'
LATENT  = '#DEEAF6'
LOSS_BG = '#F6D4D4'
LOSS_TX = '#8B1A1A'
AM_COL  = '#FFE6CC'
PR_COL  = '#D5E8D4'


def box(ax, x, y, w, h, text, fc=TRAIN, ec=ACCENT, fs=11.0, lw=1.4, bold=True):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.06',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color=ACCENT)


def arr(ax, x1, y1, x2, y2, color='#333', lw=1.3, rad=0.0, style='->'):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, lw=lw, mutation_scale=13,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)


def loss(ax, x, y, text, w=2.4, h=0.5, fs=10.0):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.10',
                       fc=LOSS_BG, ec=LOSS_TX, lw=1.1)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=LOSS_TX)


def tag(ax, x, y, text, fs=9, color='#555', ha='center', italic=True):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            style='italic' if italic else 'normal')


# ============================================================
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for ax in (ax1, ax2):
    ax.set_xlim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')

# ============================================================
# STAGE 1  —  CVAE pre-training
# ============================================================
ax1.set_ylim(0, 5.6)

ax1.add_patch(FancyBboxPatch((0.15, 4.85), 13.7, 0.55,
              boxstyle='round,pad=0.02,rounding_size=0.08',
              fc=ACCENT, ec=ACCENT))
ax1.text(7.0, 5.12,
         'Stage 1 · CVAE 預訓練  (業餘 x_a 與職業 x_p 皆作為重建目標)',
         ha='center', va='center', fontsize=13.5,
         fontweight='bold', color='white')

mel_y = 3.1
H = 0.9
box(ax1, 0.4,  mel_y, 1.5, H, 'x (mel)\n業餘 / 職業', fc=DATA,   fs=10)
box(ax1, 2.4,  mel_y, 1.7, H, 'Encoder φ',            fc=TRAIN,  fs=13)
box(ax1, 4.6,  mel_y, 2.1, H, 'z ~ q(z|x,c)\n[B,128,T/4]', fc=LATENT, fs=10)
box(ax1, 7.3,  mel_y, 1.7, H, 'Decoder θ',            fc=TRAIN,  fs=13)
box(ax1, 9.5,  mel_y, 1.5, H, 'x̂ (mel)',             fc=DATA,   fs=11)
box(ax1, 11.5, mel_y, 1.9, H, 'D_mel\n(mel 判別器)', fc=TRAIN,  fs=11)

# condition
box(ax1, 3.0, 1.85, 6.6, 0.6,
    'c  =  F0  |  PPG (phoneme)  |  spk_emb',
    fc=DATA, fs=11)

# main arrows
arr(ax1, 1.9, mel_y + H/2, 2.4, mel_y + H/2)
arr(ax1, 4.1, mel_y + H/2, 4.6, mel_y + H/2)
arr(ax1, 6.7, mel_y + H/2, 7.3, mel_y + H/2)
arr(ax1, 9.0, mel_y + H/2, 9.5, mel_y + H/2)
arr(ax1, 11.0, mel_y + H/2, 11.5, mel_y + H/2)

# condition arrows to encoder + decoder
arr(ax1, 3.3, 2.45, 3.3, mel_y, color='#777')
arr(ax1, 8.0, 2.45, 8.0, mel_y, color='#777')

# real mel short feed into D_mel (sibling input)
arr(ax1, 1.1, mel_y, 12.4, mel_y - 0.55, rad=-0.35, color='#AAA', lw=1.0)
tag(ax1, 12.4, mel_y - 0.75, '(real / fake)', fs=8.5, color='#888')

# losses
loss(ax1, 5.6, 0.9,
     'L_recon (ELBO) = ‖x̂ − x‖₁  +  KL( q(z|x,c) ‖ p(z) )',
     w=6.4, h=0.55, fs=10)
loss(ax1, 11.7, 0.9, 'L_adv  via  D_mel', w=3.0, h=0.55, fs=10)

tag(ax1, 7.0, 0.25,
    'Stage 1 可訓練：Encoder φ,  Decoder θ,  D_mel',
    fs=10.5, color=ACCENT)


# ============================================================
# STAGE 2  —  Mapping training (unpaired)
# ============================================================
ax2.set_ylim(0, 7.6)

ax2.add_patch(FancyBboxPatch((0.15, 6.85), 13.7, 0.55,
              boxstyle='round,pad=0.02,rounding_size=0.08',
              fc=ACCENT, ec=ACCENT))
ax2.text(7.0, 7.12,
         'Stage 2 · Mapping 訓練  (非對稱 unpaired — NSVB-ZH 改動所在)',
         ha='center', va='center', fontsize=13.5,
         fontweight='bold', color='white')

am_y = 5.1   # amateur path
pr_y = 2.2   # pro path
dz_y = 3.65  # D_z row
H = 0.85

# ---- amateur stream ----
box(ax2, 0.2,  am_y, 1.3, H, 'x_a (mel)\n業餘', fc=AM_COL,  fs=10)
box(ax2, 1.8,  am_y, 1.9, H, 'Encoder φ\n(凍結)', fc=FROZEN, fs=10.5)
box(ax2, 4.0,  am_y, 1.1, H, 'z_a',            fc=LATENT, fs=11)
box(ax2, 5.4,  am_y, 2.1, H, 'M  (residual)\nf(z) = z + Δ(z)',
    fc=TRAIN, fs=10.5)
box(ax2, 7.8,  am_y, 1.3, H, 'f(z_a)',         fc=LATENT, fs=11)
box(ax2, 9.4,  am_y, 1.9, H, 'Decoder θ\n(凍結)', fc=FROZEN, fs=10.5)
box(ax2, 11.6, am_y, 1.3, H, 'x̂ (mel)',       fc=DATA,   fs=11)

# arrows amateur
arr(ax2, 1.5, am_y + H/2, 1.8, am_y + H/2)
arr(ax2, 3.7, am_y + H/2, 4.0, am_y + H/2)
arr(ax2, 5.1, am_y + H/2, 5.4, am_y + H/2)
arr(ax2, 7.5, am_y + H/2, 7.8, am_y + H/2)
arr(ax2, 9.1, am_y + H/2, 9.4, am_y + H/2)
arr(ax2, 11.3, am_y + H/2, 11.6, am_y + H/2)

# ---- pro stream ----
box(ax2, 0.2, pr_y, 1.3, H, 'x_p (mel)\n職業', fc=PR_COL,  fs=10)
box(ax2, 1.8, pr_y, 1.9, H, 'Encoder φ\n(凍結)', fc=FROZEN, fs=10.5)
box(ax2, 4.0, pr_y, 1.1, H, 'z_p',            fc=LATENT, fs=11)
arr(ax2, 1.5, pr_y + H/2, 1.8, pr_y + H/2)
arr(ax2, 3.7, pr_y + H/2, 4.0, pr_y + H/2)


# ---- D_z ----
box(ax2, 5.0, dz_y, 2.9, 0.9,
    'D_z  (conditional)\n條件: Soft Register Bucket (5)\n+ Discrete Phoneme ID',
    fc=TRAIN, fs=9.5)

# feeds into D_z
arr(ax2, 8.3, am_y, 7.2, dz_y + 0.9, rad=0.15, color='#C0392B')  # f(z_a) → D_z (fake)
arr(ax2, 4.55, pr_y + H, 5.3, dz_y,   rad=-0.15, color='#2E8B57')  # z_p   → D_z (real)
tag(ax2, 8.7, dz_y + 1.25, 'fake',  fs=9, color='#C0392B')
tag(ax2, 4.75, dz_y - 0.15, 'real', fs=9, color='#2E8B57')

# condition extraction hint (below D_z)
tag(ax2, 6.45, dz_y - 0.35,
    'F0 → log-Gaussian Soft Bucket(σ=0.3)   |   PPG → argmax phoneme',
    fs=8, color='#555')

# L_adv_z
loss(ax2, 6.5, dz_y + 1.15, 'L_adv_z', w=1.6, h=0.42, fs=10)

# ---- PatchNCE (z_a ↔ f(z_a)) ----
arr(ax2, 4.55, am_y + H + 0.02, 8.45, am_y + H + 0.02,
    rad=-0.38, color=LOSS_TX, style='<->', lw=1.4)
tag(ax2, 6.5, am_y + H + 0.95, 'L_PatchNCE  (frame-wise contrastive)',
    fs=10, color=LOSS_TX)

# ---- PPG extractor + L_PPG ----
box(ax2, 11.2, 0.85, 1.9, 0.6,
    'PPG 抽取器\n(WeNet / Whisper)',
    fc=FROZEN, fs=9)
arr(ax2, 12.25, am_y, 12.15, 1.45, color='#C0392B')            # x̂ → PPG
arr(ax2, 0.85, am_y, 11.25, 1.15, rad=-0.28, color='#888')      # x_a → PPG (side)
loss(ax2, 9.2, 0.55,
     'L_PPG  =  D_KL( PPG(x̂)  ‖  PPG(x_a) )',
     w=4.0, h=0.48, fs=9.5)

# ---- D_mel (reused from Stage 1, pro-only real in Stage 2) ----
box(ax2, 2.8, 0.9, 2.4, 0.7,
    'D_mel  (復用 Stage 1)\n低 LR, Real 僅 x_p',
    fc=TRAIN, fs=9.5)
arr(ax2, 12.2, am_y, 5.0, 1.55, rad=-0.32, color='#C0392B')  # x̂ → D_mel
loss(ax2, 3.9, 0.3,
     'L_adv_mel  =  ( D_mel(x̂) − 1 )²',
     w=3.4, h=0.42, fs=9.5)

# training legend
tag(ax2, 7.0, 6.4,
    '可訓練：M,  D_z,  D_mel (低 LR)      凍結：Encoder φ,  Decoder θ',
    fs=10.5, color=ACCENT)


# ============================================================
out = 'c:/Users/neo29/workspace/SVC/NSVB-ZH/figures/nsvb_zh_flow.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
