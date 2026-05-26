"""Stage 1 z-space probe v2:量化「把 M4/VV 分開的那個維度,技巧佔多少比重」。

【framing(2026-05-20 user reframe)】
舊版問「env 有沒有漏進 z」→ 歧義:sfm 本來就部份是氣息技巧,CVAE 條件只給
ppg/f0/spk,其餘(含技巧、env)本來就落 z;且 VAE KL bottleneck + 主要用 M4 乾淨
資料 prior → 傾向不浪費 z 編碼環境噪音。所以「z 編碼 sfm」很可能是氣息技巧不是噪音。

真正該問:**z 空間把兩資料集分開的那個軸(domain axis),技巧能解釋多少比重?**

【方法】
- domain axis `w = normalize(mean_z(M4_train) − mean_z(VV_train))`,
  domain_coord = (z − global_mean_train) · w。高 = pro-like,低 = amateur-like。
  這個座標就是「分離兩資料集的維度」。
- 在 held-out VV(test split)內做變異分解:
  - Test A 技巧解釋力:domain_coord ~ amateur_score(技巧+氣息)的 r / slope / R²
  - Test B env 解釋力:domain_coord ~ sfm / hf_ratio
  - Test C 相對比重:標準化多元迴歸 domain_coord ~ score + sfm + hf,
    報 standardized β + 各自 unique R²(drop-one partial R²)
  - 共線性:report r(score, sfm) 等(breathy→低技巧 → 兩者可能相關)

【判讀】
- Test A 顯著正 + 技巧 unique R² 不小(且 ≥ env)→ 技巧是分離的實質成分 ✅
- 技巧 r≈0 但 env r 高 → 分離 env 主導 ❌
- 兩者都低 → within-VV 沒結構,分離只是均值平移(inconclusive)

【用法】
    & C:\\Users\\neo29\\miniconda3\\envs\\NSVB-ZH\\python.exe scripts\\stage1_zprobe.py ^
        --stage1-ckpt checkpoints\\stage1\\stage1_best.pt ^
        --binarized-root data\\binarized ^
        --vocalverse-label-dir data\\VocalVerse\\VocalVerse_Datasets-human_labels ^
        --out-dir outputs\\stage1_zprobe
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsvb.utils.audio_config import LATENT_DOWN_FACTOR
from scripts.audio_quality_probe import spectral_flatness_mean, high_freq_energy_ratio
from nsvb.data.vocalverse_mos import load_vocalverse_labels


def _pad_to_multiple(arr, multiple, pad_value=0.0):
    T = arr.shape[0]
    pad = (-T) % multiple
    if pad == 0:
        return arr
    pw = [(0, 0)] * arr.ndim
    pw[0] = (0, pad)
    return np.pad(arr, pw, constant_values=pad_value)


def load_svbvae(stage1_ckpt: str, device: str):
    from nsvb.model.svb_vae_zh import SVBVAEZh
    state = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = SVBVAEZh(
        num_mels=cfg.get("num_mels", 80), ppg_dim=cfg.get("ppg_dim", 1280),
        spk_emb_dim=cfg.get("spk_emb_dim", 256), latent_size=cfg.get("latent_size", 128),
        hidden_size=cfg.get("hidden_size", 192),
        enc_n_layers=cfg.get("enc_n_layers", 8), dec_n_layers=cfg.get("dec_n_layers", 4),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


@torch.no_grad()
def encode_z_and_env(cvae, npz_path: Path, device: str):
    with np.load(npz_path, allow_pickle=True) as d:
        mel_np = d["mel"].astype(np.float32)
        ppg_np = d["ppg"].astype(np.float32)
        f0_np = d["f0"].astype(np.float32)
        spk_np = d["spk_emb"].astype(np.float32)
        wav_np = d["wav"].astype(np.float32)
    T_orig = mel_np.shape[0]
    mel = torch.from_numpy(_pad_to_multiple(mel_np, LATENT_DOWN_FACTOR, -10.0)).unsqueeze(0).to(device)
    ppg = torch.from_numpy(_pad_to_multiple(ppg_np, LATENT_DOWN_FACTOR, 0.0)).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(_pad_to_multiple(f0_np, LATENT_DOWN_FACTOR, 0.0)).unsqueeze(0).to(device)
    spk = torch.from_numpy(spk_np).unsqueeze(0).to(device)
    mel_mask = torch.zeros_like(f0_t)
    mel_mask[:, :T_orig] = 1.0
    g = cvae.condition(ppg, f0_t, spk)
    g_sqz = cvae.fvae.g_pre_net(g)
    _z, m_q, _logs, _ = cvae.fvae.encoder(mel.transpose(1, 2), mel_mask.unsqueeze(1), g_sqz)
    z_mean = m_q.squeeze(0).mean(dim=-1).cpu().numpy()
    env = {"sfm": spectral_flatness_mean(wav_np), "hf_ratio": high_freq_energy_ratio(wav_np)}
    return z_mean, env


def parse_split(split_file: Path):
    items = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    return [it for it in items if "#" in it], [it for it in items if "#" not in it]


def gather(cvae, binarized_root, item_ids, ds_name, device, n_max, seed, labels=None):
    rng = random.Random(seed)
    picks = rng.sample(item_ids, min(n_max, len(item_ids))) if len(item_ids) > n_max else item_ids
    rows = []
    for i, it in enumerate(picks):
        p = binarized_root / ds_name / f"{it}.npz"
        if not p.exists():
            continue
        z, env = encode_z_and_env(cvae, p, device)
        row = {"item_id": it, "z": z, "sfm": env["sfm"], "hf_ratio": env["hf_ratio"]}
        if labels is not None:  # VV: 取 amateur_score
            base = re.sub(r"__c\d+$", "", it)
            wav_id = base.split("__", 1)[1] if "__" in base else base
            lab = labels.get(wav_id)
            row["amateur_score"] = (lab.amateur_score if (lab and lab.amateur_score is not None)
                                    else None)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"    {ds_name} {i+1}/{len(picks)}", flush=True)
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Stage 1 z-space probe v2 (technique share of domain axis)")
    ap.add_argument("--stage1-ckpt", required=True, type=Path)
    ap.add_argument("--binarized-root", required=True, type=Path)
    ap.add_argument("--vocalverse-label-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-bucket", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.binarized_root / "splits"
    print(f"[zprobe] device={args.device}, loading cvae...", flush=True)
    cvae, cfg = load_svbvae(str(args.stage1_ckpt), args.device)
    labels = load_vocalverse_labels(args.vocalverse_label_dir, require_pro=True)

    # gather
    tr_m4_items, tr_vv_items = parse_split(splits / "train.txt")
    te_m4_items, te_vv_items = parse_split(splits / "test.txt")
    print("[zprobe] gathering train M4...", flush=True)
    tr_m4 = gather(cvae, args.binarized_root, tr_m4_items, "m4singer", args.device, args.n_per_bucket, args.seed)
    print("[zprobe] gathering train VV...", flush=True)
    tr_vv = gather(cvae, args.binarized_root, tr_vv_items, "vocalverse", args.device, args.n_per_bucket, args.seed + 1, labels)
    print("[zprobe] gathering test M4...", flush=True)
    te_m4 = gather(cvae, args.binarized_root, te_m4_items, "m4singer", args.device, args.n_per_bucket, args.seed + 2)
    print("[zprobe] gathering test VV...", flush=True)
    te_vv = gather(cvae, args.binarized_root, te_vv_items, "vocalverse", args.device, args.n_per_bucket, args.seed + 3, labels)

    # ── domain axis(train split)──
    z_m4_tr = np.stack([r["z"] for r in tr_m4])
    z_vv_tr = np.stack([r["z"] for r in tr_vv])
    mean_m4, mean_vv = z_m4_tr.mean(0), z_vv_tr.mean(0)
    w = mean_m4 - mean_vv
    w = w / (np.linalg.norm(w) + 1e-10)
    global_mean = np.concatenate([z_m4_tr, z_vv_tr], 0).mean(0)

    def coord(z):
        return float((z - global_mean) @ w)

    # sanity:M4 vs VV domain_coord 分離
    m4_coords_te = np.array([coord(r["z"]) for r in te_m4])
    vv_coords_te = np.array([coord(r["z"]) for r in te_vv])
    sep = float(m4_coords_te.mean() - vv_coords_te.mean())
    pooled_std = float(np.sqrt(0.5 * (m4_coords_te.var() + vv_coords_te.var())) + 1e-10)
    sep_cohen_d = sep / pooled_std

    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    vv_labeled = [r for r in te_vv if r.get("amateur_score") is not None]
    n_lab = len(vv_labeled)
    results = {
        "domain_axis": {
            "m4_coord_mean": float(m4_coords_te.mean()),
            "vv_coord_mean": float(vv_coords_te.mean()),
            "separation": sep, "separation_cohen_d": sep_cohen_d,
        },
        "n_vv_labeled": n_lab,
    }

    # ── Sanity(point 1):env 均值兩資料集差多少(dereverb 後是否已接近)──
    def cohen_d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        ps = np.sqrt(0.5 * (a.var() + b.var())) + 1e-10
        return float((a.mean() - b.mean()) / ps)
    sfm_m4 = np.array([r["sfm"] for r in te_m4]); sfm_vv = np.array([r["sfm"] for r in te_vv])
    hf_m4 = np.array([r["hf_ratio"] for r in te_m4]); hf_vv = np.array([r["hf_ratio"] for r in te_vv])
    results["env_dataset_gap"] = {
        "sfm_m4_mean": float(sfm_m4.mean()), "sfm_vv_mean": float(sfm_vv.mean()),
        "sfm_cohen_d": cohen_d(sfm_m4, sfm_vv),
        "hf_m4_mean": float(hf_m4.mean()), "hf_vv_mean": float(hf_vv.mean()),
        "hf_cohen_d": cohen_d(hf_m4, hf_vv),
    }

    # ── Cross-dataset 殘差化(headline,point 1+3):env 能解釋多少「分離」?──
    # pooled M4+VV: domain_coord ~ sfm + hf;殘差化後 M4-VV Cohen's d 還剩多少
    dom_all = np.array([coord(r["z"]) for r in (te_m4 + te_vv)])
    env_all = np.column_stack([[r["sfm"] for r in (te_m4 + te_vv)],
                               [r["hf_ratio"] for r in (te_m4 + te_vv)]])
    is_m4 = np.array([True] * len(te_m4) + [False] * len(te_vv))
    env_model = LinearRegression().fit(env_all, dom_all)
    r2_env_on_domain = float(env_model.score(env_all, dom_all))
    resid = dom_all - env_model.predict(env_all)
    d_resid = cohen_d(resid[is_m4], resid[~is_m4])
    results["cross_dataset_residualize"] = {
        "r2_env_predicts_domain": r2_env_on_domain,
        "separation_cohen_d_original": sep_cohen_d,
        "separation_cohen_d_after_removing_env": d_resid,
        "fraction_separation_removed": float(1 - abs(d_resid) / (abs(sep_cohen_d) + 1e-10)),
    }

    # ── within-M4 分解(point 2):env only(M4 無技巧標籤)──
    if len(te_m4) >= 30:
        dom_m4 = np.array([coord(r["z"]) for r in te_m4])
        Xm4 = np.column_stack([[r["sfm"] for r in te_m4], [r["hf_ratio"] for r in te_m4]])
        Xm4s = StandardScaler().fit_transform(Xm4)
        ym4s = StandardScaler().fit_transform(dom_m4.reshape(-1, 1)).ravel()
        fm4 = LinearRegression().fit(Xm4s, ym4s)
        r2_m4 = float(fm4.score(Xm4s, ym4s))
        uniq_m4 = {}
        for i, n in enumerate(["sfm", "hf_ratio"]):
            Xi = np.delete(Xm4s, i, axis=1) if Xm4s.shape[1] > 1 else np.zeros((len(ym4s), 1))
            r2wo = float(LinearRegression().fit(Xi, ym4s).score(Xi, ym4s))
            uniq_m4[n] = max(0.0, r2_m4 - r2wo)
        results["within_m4_decomposition"] = {
            "r2_full": r2_m4, "standardized_beta": {n: float(b) for n, b in zip(["sfm", "hf_ratio"], fm4.coef_)},
            "unique_r2": uniq_m4,
        }

    if n_lab >= 30:
        dom = np.array([coord(r["z"]) for r in vv_labeled])
        score = np.array([r["amateur_score"] for r in vv_labeled])
        sfm = np.array([r["sfm"] for r in vv_labeled])
        hf = np.array([r["hf_ratio"] for r in vv_labeled])

        rA, pA = pearsonr(dom, score)
        rhoA, prhoA = spearmanr(dom, score)
        rB_sfm, pB_sfm = pearsonr(dom, sfm)
        rB_hf, pB_hf = pearsonr(dom, hf)
        r_score_sfm, _ = pearsonr(score, sfm)
        r_score_hf, _ = pearsonr(score, hf)

        X = np.column_stack([score, sfm, hf])
        names = ["technique", "sfm", "hf_ratio"]
        Xs = StandardScaler().fit_transform(X)
        ys = StandardScaler().fit_transform(dom.reshape(-1, 1)).ravel()
        full = LinearRegression().fit(Xs, ys)
        r2_full = float(full.score(Xs, ys))
        betas = {n: float(b) for n, b in zip(names, full.coef_)}
        unique_r2 = {}
        for i, n in enumerate(names):
            Xi = np.delete(Xs, i, axis=1)
            r2_wo = float(LinearRegression().fit(Xi, ys).score(Xi, ys))
            unique_r2[n] = max(0.0, r2_full - r2_wo)

        results["test_A_technique"] = {
            "pearson_r": float(rA), "p": float(pA),
            "spearman_rho": float(rhoA), "spearman_p": float(prhoA)}
        results["test_B_env"] = {
            "sfm_pearson_r": float(rB_sfm), "sfm_p": float(pB_sfm),
            "hf_pearson_r": float(rB_hf), "hf_p": float(pB_hf)}
        results["collinearity"] = {"r_score_sfm": float(r_score_sfm), "r_score_hf": float(r_score_hf)}
        results["test_C_decomposition"] = {
            "r2_full": r2_full, "standardized_beta": betas, "unique_r2": unique_r2}

        # ── 技巧方向 vs 分離方向(point 3)──
        # 先看技巧到底有沒有被 z 編碼:Ridge z→score,train/test split 內再切
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        Zvv = np.stack([r["z"] for r in vv_labeled])
        Ztr, Zte, ytr2, yte2 = train_test_split(Zvv, score, test_size=0.3, random_state=42)
        ridge = Ridge(alpha=10.0).fit(Ztr, ytr2)
        z_to_score_r2 = float(r2_score(yte2, ridge.predict(Zte)))
        w_tech = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-10)
        cos_tech_domain = float(np.abs(w_tech @ w))   # w 已正規化;abs 因方向 sign 任意
        results["technique_vs_domain_direction"] = {
            "z_to_amateur_score_r2_heldout": z_to_score_r2,
            "cosine_tech_vs_domain_axis": cos_tech_domain,
        }

    write_report(results, cfg, args.out_dir)
    with open(args.out_dir / "zprobe_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {args.out_dir / 'report.md'}", flush=True)


def write_report(R, cfg, out_dir):
    L = ["# Stage 1 z-space probe v2 — 分離軸上技巧的解釋力", ""]
    da = R["domain_axis"]
    L.append("## 0. Domain axis sanity(M4 vs VV 在分離軸上的位置,test split)")
    L.append("")
    L.append(f"- M4 coord mean = {da['m4_coord_mean']:+.3f}")
    L.append(f"- VV coord mean = {da['vv_coord_mean']:+.3f}")
    L.append(f"- separation = **{da['separation']:+.3f}**(Cohen's d = {da['separation_cohen_d']:+.2f})")
    L.append("")
    L.append("> domain axis = normalize(mean_z(M4_train) − mean_z(VV_train));"
             "domain_coord = (z − global_mean)·axis。M4 高 / VV 低 確認分離存在(d 大表分得開)。")
    L.append("")

    # 0b. env 均值差(point 1)
    if "env_dataset_gap" in R:
        e = R["env_dataset_gap"]
        L.append("## 0b. env 均值差(point 1:dereverb 後 M4/VV 是否已接近)")
        L.append("")
        L.append(f"- sfm: M4={e['sfm_m4_mean']:.4f} / VV={e['sfm_vv_mean']:.4f}, Cohen's d = **{e['sfm_cohen_d']:+.2f}**")
        L.append(f"- hf_ratio: M4={e['hf_m4_mean']:.4f} / VV={e['hf_vv_mean']:.4f}, Cohen's d = **{e['hf_cohen_d']:+.2f}**")
        L.append("> env d 若小(<0.5)= 該 metric 兩資料集已接近(dereverb 生效)→ 它就**不可能**是 domain d 分離的主因。")
        L.append("")

    # 1. headline:cross-dataset 殘差化
    if "cross_dataset_residualize" in R:
        c = R["cross_dataset_residualize"]
        L.append("## 1. 【headline】env 能解釋多少「跨資料集分離」?(cross-dataset 殘差化)")
        L.append("")
        L.append(f"- env(sfm+hf)迴歸 domain_coord 的 R² = {c['r2_env_predicts_domain']:.3f}")
        L.append(f"- 分離 Cohen's d:原始 **{c['separation_cohen_d_original']:+.2f}** → 移除 env 後 **{c['separation_cohen_d_after_removing_env']:+.2f}**")
        L.append(f"- **分離被 env 消掉的比例 = {c['fraction_separation_removed']*100:.1f}%**")
        L.append("")
        frac = c["fraction_separation_removed"]
        if frac > 0.7:
            L.append("→ **env 是分離主因**:移除後分離大幅崩塌 → M4/VV 在 z 分開主要是錄音特徵(confound)。")
        elif frac > 0.3:
            L.append("→ env 是**部分**成因:移除後分離縮小但仍在 → env + 其他(可能技巧)共同造成。")
        else:
            L.append("→ **env 不是分離主因**:移除後分離幾乎不變 → v2 within-VV 的 hf 相關只是 VV 內部品質參差,跟跨資料集分離無關。成因另尋。")
        L.append("")
        L.append("> 這是「分離成因」的正解。v2 within-VV 分解只測 VV 內部誰更靠 M4,**不等於**誰造成 M4-VV 均值分離。")
        L.append("")

    # 2. within-M4 分解(point 2)
    if "within_m4_decomposition" in R:
        m = R["within_m4_decomposition"]
        L.append("## 2. within-M4 分解(point 2:M4 內部結構,env only,無技巧標籤)")
        L.append("")
        L.append(f"- 全模型 R²(env 解釋 within-M4 domain_coord)= {m['r2_full']:.3f}")
        L.append(f"- unique R²: sfm={m['unique_r2']['sfm']:.4f}, hf_ratio={m['unique_r2']['hf_ratio']:.4f}")
        L.append("> 對照 within-VV:若 M4 內部 env 也解釋 domain 位置 → 兩資料集內部結構平行(env 是共通的 within-dataset 品質軸)。")
        L.append("")

    # 3. 技巧方向 vs 分離方向(point 3)
    if "technique_vs_domain_direction" in R:
        t = R["technique_vs_domain_direction"]
        L.append("## 3. 技巧方向 vs 分離方向(point 3:z VV→M4 真含技巧變換?)")
        L.append("")
        L.append(f"- **z → amateur_score 的 held-out R² = {t['z_to_amateur_score_r2_heldout']:+.3f}**")
        L.append(f"- cos(技巧方向, 分離軸) = {t['cosine_tech_vs_domain_axis']:.3f}")
        L.append("")
        r2zs = t["z_to_amateur_score_r2_heldout"]
        if r2zs < 0.02:
            L.append("→ ⚠ **技巧幾乎沒被 z 編碼**(R²≈0)→ 「技巧方向」w_tech 是噪音,cosine 不可信。"
                     "**這本身才是重點**:z 在 amateur 範圍內幾乎不帶可線性解碼的技巧資訊 → "
                     "z VV→M4 的變換不太可能是技巧變換。")
        elif t["cosine_tech_vs_domain_axis"] > 0.3:
            L.append("→ 技巧有被 z 編碼且技巧方向跟分離軸對齊 → 提升技巧會讓 z 更像 M4 → 分離含技巧成分 ✅")
        else:
            L.append("→ 技巧有被 z 編碼但方向跟分離軸近正交 → 提升技巧**不會**讓 z 更像 M4 → 分離不是技巧 ❌")
        L.append("")

    if "test_A_technique" not in R:
        L.append(f"⚠ VV labeled samples = {R.get('n_vv_labeled', 0)} < 30,無法做 within-VV 變異分解。")
        out_dir.joinpath("report.md").write_text("\n".join(L), encoding="utf-8")
        return

    A, B, C, col = R["test_A_technique"], R["test_B_env"], R["test_C_decomposition"], R["collinearity"]
    L.append(f"分析樣本:held-out VV test split,有 amateur_score 標籤的 {R['n_vv_labeled']} 個")
    L.append("")

    # Test A
    L.append("## Test A:技巧解釋力(domain_coord vs amateur_score)")
    L.append("")
    sigA = A["p"] < 0.05
    L.append(f"- Pearson r = **{A['pearson_r']:+.3f}**(p = {A['p']:.2e}){' ✅ 顯著' if sigA else ' ✗ 不顯著'}")
    L.append(f"- Spearman ρ = {A['spearman_rho']:+.3f}(p = {A['spearman_p']:.2e})")
    L.append("")
    L.append("> 正相關 = 技巧越高(amateur_score 大)→ 在分離軸上越靠 pro 端 → 技巧是分離的成分。")
    L.append("")

    # Test B
    L.append("## Test B:env 解釋力(domain_coord vs sfm / hf_ratio)")
    L.append("")
    L.append(f"- sfm: r = {B['sfm_pearson_r']:+.3f}(p = {B['sfm_p']:.2e})")
    L.append(f"- hf_ratio: r = {B['hf_pearson_r']:+.3f}(p = {B['hf_p']:.2e})")
    L.append("")

    # collinearity
    L.append("## 共線性檢查(predictor 之間)")
    L.append("")
    L.append(f"- r(amateur_score, sfm) = {col['r_score_sfm']:+.3f}")
    L.append(f"- r(amateur_score, hf_ratio) = {col['r_score_hf']:+.3f}")
    L.append("> 若這些很大,代表技巧分數本身跟 env 相關(例如氣息差→sfm 高),"
             "Test A/B 的單變量相關會互相污染 → 看 Test C 的 unique R² 才準。")
    L.append("")

    # Test C
    L.append("## Test C:相對比重(標準化多元迴歸 domain_coord ~ score + sfm + hf)")
    L.append("")
    L.append(f"- 全模型 R² = **{C['r2_full']:.3f}**(within-VV domain_coord 被三者解釋的總比例)")
    L.append("")
    hdr = ["predictor", "standardized β", "unique R²(drop-one)"]
    rows = [[n, f"{C['standardized_beta'][n]:+.3f}", f"{C['unique_r2'][n]:.4f}"]
            for n in ["technique", "sfm", "hf_ratio"]]
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows]) for i, h in enumerate(hdr)]
    L.append("| " + " | ".join(h.ljust(w) for h, w in zip(hdr, widths)) + " |")
    L.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        L.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    L.append("")
    L.append("> unique R² = 把該 predictor 從全模型移除後 R² 掉多少 = 它**獨有**的解釋力"
             "(已扣掉跟其他 predictor 共享的部分)。")
    L.append("")

    # 綜合判讀
    L.append("## 綜合判讀")
    L.append("")
    tech_uniq = C["unique_r2"]["technique"]
    env_uniq = C["unique_r2"]["sfm"] + C["unique_r2"]["hf_ratio"]
    r2_full = C["r2_full"]
    if r2_full < 0.05:
        L.append(f"⚠ **within-VV domain_coord 幾乎無法被解釋(R²={r2_full:.3f})**:"
                 "技巧跟 env 都解釋不了 VV 內在分離軸上的位置。代表分離主要是 M4/VV 的「均值平移」,"
                 "within-VV 沒有沿分離軸的細結構 → 無法用這方法判技巧比重。"
                 "可能 amateur_score 範圍太窄(篩 ≤3.0)或 domain axis 不是技巧的主軸。")
    elif tech_uniq >= env_uniq and A["p"] < 0.05:
        L.append(f"✅ **技巧是分離的實質成分**:technique unique R²={tech_uniq:.4f} "
                 f"≥ env({env_uniq:.4f}),且 Test A 顯著。"
                 "z 把兩資料集分開的維度,技巧佔可辨識的比重。")
    elif env_uniq > tech_uniq and (B["sfm_p"] < 0.05 or B["hf_p"] < 0.05):
        L.append(f"❌ **env 主導分離**:env unique R²={env_uniq:.4f} > technique({tech_uniq:.4f})。"
                 "分離軸偏向錄音特徵而非技巧。考慮加強 dereverb(Plan C)。"
                 "⚠ 但注意 sfm 部份是氣息技巧,env 比重高不必然是純 confound。")
    else:
        L.append(f"🤔 **混合/不明確**:technique unique R²={tech_uniq:.4f}, "
                 f"env unique R²={env_uniq:.4f}, 全模型 R²={r2_full:.3f}。看數字權衡。")
    L.append("")
    L.append("**caveat**:amateur_score 篩到 ≤3.0,範圍 [1,3] 偏窄,gradient 偵測力受限;"
             "amateur_score 是 2 評審粗標,有雜訊。結論需配合聽測佐證。")

    out_dir.joinpath("report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()