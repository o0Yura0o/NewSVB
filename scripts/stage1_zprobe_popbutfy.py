"""Stage 1 z-space probe — PopBuTFy 版(對標 v2 之 M4+VV 57/43 分解)。

【目的】
v2(中文 M4+VV)之 z-probe v3 結果:domain axis 分離度 ~~57% 由 env(sfm + hf_ratio)
解釋,43% 未知(可能技巧)。本研究承認此分離主要是 env 軸,M 之 alignment +0.84 之大
部分為 env 補償。

PopBuTFy 之 env 對稱前提下,domain axis(popbutfy_pro vs popbutfy_amateur)之 env 解
釋比例**預期顯著小於 57%**(因為 DF3+LUFS 效應於兩端對稱抵消,domain axis 不再對應
env 軸,而對應真正之技巧差)。

若觀察:
- PopBuTFy fraction_env_removed ≈ v2 之 57%   → env 軸仍是主要 separation source,
  我們之 z-probe 解讀(env 主導)有 PopBuTFy 之獨立支持
- PopBuTFy fraction_env_removed << v2 之 57%(例如 ~20%)→ env 對稱前提下,domain
  separation 不靠 env,反而暴露純技巧差;進一步強化「v2 之 +0.84 ≈ env 補償」之解讀

兩種情況都對論文有意義(前者強化 z-probe 解讀,後者強化 §4.6 之 env-symmetric
caveat narrative)。

【跟 v2 z-probe 之差異】
- Domain axis w = mean_z(popbutfy_pro) − mean_z(popbutfy_amateur)(v2 是 M4 vs VV)
- 跳過 Test A(score~domain)— PopBuTFy 沒 amateur_score
- 跳過 Test C(score + env 多元回歸)— 同上
- 保留 Test B(env~domain)+ cross-dataset residualize(headline metric)+ within-side
  env analysis(兩 side 各自:env 在 domain 內解釋多少 within-side 變異)

【使用】
    & C:\\Users\\neo29\\miniconda3\\envs\\NSVB-ZH\\python.exe ^
        scripts\\stage1_zprobe_popbutfy.py ^
        --stage1-ckpt checkpoints\\stage1\\stage1_best.pt ^
        --binarized-root data\\binarized ^
        --out-dir outputs\\stage1_zprobe_popbutfy
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 重用 v2 z-probe 之公用函式
from scripts.stage1_zprobe import (
    _pad_to_multiple,
    load_svbvae,
    encode_z_and_env,
)


def parse_popbutfy_split(split_file: Path):
    """切 train/test split 內 item_ids 為 (amateur, pro)。

    PopBuTFy item_id 格式:
      Female1#singing#Almost_lover_Amateur__0
      Female1#singing#Almost_lover_Professional__0
    """
    items = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    amateur = [it for it in items if "_Amateur_" in it]
    pro     = [it for it in items if "_Professional_" in it]
    return amateur, pro


def gather(cvae, binarized_root, item_ids, ds_name, device, n_max, seed):
    rng = random.Random(seed)
    picks = rng.sample(item_ids, min(n_max, len(item_ids))) if len(item_ids) > n_max else item_ids
    rows = []
    for i, it in enumerate(picks):
        p = binarized_root / ds_name / f"{it}.npz"
        if not p.exists():
            continue
        z, env = encode_z_and_env(cvae, p, device)
        rows.append({"item_id": it, "z": z, "sfm": env["sfm"], "hf_ratio": env["hf_ratio"]})
        if (i + 1) % 100 == 0:
            print(f"    {ds_name} {i+1}/{len(picks)}", flush=True)
    return rows


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ps = np.sqrt(0.5 * (a.var() + b.var())) + 1e-10
    return float((a.mean() - b.mean()) / ps)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Stage 1 z-probe (PopBuTFy variant)")
    ap.add_argument("--stage1-ckpt", required=True, type=Path)
    ap.add_argument("--binarized-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-bucket", type=int, default=600,
                    help="每 (split, side) 採樣上限。v2 預設 600,PopBuTFy 測試集較小(amateur 659 / pro 705),全採亦可")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.binarized_root / "splits_popbutfy"
    if not splits.exists():
        raise SystemExit(f"splits_popbutfy/ 不存在:{splits} — 先跑 setup_popbutfy_binarized.py")

    print(f"[zprobe-pop] device={args.device}, loading cvae...", flush=True)
    cvae, cfg = load_svbvae(str(args.stage1_ckpt), args.device)

    # gather train / test 各 (amateur, pro)
    tr_am_items, tr_pr_items = parse_popbutfy_split(splits / "train.txt")
    te_am_items, te_pr_items = parse_popbutfy_split(splits / "test.txt")
    print(f"[zprobe-pop] split sizes: "
          f"train(am={len(tr_am_items)} pr={len(tr_pr_items)}) "
          f"test(am={len(te_am_items)} pr={len(te_pr_items)})", flush=True)

    print("[zprobe-pop] gathering train amateur...", flush=True)
    tr_am = gather(cvae, args.binarized_root, tr_am_items, "popbutfy_amateur",
                   args.device, args.n_per_bucket, args.seed)
    print("[zprobe-pop] gathering train pro...", flush=True)
    tr_pr = gather(cvae, args.binarized_root, tr_pr_items, "popbutfy_pro",
                   args.device, args.n_per_bucket, args.seed + 1)
    print("[zprobe-pop] gathering test amateur...", flush=True)
    te_am = gather(cvae, args.binarized_root, te_am_items, "popbutfy_amateur",
                   args.device, args.n_per_bucket, args.seed + 2)
    print("[zprobe-pop] gathering test pro...", flush=True)
    te_pr = gather(cvae, args.binarized_root, te_pr_items, "popbutfy_pro",
                   args.device, args.n_per_bucket, args.seed + 3)

    # ── domain axis(train split)── pro - amateur,跟 v2 之 M4 - VV 方向一致(pro 端為正)
    z_pr_tr = np.stack([r["z"] for r in tr_pr])
    z_am_tr = np.stack([r["z"] for r in tr_am])
    mean_pr, mean_am = z_pr_tr.mean(0), z_am_tr.mean(0)
    w = mean_pr - mean_am
    w = w / (np.linalg.norm(w) + 1e-10)
    global_mean = np.concatenate([z_pr_tr, z_am_tr], 0).mean(0)

    def coord(z):
        return float((z - global_mean) @ w)

    pr_coords_te = np.array([coord(r["z"]) for r in te_pr])
    am_coords_te = np.array([coord(r["z"]) for r in te_am])
    sep = float(pr_coords_te.mean() - am_coords_te.mean())
    pooled_std = float(np.sqrt(0.5 * (pr_coords_te.var() + am_coords_te.var())) + 1e-10)
    sep_cohen_d = sep / pooled_std

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    results = {
        "n_test": {"amateur": len(te_am), "pro": len(te_pr)},
        "domain_axis": {
            "pro_coord_mean":     float(pr_coords_te.mean()),
            "amateur_coord_mean": float(am_coords_te.mean()),
            "separation":         sep,
            "separation_cohen_d": sep_cohen_d,
        },
    }

    # ── Sanity:env 兩 side gap(PopBuTFy 預期 ≈ 0,v2 之 sfm cohen_d > 1.5)──
    sfm_pr = np.array([r["sfm"] for r in te_pr]); sfm_am = np.array([r["sfm"] for r in te_am])
    hf_pr  = np.array([r["hf_ratio"] for r in te_pr]); hf_am = np.array([r["hf_ratio"] for r in te_am])
    results["env_dataset_gap"] = {
        "sfm_pro_mean":     float(sfm_pr.mean()),
        "sfm_amateur_mean": float(sfm_am.mean()),
        "sfm_cohen_d":      cohen_d(sfm_pr, sfm_am),
        "hf_pro_mean":      float(hf_pr.mean()),
        "hf_amateur_mean":  float(hf_am.mean()),
        "hf_cohen_d":       cohen_d(hf_pr, hf_am),
    }

    # ── Cross-dataset 殘差化(headline)──
    # v2 對應結果:r2_env_predicts_domain ~ 0.57,fraction_separation_removed ~ 0.57
    # 即 env 解釋了 ~57% 的 pro/amateur 分離度。剩 ~43% 是「未知(可能技巧)」。
    # PopBuTFy 預期:env 對稱 → env 解釋力顯著降低
    dom_all = np.array([coord(r["z"]) for r in (te_pr + te_am)])
    env_all = np.column_stack([
        [r["sfm"] for r in (te_pr + te_am)],
        [r["hf_ratio"] for r in (te_pr + te_am)],
    ])
    is_pro = np.array([True] * len(te_pr) + [False] * len(te_am))
    env_model = LinearRegression().fit(env_all, dom_all)
    r2_env_on_domain = float(env_model.score(env_all, dom_all))
    resid = dom_all - env_model.predict(env_all)
    d_resid = cohen_d(resid[is_pro], resid[~is_pro])
    results["cross_dataset_residualize"] = {
        "r2_env_predicts_domain":                  r2_env_on_domain,
        "separation_cohen_d_original":             sep_cohen_d,
        "separation_cohen_d_after_removing_env":   d_resid,
        "fraction_separation_removed":             float(1 - abs(d_resid) / (abs(sep_cohen_d) + 1e-10)),
        "_note_v2_baseline":                       "v2(M4+VV)對應數字 ~ 0.57;<< 0.57 表 env 不再主導分離",
    }

    # ── Within-side env analysis(env 在「同 side」內解釋多少 domain 變異)──
    for side_name, side_rows in [("pro", te_pr), ("amateur", te_am)]:
        if len(side_rows) < 30:
            continue
        dom_s = np.array([coord(r["z"]) for r in side_rows])
        X = np.column_stack([[r["sfm"] for r in side_rows], [r["hf_ratio"] for r in side_rows]])
        Xs = StandardScaler().fit_transform(X)
        ys = StandardScaler().fit_transform(dom_s.reshape(-1, 1)).ravel()
        f = LinearRegression().fit(Xs, ys)
        r2 = float(f.score(Xs, ys))
        results[f"within_{side_name}_env"] = {
            "n":                len(side_rows),
            "r2_env_on_domain": r2,
            "beta_sfm":         float(f.coef_[0]),
            "beta_hf":          float(f.coef_[1]),
        }

    # ── output ──
    out_json = args.out_dir / "report.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[zprobe-pop] saved {out_json}", flush=True)

    # text summary 給人眼
    s = []
    s.append("# Stage 1 z-probe — PopBuTFy variant")
    s.append("")
    s.append(f"- test n: amateur={len(te_am)}, pro={len(te_pr)}")
    da = results["domain_axis"]
    s.append(f"- domain axis (pro - amateur) Cohen's d: **{da['separation_cohen_d']:.3f}**")
    eg = results["env_dataset_gap"]
    s.append(f"- env gap: sfm_d={eg['sfm_cohen_d']:.3f}, hf_d={eg['hf_cohen_d']:.3f}")
    s.append("  (v2 對應 sfm_d > 1.5; PopBuTFy 預期 ≈ 0,因 env 對稱)")
    cdr = results["cross_dataset_residualize"]
    s.append("")
    s.append("## Headline: env 解釋 domain separation 之比例")
    s.append(f"- r²(env → domain) = **{cdr['r2_env_predicts_domain']:.3f}**")
    s.append(f"- separation Cohen's d 原本 = {cdr['separation_cohen_d_original']:.3f}")
    s.append(f"- 扣除 env 後剩 = {cdr['separation_cohen_d_after_removing_env']:.3f}")
    s.append(f"- **fraction removed by env = {cdr['fraction_separation_removed']*100:.1f}%**")
    s.append("  (v2 對應 ~57%,值 << 57% 表 PopBuTFy 之分離不靠 env)")
    s.append("")
    for side in ("pro", "amateur"):
        key = f"within_{side}_env"
        if key in results:
            w = results[key]
            s.append(f"- within {side}: r²(env→domain)={w['r2_env_on_domain']:.3f}, "
                     f"β_sfm={w['beta_sfm']:.3f}, β_hf={w['beta_hf']:.3f}")

    out_md = args.out_dir / "report.md"
    out_md.write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"[zprobe-pop] saved {out_md}", flush=True)

    # console summary
    print("\n" + "=" * 60)
    for line in s:
        print(line)
    print("=" * 60)


if __name__ == "__main__":
    main()