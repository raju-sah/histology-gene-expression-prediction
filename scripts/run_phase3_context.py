"""Phase 3: does spatial context (neighbouring patches) improve per-gene prediction?

    .venv/bin/python scripts/run_phase3_context.py [--encoders uni2_h] [--attention-encoder uni2_h]

Same LOGO folds, targets and ridge protocol as Phase 2. Variants per encoder:
  single  : the patch embedding
  k1, k2  : [patch, mean of neighbours within 1 / 2 grid rings]   (PCA-256 on the concatenation, fitted on train)
  attn_k1 : centre-query attention pooling over the 3x3 neighbourhood (one encoder; torch, CPU)
Reports per-gene gains (variant - single), the number of folds with a positive gain, and how gains relate to the
Phase 2 gene features. Outputs: reports/phase3/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.atlas import common_genes, load_sparse_samples, run_atlas, transform_y  # noqa: E402
from hestfm.context import AttentionContextHead, ring_mean_features  # noqa: E402
from hestfm.evaluate import log_experiment  # noqa: E402
from hestfm.metrics import pearson_per_gene  # noqa: E402
from hestfm.splits import assert_no_leakage, inner_validation_mask, leave_one_group_out  # noqa: E402

DATASET = "phase2/xenium5k"


def attention_cv(samples, folds, enc, groups, y_kind, genes_idx, k=1, seed=0, max_epochs=60):
    assert_no_leakage(folds, groups)
    rows = []
    Ys = {s: transform_y(v.Y, y_kind)[:, genes_idx].tocsr() for s, v in samples.items()}
    coords = {s: v.coords for s, v in samples.items()}
    for fold in folds:
        tr, te = list(fold.train), list(fold.test)
        X = np.concatenate([samples[s].X[enc] for s in tr])
        sid = np.concatenate([[s] * len(samples[s].X[enc]) for s in tr])
        val = inner_validation_mask(sid, np.concatenate([coords[s] for s in tr]), seed=seed, groups=groups)
        h = AttentionContextHead(k=k, seed=seed, max_epochs=max_epochs).fit(X, coords, sid, sp.vstack([Ys[s] for s in tr]), val)
        Xte = np.concatenate([samples[s].X[enc] for s in te])
        sid_te = np.concatenate([[s] * len(samples[s].X[enc]) for s in te])
        P = h.predict(Xte, coords, sid_te)
        Yte = sp.vstack([Ys[s] for s in te]).toarray()
        rows.append(pd.DataFrame({"fold": fold.name, "encoder": f"{enc}_attn_k{k}", "gene_idx": genes_idx,
                                  "pearson": pearson_per_gene(Yte, P)}))
        print(f"  attention {fold.name}: epochs {h.epochs_}, mean r {np.nanmean(rows[-1].pearson):.4f}")
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--encoders", nargs="+", default=["uni2_h", "virchow2", "conch_v15"])
    ap.add_argument("--attention-encoder", default="uni2_h")
    ap.add_argument("--attention-genes", type=int, default=0, help="0 = all genes")
    ap.add_argument("--attention-epochs", type=int, default=60)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--y-kind", default="log1p")
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase3"))
    ap.add_argument("--phase2-table", default=str(ROOT / "reports" / "phase2" / "log1p" / "gene_table.csv"))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = load_config()["xenium5k_samples"]
    root = Path(a.root)
    groups = {s: g for s, g in groups.items() if (root / DATASET / "expression" / f"{s}.h5").exists()}
    encs = [e for e in a.encoders if all((root / DATASET / e / f"{s}.h5").exists() for s in groups)]
    folds = leave_one_group_out(groups)
    genes = common_genes(root, DATASET, list(groups))
    samples = load_sparse_samples(root, DATASET, encs, list(groups), genes)
    variants = []
    for s in samples.values():
        for e in encs:
            for k in a.ks:
                s.X[f"{e}_k{k}"] = ring_mean_features(s.X[e], s.coords, k)
    for e in encs:
        variants += [e] + [f"{e}_k{k}" for k in a.ks]
    gf, sg, _, fi = run_atlas(samples, folds, variants, groups, y_kind=a.y_kind, block=a.block)

    if a.attention_encoder in encs:
        rng = np.random.default_rng(0)
        gidx = np.arange(len(genes)) if a.attention_genes <= 0 else np.sort(rng.choice(len(genes), a.attention_genes, replace=False))
        gf = pd.concat([gf, attention_cv(samples, folds, a.attention_encoder, groups, a.y_kind, gidx,
                                          max_epochs=a.attention_epochs)])
    gname = np.asarray(genes)
    gf["gene"] = gname[gf.gene_idx.values]
    gf.to_csv(out / "gene_fold.csv.gz", index=False)

    wide = gf.pivot_table(index=["gene", "fold"], columns="encoder", values="pearson")
    rows = []
    for e in encs:
        for v in [c for c in wide.columns if c.startswith(e + "_")]:
            d = (wide[v] - wide[e]).rename("delta").reset_index()
            g = d.groupby("gene").delta.agg(mean_delta="mean", n_pos=lambda x: int((x > 0).sum()), n_folds="count")
            g["encoder"], g["variant"] = e, v
            rows.append(g.reset_index())
    gains = pd.concat(rows, ignore_index=True)
    table = gf.pivot_table(index="gene", columns="encoder", values="pearson", aggfunc="mean").add_prefix("r_")
    if Path(a.phase2_table).exists():
        feats = pd.read_csv(a.phase2_table, index_col=0)[["mean_log1p", "frac_zero", "morans_i", "tau", "top_type"]]
        table = table.join(feats)
    for v in gains.variant.unique():
        g = gains[gains.variant == v].set_index("gene")
        table[f"gain_{v}"] = g["mean_delta"]
        table[f"npos_{v}"] = g["n_pos"]
    table.to_csv(out / "context_gene_table.csv")

    summ = gf.groupby(["encoder", "fold"]).pearson.mean().groupby("encoder").agg(["mean", "std"]).rename(
        columns={"mean": "pearson_mean", "std": "fold_sd"})
    n_f = len(folds)
    gsum = gains.groupby("variant").apply(lambda d: pd.Series({
        "mean_gain": d.mean_delta.mean(), "median_gain": d.mean_delta.median(),
        "genes_gain_gt_0_05_all_folds": int(((d.mean_delta > 0.05) & (d.n_pos == d.n_folds)).sum()),
        "genes_gain_gt_0_05_majority": int(((d.mean_delta > 0.05) & (d.n_pos > n_f / 2)).sum()),
        "genes_loss_gt_0_05_majority": int(((d.mean_delta < -0.05) & (d.n_pos < n_f / 2)).sum())}), include_groups=False)
    summ.to_csv(out / "summary.csv")
    gsum.to_csv(out / "gain_summary.csv")
    if "morans_i" in table:
        from scipy.stats import spearmanr
        cors = []
        for v in gains.variant.unique():
            for f in ["morans_i", "mean_log1p", "tau"]:
                m = table[[f"gain_{v}", f]].dropna()
                if len(m) > 10:
                    rho, p = spearmanr(m.iloc[:, 0], m.iloc[:, 1])
                    cors.append({"variant": v, "feature": f, "spearman_rho": float(rho), "p": float(p), "n": len(m)})
        pd.DataFrame(cors).to_csv(out / "gain_vs_features.csv", index=False)
    print(summ.to_string()); print(gsum.to_string())
    if not a.no_log:
        for enc, r in summ.iterrows():
            log_experiment(enc, "xenium5k_context", "logo_5groups", "ridge_pergene_alpha_pca256" if "attn" not in enc else "attention_k1",
                           "pearson_mean", float(r.pearson_mean), notes=f"y={a.y_kind};genes={len(genes)}")
    return summ, gsum


if __name__ == "__main__":
    main()
