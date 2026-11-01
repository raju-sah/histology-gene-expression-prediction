"""Phase 2: Xenium 5k gene predictability atlas.

    .venv/bin/python scripts/run_phase2_atlas.py [--root data/embeddings] [--y-kind log1p lognorm]

Leave-one-group-out over the 6 Xenium 5k samples (groups in configs/project.yaml), per-gene-alpha ridge on PCA-256
of each frozen encoder, all common genes. Baselines on the same folds: measured cell-type composition of the
pseudo-spot ("composition", an upper bound for what composition alone explains) and the number of cells ("ncells").
A label-permutation control checks for leakage. Outputs go to reports/phase2/<y_kind>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.atlas import (shared_fit_cache, common_genes, gene_features, load_sparse_samples, load_y, run_atlas,  # noqa: E402
                          run_atlas_chunked)
from hestfm.celltypes import tau  # noqa: E402
from hestfm.evaluate import log_experiment  # noqa: E402
from hestfm.metrics import bootstrap_ci  # noqa: E402
from hestfm.splits import leave_one_group_out  # noqa: E402

DATASET = "phase2/xenium5k"
ENCODERS = ["uni2_h", "virchow2", "conch_v15", "hoptimus1"]
TOO_GOOD = 0.6   # mean per-gene r over ~5k genes above this would be implausible -> investigate leakage


def pooled_celltypes(root, samples, genes):
    """Cell-count-weighted mean type profiles over samples -> tau, top type per gene."""
    acc, n_tot, types = None, None, None
    for s in samples:
        with h5py.File(f"{root}/{DATASET}/cells/{s}.h5") as f:
            g = f["genes"][:].astype(str)
            pos = {x: i for i, x in enumerate(g)}
            P = f["type_profiles"][:][:, [pos[x] for x in genes]]
            n = f["n_per_type"][:].astype(float)
            t = list(f["types"][:].astype(str))
        if types is None:
            types, acc, n_tot = t, np.zeros_like(P, dtype=float), np.zeros(len(t))
        assert t == types, "cell-type order differs between samples"
        acc += P * n[:, None]
        n_tot += n
    prof = acc / np.maximum(n_tot, 1)[:, None]
    top = np.array(types)[prof.argmax(0)]
    top[prof.max(0) <= 0] = "unassigned"
    return pd.DataFrame({"tau": tau(prof), "top_type": top}, index=pd.Index(genes, name="gene")), \
        pd.DataFrame(prof.T, index=genes, columns=types)


def summarize(gf, sg, genes, model):
    g = gf[gf.encoder == model]
    per_fold = g.groupby("fold").pearson.mean()
    m = sg[sg.encoder == model].pivot_table(index="sample", columns="gene_idx", values="pearson")
    p, lo, hi = bootstrap_ci(m.values)
    per_gene = g.groupby("gene_idx").pearson.mean()
    return {"model": model, "pearson_mean": float(per_fold.mean()), "ci_lo": lo, "ci_hi": hi,
            "per_sample_mean": p, "median_gene_r": float(per_gene.median()),
            "frac_gt_0_3": float((per_gene > 0.3).mean()), "frac_gt_0_5": float((per_gene > 0.5).mean()),
            "n_genes_evaluable": int(per_gene.notna().sum()), "folds": ";".join(f"{x:.4f}" for x in per_fold)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--y-kind", nargs="+", default=["log1p", "lognorm"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase2"))
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--null-genes", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=250, help="genes loaded per pass (memory bound)")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)

    groups = load_config()["xenium5k_samples"]
    root = Path(a.root)
    present = [s for s in groups if (root / DATASET / "expression" / f"{s}.h5").exists()]
    if len(present) < len(groups):
        print("WARNING: missing samples:", sorted(set(groups) - set(present)))
    groups = {s: groups[s] for s in present}
    encs = [e for e in a.encoders if all((root / DATASET / e / f"{s}.h5").exists() for s in present)]
    print("samples", present, "| encoders", encs)
    folds = leave_one_group_out(groups)
    genes = common_genes(root, DATASET, present)
    print(len(genes), "common genes")
    # Memory-lean: embeddings kept as float16, expression read from disk in gene chunks (see run_atlas_chunked).
    samples = load_sparse_samples(root, DATASET, encs, present, genes, extra_keys=("n_cells", "composition"),
                                  x_dtype=np.float16, keep_y=False)

    feats = None
    cell_df, prof = pooled_celltypes(root, present, genes)
    flags, all_summ = [], []
    for yk in a.y_kind:
        out = Path(a.out) / yk
        out.mkdir(parents=True, exist_ok=True)
        per_chunk = (lambda s, g: gene_features(s, g)) if feats is None else None
        res = run_atlas_chunked(samples, folds, encs, groups, root, DATASET, genes, chunk=a.chunk, fit_cache=shared_fit_cache(root),
                                y_kind=yk, block=a.block, per_chunk=per_chunk)
        gf, sg, ag, fi = res[:4]
        if per_chunk is not None:
            feats = res[4]
        bl = {
            "composition": lambda s: np.log1p(s.extra["composition"].astype(np.float32)),
            "ncells": lambda s: np.log1p(s.extra["n_cells"].astype(np.float32))[:, None],
            "composition_ncells": lambda s: np.column_stack([
                np.log1p(s.extra["composition"].astype(np.float32)),
                np.log1p(s.extra["n_cells"].astype(np.float32)),
            ]),
        }
        for name, fn in bl.items():
            g2, s2, _, f2 = run_atlas_chunked(samples, folds, [], groups, root, DATASET, genes, chunk=a.chunk, fit_cache=shared_fit_cache(root),
                                              y_kind=yk, block=a.block, feature_fn=fn, verbose=False)
            for d in (g2, s2, f2):
                d["encoder"] = d["encoder"].replace("features", name)
            gf, sg, fi = pd.concat([gf, g2]), pd.concat([sg, s2]), pd.concat([fi, f2])
        # permutation control: shuffle expression rows within each sample (breaks image<->expression link)
        rng = np.random.default_rng(0)
        sub = sorted(rng.choice(len(genes), size=min(a.null_genes, len(genes)), replace=False))
        null = {}
        for s, v in samples.items():
            Y = load_y(root, DATASET, s, [genes[i] for i in sub], v.barcodes)
            perm = rng.permutation(Y.shape[0])
            null[s] = type(v)(s, {encs[0]: v.X[encs[0]]}, Y[perm], v.barcodes, v.coords, {"total": v.extra["total"][perm]})
        gn, *_ = run_atlas(null, folds, [encs[0]], groups, y_kind=yk, block=a.block, verbose=False)
        null_r = float(gn.groupby("fold").pearson.mean().mean())
        del null

        gname = np.asarray(genes)
        for d in (gf, sg, ag):
            d["gene"] = gname[d["gene_idx"].values]
        gf.to_csv(out / "gene_fold.csv.gz", index=False)
        sg.to_csv(out / "sample_gene.csv.gz", index=False)
        ag.to_csv(out / "agreement.csv.gz", index=False)
        fi.to_csv(out / "fold_info.csv", index=False)

        table = gf.pivot_table(index="gene", columns="encoder", values="pearson", aggfunc="mean")
        table.columns = [f"r_{c}" for c in table.columns]
        nf = gf[gf.encoder == encs[0]].groupby("gene").pearson.count().rename("n_folds_evaluable")
        agm = ag.groupby("gene").pred_corr.mean().rename("agreement_mean")
        rcols = [f"r_{e}" for e in encs]
        table = table.join(nf).join(agm).join(feats).join(cell_df)
        table["r_best_encoder"] = table[rcols].max(1)
        table["r_encoder_sd"] = table[rcols].std(1)
        table["r_image_minus_composition"] = table[rcols].mean(1) - table["r_composition"]
        table["r_image_minus_ncells"] = table[rcols].mean(1) - table["r_ncells"]
        table["r_image_minus_comb"] = table[rcols].mean(1) - table["r_composition_ncells"]
        table.sort_values(rcols[0], ascending=False).to_csv(out / "gene_table.csv")
        prof.to_csv(out / "celltype_profiles.csv")

        # Step 3: Per-group performance table
        organs = load_config().get("xenium5k_organ", {})
        group_rows = []
        for fold in folds:
            test_sample = fold.test[0] if fold.test else ""
            organ = organs.get(test_sample, fold.name.replace("logo_", ""))
            row = {"fold": fold.name, "test_group": fold.meta["test_group"], "organ": organ}
            for m in encs + ["composition", "ncells", "composition_ncells"]:
                sub_gf = gf[(gf.fold == fold.name) & (gf.encoder == m)]
                row[f"r_{m}"] = float(sub_gf.pearson.mean()) if len(sub_gf) else np.nan
            group_rows.append(row)
        pg_df = pd.DataFrame(group_rows)
        pg_df.to_csv(out / "per_group_summary.csv", index=False)

        # Step 3: Paired statistics (Wilcoxon over genes, group bootstrap, win rate)
        from scipy.stats import wilcoxon
        paired_rows = []
        baselines = ["ncells", "composition", "composition_ncells"]
        for enc in encs:
            enc_gf = gf[gf.encoder == enc].groupby(["fold", "gene"])["pearson"].mean().unstack("fold")
            for b in baselines:
                base_gf = gf[gf.encoder == b].groupby(["fold", "gene"])["pearson"].mean().unstack("fold")
                common_idx = enc_gf.index.intersection(base_gf.index)
                rg_enc = enc_gf.loc[common_idx].mean(1)
                rg_base = base_gf.loc[common_idx].mean(1)
                diff_g = rg_enc - rg_base
                diff_clean = diff_g.dropna()

                try:
                    w_res = wilcoxon(rg_enc.loc[diff_clean.index], rg_base.loc[diff_clean.index])
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, w_p = np.nan, np.nan

                fold_diffs = (enc_gf.loc[common_idx].mean(0) - base_gf.loc[common_idx].mean(0)).values
                rng = np.random.default_rng(0)
                b_diffs = [rng.choice(fold_diffs, size=len(fold_diffs), replace=True).mean() for _ in range(2000)]
                ci_lo, ci_hi = float(np.percentile(b_diffs, 2.5)), float(np.percentile(b_diffs, 97.5))

                paired_rows.append({
                    "encoder": enc, "baseline": b,
                    "mean_diff": float(diff_clean.mean()),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                    "frac_win": float((diff_clean > 0).mean()),
                    "frac_win_gt_0_05": float((diff_clean > 0.05).mean()),
                    "n_genes": len(diff_clean),
                })
        paired_df = pd.DataFrame(paired_rows)
        paired_df.to_csv(out / "paired_stats.csv", index=False)

        summ = [summarize(gf, sg, genes, m) for m in encs + ["composition", "ncells", "composition_ncells"]]
        summ = pd.DataFrame(summ).assign(y_kind=yk, null_permuted_r=null_r)
        summ.to_csv(out / "summary.csv", index=False)
        all_summ.append(summ)
        print(summ[["model", "pearson_mean", "ci_lo", "ci_hi", "median_gene_r", "frac_gt_0_3"]].to_string())
        print(f"permutation control ({encs[0]}, {len(sub)} genes): mean r = {null_r:.4f}")
        if abs(null_r) > 0.05:
            flags.append(f"{yk}: permutation control r={null_r:.3f} (should be ~0) -> leakage suspected")
        for r in summ.itertuples():
            if r.pearson_mean > TOO_GOOD:
                flags.append(f"{yk}/{r.model}: mean r {r.pearson_mean:.3f} implausibly high -> investigate leakage")
            if not a.no_log:
                log_experiment(r.model, "xenium5k_atlas", f"logo_{len(folds)}groups", "ridge_pergene_alpha_pca256", "pearson_mean",
                               r.pearson_mean, n_test=int(fi[fi.encoder == r.model].n_test.sum()),
                               notes=f"y={yk};genes={len(genes)};ci95=[{r.ci_lo:.3f},{r.ci_hi:.3f}];null={null_r:.3f}")
    json.dump({"flags": flags, "samples": present, "encoders": encs, "n_genes": len(genes)},
              open(Path(a.out) / "run_info.json", "w"), indent=2)
    print("\n".join(["FLAGS:"] + flags) if flags else "no flags")
    return pd.concat(all_summ), flags


if __name__ == "__main__":
    main()
