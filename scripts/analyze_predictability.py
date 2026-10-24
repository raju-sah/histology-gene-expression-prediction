"""Phase 2: what explains per-gene predictability?

    .venv/bin/python scripts/analyze_predictability.py [--table reports/phase2/log1p/gene_table.csv]

Response: per-gene Pearson r of the best-median encoder (and each encoder). Explanatory features: mean expression,
variance, sparsity, Moran's I, cell-type specificity (tau), cross-encoder agreement, plus the composition baseline.
Reports Spearman rho, partial Spearman controlling for mean expression, and a standardised OLS with HC3 errors.
Writes explain_*.csv and explain.md next to the table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT  # noqa: E402

FEATURES = ["mean_log1p", "var_log1p", "frac_zero", "morans_i", "tau", "agreement_mean"]
BASELINES = {"r_composition", "r_ncells"}


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(rx), res(ry))


def explain(table: pd.DataFrame, response: str):
    d = table.dropna(subset=[response])
    rows = []
    for f in FEATURES + ["r_composition"]:
        if f not in d or d[f].notna().sum() < 10:
            continue
        m = d[list(dict.fromkeys([f, response, "mean_log1p"]))].dropna()
        rho, p = stats.spearmanr(m[f], m[response])
        if f != "mean_log1p":
            prho, pp = partial_spearman(m[f].values, m[response].values, m["mean_log1p"].values)
        else:
            prho, pp = np.nan, np.nan
        rows.append({"feature": f, "n": len(m), "spearman_rho": float(rho), "p": float(p),
                     "partial_rho_given_mean": float(prho), "partial_p": float(pp)})
    sp_df = pd.DataFrame(rows)

    import statsmodels.api as sm
    feats = [f for f in FEATURES if f in d and d[f].notna().sum() >= 10 and f != "var_log1p"]  # var ~ mean: collinear
    m = d[feats + [response]].dropna()
    X = (m[feats] - m[feats].mean()) / m[feats].std()
    ols = sm.OLS(m[response], sm.add_constant(X)).fit(cov_type="HC3")
    ols_df = pd.DataFrame({"coef_std": ols.params, "se_hc3": ols.bse, "p": ols.pvalues}).drop("const")
    ols_df["r2_model"] = ols.rsquared
    # leave-one-feature-out drop in R^2 (unique contribution)
    for f in feats:
        sub = [g for g in feats if g != f]
        r2 = sm.OLS(m[response], sm.add_constant(X[sub])).fit().rsquared if sub else 0.0
        ols_df.loc[f, "delta_r2_unique"] = ols.rsquared - r2
    return sp_df, ols_df.reset_index(names="feature"), len(m)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "reports" / "phase2" / "log1p" / "gene_table.csv"))
    a = ap.parse_args(argv)
    t = pd.read_csv(a.table, index_col=0)
    out = Path(a.table).parent
    encs = [c for c in t.columns if c.startswith("r_") and c not in BASELINES
            and c not in {"r_best_encoder", "r_encoder_sd", "r_image_minus_composition"}]
    best = max(encs, key=lambda c: t[c].median())
    lines = [f"# What explains predictability ({Path(a.table).parent.name} targets)", "",
             f"Genes: {len(t)}; encoders: {', '.join(e[2:] for e in encs)}; response for the main model: `{best}` (highest median r).", ""]
    all_sp, all_ols = [], []
    for resp in [best] + [e for e in encs if e != best]:
        sp_df, ols_df, n = explain(t, resp)
        all_sp.append(sp_df.assign(response=resp))
        all_ols.append(ols_df.assign(response=resp, n=n))
    sp_all, ols_all = pd.concat(all_sp), pd.concat(all_ols)
    sp_all.to_csv(out / "explain_spearman.csv", index=False)
    ols_all.to_csv(out / "explain_ols.csv", index=False)
    by_type = t.groupby(t["top_type"].fillna("unassigned"))[encs + ["r_composition"]].agg(["median", "count"])
    by_type.to_csv(out / "explain_by_celltype.csv")

    fmt = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else "n/a"
    lines += ["## Spearman correlation with per-gene r (best encoder)", "",
              "| feature | n | rho | p | partial rho (given mean expr.) |", "|---|---|---|---|---|"]
    for r in all_sp[0].itertuples():
        lines.append(f"| {r.feature} | {r.n} | {fmt(r.spearman_rho)} | {r.p:.1e} | {fmt(r.partial_rho_given_mean)} |")
    o = all_ols[0]
    lines += ["", f"## Standardised OLS (HC3), R^2 = {o.r2_model.iloc[0]:.3f}, n = {o.n.iloc[0]}", "",
              "| feature | coef (per SD) | SE | p | unique delta R^2 |", "|---|---|---|---|---|"]
    for r in o.itertuples():
        lines.append(f"| {r.feature} | {fmt(r.coef_std)} | {fmt(r.se_hc3)} | {r.p:.1e} | {fmt(r.delta_r2_unique)} |")
    lines += ["", "## Median r by dominant cell type", "", by_type.round(3).to_markdown(), ""]
    (out / "explain.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return sp_all, ols_all


if __name__ == "__main__":
    main()
