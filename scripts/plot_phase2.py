#!/usr/bin/env python
"""Static paper figures from the phase2 gene-level results table.

Usage: python scripts/plot_phase2.py --table gene_table.csv --out figures/
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

TEXT = "#52514e"
GRID = "#e6e5e0"
DARKLINE = "#52514e"  # dark gray
GRAY = "#8c8b85"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
BASELINES = ("composition", "ncells")
DERIVED = ("best_encoder", "encoder_sd", "image_minus_composition")  # summary columns in gene_table.csv, not encoders
BLUES1 = LinearSegmentedColormap.from_list("blues1", ["#f4f7fb", "#2a78d6"])


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    ax.tick_params(colors=TEXT, labelsize=8, length=2)
    ax.set_axisbelow(True)


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5)


def _save(fig, out_dir, stem, paths):
    png = os.path.join(out_dir, stem + ".png")
    pdf = os.path.join(out_dir, stem + ".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    paths.extend([png, pdf])


def _legend(target, handles, labels, loc="best"):
    return target.legend(handles, labels, loc=loc, frameon=False, fontsize=8,
                         labelcolor=TEXT, handlelength=1.2, handletextpad=0.5)


def _binned_median_line(x, y, nbins=20):
    """20 quantile bins of x; per-bin median of y."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if len(x) < nbins or np.ptp(x) == 0:
        return None
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            xs.append(np.median(x[m]))
            ys.append(np.median(y[m]))
    if len(xs) < 2:
        return None
    return np.asarray(xs), np.asarray(ys)


def _limits(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return (-1.0, 1.0)
    v = np.r_[x[ok], y[ok]]
    lo, hi = min(0.0, v.min()), v.max()   # tight data range (identity line stays on the diagonal)
    pad = 0.05 * (hi - lo or 1.0)
    return (lo - pad, hi + pad)


def _hexbin_panel(ax, x, y, xcol, enc):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if len(x):
        ax.hexbin(x, y, gridsize=30, mincnt=1, cmap=BLUES1, linewidths=0.2)
        line = _binned_median_line(x, y)
        if line is not None:
            ax.plot(line[0], line[1], color=DARKLINE, linewidth=1.2)
    _style_axes(ax)
    _grid(ax)
    ax.set_xlabel({"mean_log1p": "Mean log(1 + count)", "morans_i": "Moran's I (spatial autocorrelation)"}.get(xcol, xcol))
    ax.set_title(enc, color=TEXT, fontsize=8)
    _legend(ax, [Line2D([], [], color=DARKLINE, lw=1.2)], ["binned median"],
            loc="upper left")


def fig_r_distribution(df, encoders, palette, out_dir, paths):
    order = sorted(encoders, key=lambda e: np.nanmedian(df[f"r_{e}"]))
    cols = [palette[e] for e in order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(3.3 if n <= 2 else 6.8, 0.35 + 0.45 * n))
    data = [df[f"r_{e}"].dropna().to_numpy() for e in order]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                    tick_labels=order, showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.0},
                    whiskerprops={"color": TEXT, "linewidth": 0.8},
                    capprops={"color": TEXT, "linewidth": 0.8},
                    boxprops={"linewidth": 0.6})
    for box, c in zip(bp["boxes"], cols):
        box.set_facecolor(c)
        box.set_edgecolor("white")
    ax.axvline(0, color=TEXT, linestyle="--", linewidth=0.8, zorder=5)
    _style_axes(ax)
    _grid(ax)
    ax.set_xlabel("Per-gene Pearson r (held-out samples)")
    # encoders are named on the y axis (direct labels), so no separate legend box
    fig.tight_layout()
    _save(fig, out_dir, "fig_r_distribution", paths)


def fig_r_vs(df, encoders, palette, out_dir, paths, xcol, stem):
    n = len(encoders)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 if n == 1 else 6.8, 2.2 * nrows + 0.3),
                             sharey=True, squeeze=False)
    for k, enc in enumerate(encoders):
        ax = axes[k // ncols, k % ncols]
        _hexbin_panel(ax, df[xcol].to_numpy(), df[f"r_{enc}"].to_numpy(), xcol, enc)
        if k % ncols == 0:
            ax.set_ylabel("Per-gene Pearson r")
        else:
            ax.set_ylabel("")
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.tight_layout()
    _save(fig, out_dir, stem, paths)


def fig_r_by_celltype(df, best, color, out_dir, paths):
    r = df[f"r_{best}"].to_numpy()
    types = df["top_type"].fillna("unassigned").astype(str).to_numpy()
    groups = sorted({t for t in set(types) if (types == t).sum() >= 10},
                    key=lambda t: np.nanmedian(r[types == t]))
    fig, ax = plt.subplots(figsize=(3.3, 0.5 + 0.28 * max(len(groups), 1)))
    if groups:
        data = [r[types == t] for t in groups]
        bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6,
                        tick_labels=[f"{t} (n={len(d)})" for t, d in zip(groups, data)],
                        showfliers=False,
                        medianprops={"color": "white", "linewidth": 1.0},
                        whiskerprops={"color": TEXT, "linewidth": 0.8},
                        capprops={"color": TEXT, "linewidth": 0.8},
                        boxprops={"linewidth": 0.6})
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_edgecolor("white")
        _legend(ax, [Patch(facecolor=color)], [best], loc="lower right")
    ax.axvline(0, color=TEXT, linestyle="--", linewidth=0.8)
    _style_axes(ax)
    _grid(ax)
    ax.set_xlabel(f"Per-gene Pearson r ({best})")
    fig.tight_layout()
    _save(fig, out_dir, "fig_r_by_celltype", paths)


def fig_encoder_agreement(df, encoders, palette, out_dir, paths):
    n = len(encoders)
    if n < 2:
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        _style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "needs >= 2 encoders", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=TEXT)
        _save(fig, out_dir, "fig_encoder_agreement", paths)
        return
    fig, axes = plt.subplots(n, n, figsize=(6.8, 6.8), squeeze=False)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j
             and (n <= 3 or i < j)]
    for i, j in pairs:
        ax = axes[i][j]
        x = df[f"r_{encoders[j]}"].to_numpy()
        y = df[f"r_{encoders[i]}"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=3, alpha=0.3, color=palette[encoders[i]],
                   linewidths=0, edgecolors="none")
        lim = _limits(x, y)
        ax.plot(lim, lim, "--", color=GRAY, linewidth=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 1 else np.nan
        label = f"$\\rho$={rho:.2f}" if np.isfinite(rho) else "$\\rho$=n/a"
        ax.text(0.04, 0.94, label, transform=ax.transAxes, fontsize=7,
                color=GRAY, va="top")
        _style_axes(ax)
        _grid(ax)
        if i == n - 1:
            ax.set_xlabel(encoders[j])
        if j == 0:
            ax.set_ylabel(encoders[i])
    for k in range(n):
        axes[k][k].axis("off")
    _legend(fig, [Line2D([], [], marker="o", linestyle="", markersize=3,
                         color=palette[e]) for e in encoders],
            encoders, loc="upper center")
    fig.tight_layout()
    _save(fig, out_dir, "fig_encoder_agreement", paths)


def fig_image_vs_composition(df, best, color, out_dir, paths):
    y = df[f"r_{best}"].to_numpy()
    x = df["r_composition"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.scatter(x[ok], y[ok], s=3, alpha=0.3, color=color, linewidths=0,
               edgecolors="none")
    lim = _limits(x, y)
    ax.plot(lim, lim, "--", color=GRAY, linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    diff = np.where(ok, y - x, -np.inf)
    genes = df["gene"].astype(str).to_numpy()
    top = [i for i in np.argsort(diff)[::-1][:10] if np.isfinite(diff[i])]
    # labels in a column at the upper left, ordered by y, with thin leader lines (no overlaps)
    span = lim[1] - lim[0]
    for rank, i in enumerate(sorted(top, key=lambda j: -y[j])):
        ty = lim[1] - span * (0.04 + 0.055 * rank)
        tx = lim[0] + span * 0.04
        ax.annotate(genes[i], (x[i], y[i]), xytext=(tx, ty), fontsize=6.5, color=TEXT, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.4, shrinkA=0, shrinkB=1))
    _style_axes(ax)
    _grid(ax)
    ax.set_xlabel("r (composition baseline)")
    ax.set_ylabel(f"r ({best})")
    _legend(ax, [Line2D([], [], marker="o", linestyle="", markersize=3, color=color),
                 Line2D([], [], linestyle="--", color=GRAY)],
            [best, "identity"], loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig_image_vs_composition", paths)


def write_summary(df, encoders, out_dir, paths):
    rows = []
    for name in encoders + list(BASELINES):
        r = pd.to_numeric(df[f"r_{name}"], errors="coerce").dropna().astype(float)
        rows.append({
            "model": name,
            "n_genes": int(r.size),
            "mean_r": float(r.mean()) if r.size else np.nan,
            "median_r": float(r.median()) if r.size else np.nan,
            "frac_r_gt_0_3": float((r > 0.3).mean()) if r.size else np.nan,
            "frac_r_gt_0_5": float((r > 0.5).mean()) if r.size else np.nan,
        })
    out = os.path.join(out_dir, "fig_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    paths.append(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase2 gene-table paper figures")
    ap.add_argument("--table", required=True, help="path to gene_table.csv")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.table)
    paths = []

    encoders = [c[2:] for c in df.columns
                if c.startswith("r_") and c[2:] not in BASELINES + DERIVED]
    encoders.sort()  # fixed palette order by sorted encoder name
    # all-NaN encoder columns are skipped with a printed warning
    kept = []
    for enc in encoders:
        col = pd.to_numeric(df[f"r_{enc}"], errors="coerce")
        if col.notna().sum() == 0:
            print(f"warning: encoder '{enc}' has all-NaN r values; skipping")
        else:
            df[f"r_{enc}"] = col
            kept.append(enc)
    encoders = kept
    palette = {enc: PALETTE[i % len(PALETTE)] for i, enc in enumerate(encoders)}

    # best encoder = highest median r
    med = {e: np.nanmedian(df[f"r_{e}"]) for e in encoders
           if np.isfinite(np.nanmedian(df[f"r_{e}"]))}
    best = max(med, key=med.get) if med else (encoders[0] if encoders else None)

    fig_r_distribution(df, encoders, palette, args.out, paths)
    fig_r_vs(df, encoders, palette, args.out, paths, "mean_log1p", "fig_r_vs_mean")
    fig_r_vs(df, encoders, palette, args.out, paths, "morans_i", "fig_r_vs_morans")
    fig_r_by_celltype(df, best, palette.get(best, TEXT), args.out, paths)
    fig_encoder_agreement(df, encoders, palette, args.out, paths)
    if best is not None:
        fig_image_vs_composition(df, best, palette[best], args.out, paths)
    write_summary(df, encoders, args.out, paths)
    return paths


if __name__ == "__main__":
    main()
