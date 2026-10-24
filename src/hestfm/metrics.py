"""Per-gene Pearson correlation and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def pearson_per_gene(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r. NaN where either column is constant."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    yt = yt - yt.mean(0)
    yp = yp - yp.mean(0)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den <= 1e-12] = np.nan
    return r


def per_sample_pearson(y_true: np.ndarray, y_pred: np.ndarray, sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    sample_ids = np.asarray(sample_ids).astype(str)
    return {s: pearson_per_gene(y_true[sample_ids == s], y_pred[sample_ids == s]) for s in np.unique(sample_ids)}


def bootstrap_ci(per_sample_gene_r: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """95% CI of the gene-averaged Pearson, resampling held-out samples with replacement.

    `per_sample_gene_r` is (n_samples, n_genes). Returns (point, lo, hi). With one sample the CI is degenerate.
    """
    m = np.asarray(per_sample_gene_r, dtype=np.float64)
    if m.ndim == 1:
        m = m[None]
    point = float(np.nanmean(np.nanmean(m, axis=1)))
    if m.shape[0] < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    sample_means = np.nanmean(m, axis=1)
    boots = np.nanmean(sample_means[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def gene_bootstrap_ci(gene_r: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI of the mean over genes, resampling genes (describes gene-set uncertainty, not sample uncertainty)."""
    g = np.asarray(gene_r, dtype=np.float64)
    g = g[~np.isnan(g)]
    rng = np.random.default_rng(seed)
    boots = g[rng.integers(0, len(g), size=(n_boot, len(g)))].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(g.mean()), float(lo), float(hi)
