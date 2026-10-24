"""Phase 2 gene-level predictability atlas: memory-lean ridge over ~5k genes, several encoders at once.

Y is kept sparse (log1p keeps zeros at zero). For each fold and encoder the PCA/SVD is computed once; genes are then
processed in blocks. Per-gene alpha is chosen on an inner validation group drawn from the training samples only.
Cross-encoder agreement (correlation between two encoders' held-out predictions of the same gene) is computed
block by block, so predictions never need to be stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import read_embeddings, read_expression
from .heads import DEFAULT_ALPHAS, _Preproc
from .metrics import pearson_per_gene
from .splits import Fold, assert_no_leakage, assert_spot_disjoint, inner_validation_mask


@dataclass
class SparseSample:
    sample_id: str
    X: dict            # encoder -> (N, D) float32
    Y: sp.csr_matrix   # (N, G) raw counts on the common gene set
    barcodes: np.ndarray
    coords: np.ndarray
    extra: dict


def common_genes(root, dataset, sample_ids, subdir="expression") -> list[str]:
    sets = [set(read_expression(f"{root}/{dataset}/{subdir}/{s}.h5")[3]) for s in sample_ids]
    return sorted(set.intersection(*sets))


def load_sparse_samples(root, dataset, encoders, sample_ids, genes, subdir="expression", extra_keys=()):
    import h5py
    out = {}
    for s in sample_ids:
        M, xbc, coords, _ = read_expression(f"{root}/{dataset}/{subdir}/{s}.h5", genes)
        M = sp.csr_matrix(M)
        pos = {b: i for i, b in enumerate(xbc)}
        X, keep_bc = {}, None
        for e in encoders:
            emb, ebc, _ = read_embeddings(f"{root}/{dataset}/{e}/{s}.h5")
            d = dict(zip(ebc, range(len(ebc))))
            if keep_bc is None:
                keep_bc = np.array([b for b in xbc if b in d])
            X[e] = emb[[d[b] for b in keep_bc]].astype(np.float32)
        rows = np.array([pos[b] for b in keep_bc])
        extra = {}
        with h5py.File(f"{root}/{dataset}/{subdir}/{s}.h5") as f:
            for k in extra_keys:
                if k in f:
                    extra[k] = f[k][:][rows]
        out[s] = SparseSample(s, X, M[rows], keep_bc, coords[rows], extra)
    return out


def transform_y(Y: sp.csr_matrix, kind: str = "log1p") -> sp.csr_matrix:
    Y = Y.astype(np.float32).tocsr(copy=True)
    if kind == "log1p":
        Y.data = np.log1p(Y.data)
    elif kind == "lognorm":   # library-size normalised to the median spot total, then log1p
        tot = np.asarray(Y.sum(1)).ravel()
        med = np.median(tot[tot > 0]) if (tot > 0).any() else 1.0
        Y = (sp.diags(med / np.maximum(tot, 1)) @ Y).tocsr()
        Y.data = np.log1p(Y.data)
    else:
        raise ValueError(kind)
    return Y


class _FoldSVD:
    def __init__(self, X, pca_dim, seed):
        self.pre = _Preproc(pca_dim, seed).fit(X)
        Z = self.pre.transform(X).astype(np.float64)
        self.zm = Z.mean(0)
        U, self.s, Vt = np.linalg.svd(Z - self.zm, full_matrices=False)
        self.U, self.V = U, Vt.T

    def project(self, X):
        return (self.pre.transform(X).astype(np.float64) - self.zm) @ self.V

    def uty(self, Yb):
        """U^T (Y - mean) for a dense or sparse block."""
        ym = np.asarray(Yb.mean(0)).ravel()
        uy = self.U.T @ (Yb.toarray() if sp.issparse(Yb) else Yb)
        return uy - np.outer(self.U.sum(0), ym), ym


def _ridge_pred(B, s, UTy, ym, alpha):
    """alpha: scalar or (g,) per gene."""
    f = s[:, None] / (s[:, None] ** 2 + np.broadcast_to(alpha, (UTy.shape[1],))[None, :])
    return B @ (f * UTy) + ym


def run_atlas(samples: dict[str, SparseSample], folds: list[Fold], encoders: list[str], groups: dict[str, str],
              y_kind: str = "log1p", block: int = 1000, pca_dim: int | None = 256, alphas=DEFAULT_ALPHAS,
              seed: int = 0, feature_fn=None, verbose=True):
    """Returns (gene_fold, sample_gene, agreement, fold_info) DataFrames.

    `feature_fn(sample) -> (N, k)` adds a pseudo-encoder named "features" (e.g. cell-type composition) when given.
    """
    assert_no_leakage(folds, groups)
    ids = list(samples)
    G = samples[ids[0]].Y.shape[1]
    Ys = {s: transform_y(samples[s].Y, y_kind) for s in ids}
    names = list(encoders) + (["features"] if feature_fn else [])

    def X_of(s, e):
        return feature_fn(samples[s]) if e == "features" else samples[s].X[e]

    gf, sg, ag, fi = [], [], [], []
    for fold in folds:
        tr, te = list(fold.train), list(fold.test)
        sid_tr = np.concatenate([[s] * samples[s].Y.shape[0] for s in tr])
        sid_te = np.concatenate([[s] * samples[s].Y.shape[0] for s in te])
        bc_tr = np.concatenate([samples[s].barcodes for s in tr])
        bc_te = np.concatenate([samples[s].barcodes for s in te])
        assert_spot_disjoint(np.char.add(np.char.add(sid_tr.astype(str), "::"), bc_tr.astype(str)),
                             np.char.add(np.char.add(sid_te.astype(str), "::"), bc_te.astype(str)))
        c_tr = np.concatenate([samples[s].coords for s in tr])
        val = inner_validation_mask(sid_tr, c_tr, seed=seed, groups=groups)
        Ytr = sp.vstack([Ys[s] for s in tr]).tocsc()
        Yte = sp.vstack([Ys[s] for s in te]).tocsc()
        Ytr_fit, Ytr_val = Ytr[np.where(~val)[0]], Ytr[np.where(val)[0]]
        fits = {}
        for e in names:
            Xtr = np.concatenate([X_of(s, e) for s in tr]).astype(np.float32)
            Xte = np.concatenate([X_of(s, e) for s in te]).astype(np.float32)
            pdim = pca_dim if e != "features" else None
            sel = _FoldSVD(Xtr[~val], pdim, seed)
            full = _FoldSVD(Xtr, pdim, seed)
            fits[e] = (sel, sel.project(Xtr[val]), full, full.project(Xte))
            fi.append({"fold": fold.name, "encoder": e, "n_train": len(Xtr), "n_val": int(val.sum()), "n_test": len(Xte),
                       "dim": Xtr.shape[1]})
        for b0 in range(0, G, block):
            gidx = np.arange(b0, min(G, b0 + block))
            yte = Yte[:, gidx].toarray()
            yval = Ytr_val[:, gidx].toarray()
            preds = {}
            for e in names:
                sel, Bval, full, Bte = fits[e]
                uty, ym = sel.uty(Ytr_fit[:, gidx])
                scores = np.stack([pearson_per_gene(yval, _ridge_pred(Bval, sel.s, uty, ym, a)) for a in alphas])
                best = np.asarray(alphas)[np.argmax(np.nan_to_num(scores, nan=-np.inf), axis=0)]
                uty, ym = full.uty(Ytr[:, gidx])
                P = _ridge_pred(Bte, full.s, uty, ym, best)
                preds[e] = P
                gf.append(pd.DataFrame({"fold": fold.name, "encoder": e, "gene_idx": gidx,
                                        "pearson": pearson_per_gene(yte, P), "alpha": best,
                                        "val_pearson": np.nanmax(scores, axis=0)}))
                for s in te:
                    m = sid_te == s
                    sg.append(pd.DataFrame({"fold": fold.name, "sample": s, "encoder": e, "gene_idx": gidx,
                                            "pearson": pearson_per_gene(yte[m], P[m])}))
            for a, b in combinations([e for e in names if e != "features"], 2):
                ag.append(pd.DataFrame({"fold": fold.name, "enc_a": a, "enc_b": b, "gene_idx": gidx,
                                        "pred_corr": pearson_per_gene(preds[a], preds[b])}))
            if verbose:
                print(f"  {fold.name}: genes {gidx[-1] + 1}/{G}", end="\r")
        if verbose:
            print()
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(gf), cat(sg), cat(ag), pd.DataFrame(fi)


def morans_i(Y: sp.csr_matrix, coords: np.ndarray, radius: float) -> np.ndarray:
    """Moran's I per gene with binary weights for spots within `radius` (Chebyshev) of each other."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(coords).query_pairs(r=radius, p=np.inf, output_type="ndarray")
    n = len(coords)
    if len(pairs) == 0:
        return np.full(Y.shape[1], np.nan)
    W = sp.coo_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                      shape=(n, n)).tocsr()
    Yd = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    Z = Yd - Yd.mean(0)
    num = (Z * (W @ Z)).sum(0)
    den = (Z ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n / W.sum()) * num / den


def grid_spacing(coords: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coords).query(coords, k=2, p=np.inf)
    return float(np.median(d[:, 1]))


def gene_features(samples: dict[str, SparseSample], genes: list[str], y_kind="log1p", block=1000) -> pd.DataFrame:
    """Descriptive per-gene statistics pooled over all spots (no model fitting, no use of test labels for tuning)."""
    rows = {}
    Yall = sp.vstack([transform_y(s.Y, y_kind) for s in samples.values()]).tocsc()
    raw = sp.vstack([s.Y for s in samples.values()]).tocsc()
    n = Yall.shape[0]
    mean = np.asarray(Yall.mean(0)).ravel()
    sq = np.asarray(Yall.multiply(Yall).mean(0)).ravel()
    rows["mean_log1p"] = mean
    rows["var_log1p"] = sq - mean ** 2
    rows["mean_counts"] = np.asarray(raw.mean(0)).ravel()
    rows["frac_zero"] = 1 - np.diff(raw.indptr) / n
    mi = []
    for s in samples.values():
        Ys = transform_y(s.Y, y_kind)
        r = 1.5 * grid_spacing(s.coords)
        mi.append(np.concatenate([morans_i(Ys[:, b0:b0 + block], s.coords, r) for b0 in range(0, len(genes), block)]))
    rows["morans_i"] = np.nanmean(np.vstack(mi), axis=0)
    return pd.DataFrame(rows, index=pd.Index(genes, name="gene"))
