"""Phase 2 gene-level predictability atlas: memory-lean ridge over ~5k genes, several encoders at once.

Y is kept sparse (log1p keeps zeros at zero). For each fold and encoder the PCA/SVD is computed once; genes are then
processed in blocks. Per-gene alpha is chosen on an inner validation group drawn from the training samples only.
Cross-encoder agreement (correlation between two encoders' held-out predictions of the same gene) is computed
block by block, so predictions never need to be stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

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


def load_sparse_samples(root, dataset, encoders, sample_ids, genes, subdir="expression", extra_keys=(),
                        x_dtype=np.float32, keep_y=True):
    """Embeddings + expression on `genes`, rows matched on barcodes. `extra["total"]` is each spot's count total
    over `genes` (used for library-size normalisation when Y is later loaded in gene chunks). With keep_y=False
    the expression matrix is dropped after computing the totals (load chunks with `load_y`)."""
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
            X[e] = emb[[d[b] for b in keep_bc]].astype(x_dtype)
        if keep_bc is None:
            keep_bc = xbc
        rows = np.array([pos[b] for b in keep_bc])
        extra = {"total": np.asarray(M[rows].sum(1)).ravel().astype(np.float64)}
        with h5py.File(f"{root}/{dataset}/{subdir}/{s}.h5") as f:
            for k in extra_keys:
                if k in f:
                    extra[k] = f[k][:][rows]
        out[s] = SparseSample(s, X, M[rows] if keep_y else None, keep_bc, coords[rows], extra)
        del M
        release_memory()
    return out


def load_y(root, dataset, sample_id, genes, barcodes, subdir="expression") -> sp.csr_matrix:
    """Expression of `genes` for one sample, rows in the order of `barcodes`."""
    M, xbc, _, _ = read_expression(f"{root}/{dataset}/{subdir}/{sample_id}.h5", genes)
    pos = {b: i for i, b in enumerate(xbc)}
    return sp.csr_matrix(M)[[pos[b] for b in barcodes]]


def run_atlas_chunked(samples, folds, encoders, groups, root, dataset, genes, chunk=1000, subdir="expression",
                      per_chunk=None, fit_cache=None, **kw):
    """run_atlas over gene chunks loaded from disk one at a time (bounded memory). Fold PCA is recomputed per
    chunk; results are identical to a single run_atlas call. `per_chunk(samples, gene_names)` is called after
    each chunk is loaded (e.g. to compute gene features) and its DataFrames are concatenated."""
    import shutil
    import tempfile
    own_cache = fit_cache is None   # temp cache removed at the end; pass a directory to keep/reuse fits
    if own_cache:
        base = Path(root).parent / "tmp_fitcache"   # data/tmp_fitcache (gitignored), removed below
        base.mkdir(parents=True, exist_ok=True)
        fit_cache = tempfile.mkdtemp(dir=base)
    Path(fit_cache).mkdir(parents=True, exist_ok=True)
    kw["fit_cache"] = fit_cache
    parts, extra_parts, info = ([], [], []), [], None
    for c0 in range(0, len(genes), chunk):
        g = list(genes[c0:c0 + chunk])
        for s, v in samples.items():   # `dataset` may map sample -> dataset (e.g. Xenium + Visium in Phase 4)
            v.Y = load_y(root, dataset[s] if isinstance(dataset, dict) else dataset, s, g, v.barcodes, subdir)
        gf, sg, ag, fi = run_atlas(samples, folds, encoders, groups, **kw)
        for p, d in zip(parts, (gf, sg, ag)):
            if len(d):
                d["gene_idx"] = d["gene_idx"] + c0
                p.append(d)
        info = fi if info is None else info
        if per_chunk is not None:
            extra_parts.append(per_chunk(samples, g))
        print(f"  chunk genes {c0 + len(g)}/{len(genes)} done", flush=True)
    for v in samples.values():
        v.Y = None
    if own_cache:
        shutil.rmtree(fit_cache, ignore_errors=True)
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    out = (cat(parts[0]), cat(parts[1]), cat(parts[2]), info)
    return out + ((pd.concat(extra_parts),) if per_chunk is not None else ())


def transform_y(Y: sp.csr_matrix, kind: str = "log1p", totals: np.ndarray | None = None) -> sp.csr_matrix:
    """`totals`: per-spot count totals over the full gene set (needed for lognorm when Y is a gene chunk)."""
    Y = Y.astype(np.float32).tocsr(copy=True)
    if kind == "log1p":
        Y.data = np.log1p(Y.data)
    elif kind == "lognorm":   # library-size normalised to the median spot total, then log1p
        tot = np.asarray(Y.sum(1)).ravel() if totals is None else np.asarray(totals, dtype=np.float64)
        med = np.median(tot[tot > 0]) if (tot > 0).any() else 1.0
        Y = (sp.diags(med / np.maximum(tot, 1)) @ Y).tocsr()
        Y.data = np.log1p(Y.data)
    else:
        raise ValueError(kind)
    return Y


class StreamPCA:
    """Standardise + exact PCA with bounded memory: the D x D covariance is accumulated over row chunks of X
    (any float dtype, e.g. float16), so no full-size float32/float64 copy of X is ever made. Equivalent to
    StandardScaler -> PCA(k) (exact eigendecomposition instead of randomized SVD; component signs may differ,
    which does not change ridge predictions)."""

    def __init__(self, k: int, rows: int = 8192):
        self.k, self.rows = k, rows

    def _chunks(self, X):
        for i in range(0, len(X), self.rows):
            yield np.asarray(X[i:i + self.rows], dtype=np.float64)

    def fit(self, X):
        n, d = X.shape
        s1, s2 = np.zeros(d), np.zeros(d)
        for c in self._chunks(X):
            s1 += c.sum(0)
            s2 += (c * c).sum(0)
        self.mean_ = s1 / n
        var = np.maximum(s2 / n - self.mean_ ** 2, 0)
        self.scale_ = np.where(var > 0, np.sqrt(var), 1.0)
        C = np.zeros((d, d))
        for c in self._chunks(X):
            z = (c - self.mean_) / self.scale_
            C += z.T @ z
        w, v = np.linalg.eigh(C / max(n - 1, 1))
        self.components_ = v[:, ::-1][:, :self.k]
        return self

    def transform(self, X):
        return np.concatenate([((c - self.mean_) / self.scale_) @ self.components_ for c in self._chunks(X)])


def release_memory():
    """Return freed heap pages to the OS (glibc keeps them otherwise, inflating RSS after large temporaries)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class _FoldSVD:
    def __init__(self, X, pca_dim, seed):
        small = pca_dim is None or pca_dim >= min(X.shape)
        self.pre = _Preproc(None, seed).fit(np.asarray(X, np.float32)) if small else StreamPCA(pca_dim).fit(X)
        Z = self.pre.transform(X).astype(np.float64)
        self.zm = Z.mean(0)
        U, self.s, Vt = np.linalg.svd(Z - self.zm, full_matrices=False)
        del Z
        self.U, self.V = U.astype(np.float32), Vt.T   # U is the large factor (n x k): float32 halves memory
        self.Usum = U.sum(0)

    def project(self, X):
        return ((self.pre.transform(X).astype(np.float64) - self.zm) @ self.V).astype(np.float32)

    def uty(self, Yb):
        """U^T (Y - mean) for a dense or sparse block (sparse stays sparse: no dense copy of Y)."""
        ym = np.asarray(Yb.mean(0)).ravel()
        uy = (Yb.T @ self.U).T if sp.issparse(Yb) else self.U.T @ Yb
        return np.asarray(uy, dtype=np.float64) - np.outer(self.Usum, ym), ym

    FIELDS = ("U", "s", "Usum")

    def save(self, d: Path, prefix: str):
        for k in self.FIELDS:
            np.save(d / f"{prefix}_{k}.npy", getattr(self, k))

    @classmethod
    def load(cls, d: Path, prefix: str):
        """A fitted factorisation read back from disk (U memory-mapped); supports uty() and .s only."""
        obj = cls.__new__(cls)
        for k in cls.FIELDS:
            setattr(obj, k, np.load(d / f"{prefix}_{k}.npy", mmap_mode="r" if k == "U" else None))
        return obj


def _fit_key(fold, e, pca_dim, seed, X):
    import hashlib
    probe = np.asarray(X[:: max(1, len(X) // 50)], dtype=np.float64)
    h = hashlib.sha1(repr((fold.train, fold.test, e, pca_dim, seed, X.shape, float(probe.sum()),
                           float((probe ** 2).sum()))).encode()).hexdigest()[:16]
    return f"{e}_{h}"


def _fold_fits(Xtr, Xte, val, pdim, seed, cache: Path | None, key: str | None):
    """(sel, B_val, full, B_test) for one encoder/fold; cached on disk when `cache` is given, because the fits
    depend only on the embeddings and the fold, not on the genes (reused across gene chunks)."""
    if cache is not None and (cache / f"{key}_done").exists():
        return (_FoldSVD.load(cache, key + "_sel"), np.load(cache / f"{key}_Bval.npy"),
                _FoldSVD.load(cache, key + "_full"), np.load(cache / f"{key}_Bte.npy"))
    sel = _FoldSVD(Xtr[~val], pdim, seed)
    full = _FoldSVD(Xtr, pdim, seed)
    out = (sel, sel.project(Xtr[val]), full, full.project(Xte))
    if cache is not None:
        sel.save(cache, key + "_sel")
        full.save(cache, key + "_full")
        np.save(cache / f"{key}_Bval.npy", out[1])
        np.save(cache / f"{key}_Bte.npy", out[3])
        (cache / f"{key}_done").touch()
        return _fold_fits(Xtr, Xte, val, pdim, seed, cache, key)   # use the memory-mapped copy
    return out



def _ridge_pred(B, s, UTy, ym, alpha):
    """alpha: scalar or (g,) per gene."""
    f = s[:, None] / (s[:, None] ** 2 + np.broadcast_to(alpha, (UTy.shape[1],))[None, :])
    return B @ (f * UTy) + ym


def run_atlas(samples: dict[str, SparseSample], folds: list[Fold], encoders: list[str], groups: dict[str, str],
              y_kind: str = "log1p", block: int = 1000, pca_dim: int | None = 256, alphas=DEFAULT_ALPHAS,
              seed: int = 0, feature_fn=None, verbose=True, fit_cache: str | Path | None = None):
    """Returns (gene_fold, sample_gene, agreement, fold_info) DataFrames.

    `fit_cache`: directory where per-encoder/fold PCA+SVD fits are stored and reused (gene-chunked runs).

    `feature_fn(sample) -> (N, k)` adds a pseudo-encoder named "features" (e.g. cell-type composition) when given.
    """
    assert_no_leakage(folds, groups)
    ids = list(samples)
    G = samples[ids[0]].Y.shape[1]
    Ys = {s: transform_y(samples[s].Y, y_kind, samples[s].extra.get("total")) for s in ids}
    names = list(encoders) + (["features"] if feature_fn else [])

    def X_of(s, e):
        if e == "features":
            return feature_fn(samples[s])
        x = samples[s].X[e]
        return x() if callable(x) else x   # callables: features computed on demand (e.g. spatial context)

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
            Xtr = np.concatenate([X_of(s, e) for s in tr])   # keeps the stored dtype (float16 in lean mode)
            dim = Xtr.shape[1]
            Xte = np.concatenate([X_of(s, e) for s in te])
            pdim = pca_dim if e != "features" else None
            cache = Path(fit_cache) if (fit_cache is not None and e != "features") else None
            key = _fit_key(fold, e, pdim, seed, Xtr) if cache is not None else None
            fits[e] = _fold_fits(Xtr, Xte, val, pdim, seed, cache, key)
            del Xtr, Xte
            release_memory()
            fi.append({"fold": fold.name, "encoder": e, "n_train": int(len(val)), "n_val": int(val.sum()),
                       "n_test": int(len(fits[e][3])), "dim": int(dim)})
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


def shared_fit_cache(root) -> Path:
    """Persistent fold-fit cache next to the embeddings (data/tmp_fitcache/shared, gitignored). Keys hash the fold,
    encoder, seed, PCA size and the embeddings, so fits are reused safely across runs, target kinds and phases."""
    return Path(root).resolve().parent / "tmp_fitcache" / "shared"
