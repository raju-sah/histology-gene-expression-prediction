import numpy as np
import pandas as pd
import scipy.sparse as sp

from hestfm.atlas import SparseSample, gene_features, grid_spacing, morans_i, run_atlas
from hestfm.data import stack
from hestfm.heads import RidgeCVHead
from hestfm.metrics import pearson_per_gene
from hestfm.splits import inner_validation_mask, leave_one_group_out
from conftest import make_samples


def to_sparse(samples, encoders=("a", "b")):
    rng = np.random.default_rng(0)
    out = {}
    for k, s in samples.items():
        X = {"a": s.X, "b": s.X + rng.normal(0, 2.0, s.X.shape).astype(np.float32)}
        out[k] = SparseSample(k, {e: X[e] for e in encoders}, sp.csr_matrix(s.counts), s.barcodes, s.coords,
                              {"n_cells": rng.integers(1, 30, len(s.X))})
    return out


def test_atlas_matches_ridgecv_per_gene_alpha():
    s = make_samples(n_samples=4, g=23, d=40)
    ss = to_sparse(s)
    groups = {k: k for k in s}
    folds = leave_one_group_out(groups)
    gf, sg, ag, fi = run_atlas(ss, folds, ["a", "b"], groups, block=7, pca_dim=16, verbose=False)
    f = folds[0]
    Xtr, Ytr, sid, _, c = stack([s[x] for x in f.train])
    Xte, Yte, *_ = stack([s[x] for x in f.test])
    val = inner_validation_mask(sid, c, groups=groups)
    h = RidgeCVHead(pca_dim=16, per_gene_alpha=True).fit(Xtr, np.log1p(Ytr), val_mask=val)
    ref = pearson_per_gene(np.log1p(Yte), h.predict(Xte))
    ours = gf[(gf.fold == f.name) & (gf.encoder == "a")].sort_values("gene_idx").pearson.values
    assert np.allclose(ours, ref, atol=1e-4, equal_nan=True)
    assert set(ag.enc_a) == {"a"} and len(ag) == 23 * len(folds)
    assert (ag.pred_corr.dropna() > 0).mean() > 0.8   # two views of the same signal agree
    assert gf.groupby("encoder").pearson.mean()["a"] > gf.groupby("encoder").pearson.mean()["b"]
    assert set(sg["sample"]) == set(s)


def test_feature_pseudo_encoder():
    s = make_samples(n_samples=3, g=5)
    ss = to_sparse(s)
    groups = {k: k for k in s}
    gf, *_ = run_atlas(ss, leave_one_group_out(groups), ["a"], groups,
                       feature_fn=lambda x: np.log1p(x.extra["n_cells"][:, None]).astype(np.float32), verbose=False)
    assert set(gf.encoder) == {"a", "features"}
    assert abs(gf[gf.encoder == "features"].pearson.mean()) < 0.15   # random n_cells carries no signal


def test_morans_i_smooth_vs_random():
    xx, yy = np.meshgrid(np.arange(30), np.arange(30))
    coords = np.c_[xx.ravel(), yy.ravel()] * 100.0
    rng = np.random.default_rng(0)
    smooth = np.sin(coords[:, 0] / 600) + np.cos(coords[:, 1] / 500)
    Y = np.c_[smooth, rng.normal(size=len(coords))]
    I = morans_i(sp.csr_matrix(Y), coords, 1.5 * grid_spacing(coords))
    assert grid_spacing(coords) == 100.0
    assert I[0] > 0.8 and abs(I[1]) < 0.1


def test_gene_features_shapes():
    ss = to_sparse(make_samples(n_samples=2, g=6))
    df = gene_features(ss, [f"G{i}" for i in range(6)])
    assert list(df.columns) == ["mean_log1p", "var_log1p", "mean_counts", "frac_zero", "morans_i"]
    assert ((df.frac_zero >= 0) & (df.frac_zero <= 1)).all()


def test_chunked_matches_single_run(tmp_path):
    """Gene-chunked atlas (Y loaded from disk per chunk) equals one in-memory run, incl. lognorm totals."""
    from hestfm.atlas import common_genes, load_sparse_samples, run_atlas_chunked
    from fake_phase2 import make_fake_phase2
    samples_ids, genes = make_fake_phase2(tmp_path, encoders=("uni2_h",), n_genes=14)
    groups = {s: g for s, g in __import__("hestfm").load_config()["xenium5k_samples"].items()}
    folds = leave_one_group_out(groups)
    for kind in ("log1p", "lognorm"):
        full = load_sparse_samples(tmp_path, "phase2/xenium5k", ["uni2_h"], samples_ids, genes)
        gf1, *_ = run_atlas(full, folds, ["uni2_h"], groups, y_kind=kind, pca_dim=16, verbose=False)
        lean = load_sparse_samples(tmp_path, "phase2/xenium5k", ["uni2_h"], samples_ids, genes,
                                   x_dtype=np.float16, keep_y=False)
        for v in full.values():   # same float16 rounding of X as the lean path
            v.X["uni2_h"] = v.X["uni2_h"].astype(np.float16)
        gf1, *_ = run_atlas(full, folds, ["uni2_h"], groups, y_kind=kind, pca_dim=16, verbose=False)
        gf2, _, _, _, feats = run_atlas_chunked(lean, folds, ["uni2_h"], groups, tmp_path, "phase2/xenium5k", genes,
                                                chunk=5, y_kind=kind, pca_dim=16, verbose=False,
                                                per_chunk=lambda s, g: pd.DataFrame({"gene": g}))
        a = gf1.sort_values(["fold", "gene_idx"]).pearson.values
        b = gf2.sort_values(["fold", "gene_idx"]).pearson.values
        assert np.allclose(a, b, atol=1e-5, equal_nan=True), kind
        assert list(feats.gene) == genes


def test_stream_pca_matches_sklearn():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from hestfm.atlas import StreamPCA
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(3000, 40)) @ rng.normal(size=(40, 40))).astype(np.float16)
    sp_ = StreamPCA(10, max_elems=700 * 40).fit(X)
    Z1 = sp_.transform(X)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    Z2 = PCA(10, svd_solver="full").fit_transform(Xs)
    for j in range(10):   # equal up to sign
        assert min(np.abs(Z1[:, j] - Z2[:, j]).max(), np.abs(Z1[:, j] + Z2[:, j]).max()) < 1e-6 * np.abs(Z2).max() + 1e-6
