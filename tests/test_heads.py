import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from hestfm.data import stack
from hestfm.heads import HESTRidge, MLPHead, RidgeCVHead
from hestfm.metrics import pearson_per_gene
from conftest import make_samples


def _xy(samples, ids):
    X, Y, sid, _, coords = stack([samples[s] for s in ids])
    return X, np.log1p(Y), sid, coords


def test_hest_ridge_matches_reference_protocol(samples):
    X, Y, _, _ = _xy(samples, ["S0", "S1"])
    Xt, _, _, _ = _xy(samples, ["S2"])
    h = HESTRidge(latent_dim=32).fit(X, Y)
    sc = StandardScaler().fit(X)
    pca = PCA(32, random_state=1).fit(sc.transform(X))
    ref = Ridge(solver="lsqr", alpha=100 / (32 * Y.shape[1]), fit_intercept=False, random_state=1, max_iter=1000)
    ref.fit(pca.transform(sc.transform(X)), Y)
    assert np.allclose(h.predict(Xt), ref.predict(pca.transform(sc.transform(Xt))), atol=1e-4)


def test_ridge_cv_learns_signal_and_matches_sklearn(samples):
    X, Y, sid, c = _xy(samples, ["S0", "S1", "S2"])
    Xt, Yt, _, _ = _xy(samples, ["S3"])
    h = RidgeCVHead(pca_dim=None).fit(X, Y, val_mask=sid == "S2")
    r = np.nanmean(pearson_per_gene(Yt, h.predict(Xt)))
    assert r > 0.5
    a = float(h.alpha_[0])
    sc = StandardScaler().fit(X)
    ref = Ridge(alpha=a).fit(sc.transform(X), Y)
    assert np.allclose(h.predict(Xt), ref.predict(sc.transform(Xt)), atol=1e-3)


def test_per_gene_alpha_shape(samples):
    X, Y, sid, _ = _xy(samples, ["S0", "S1"])
    h = RidgeCVHead(per_gene_alpha=True, pca_dim=16).fit(X, Y, val_mask=sid == "S1")
    assert h.alpha_.shape == (Y.shape[1],)


def test_no_signal_gives_chance():
    s = make_samples(signal=False, seed=3)
    X, Y, sid, _ = _xy(s, ["S0", "S1", "S2"])
    Xt, Yt, _, _ = _xy(s, ["S3"])
    h = RidgeCVHead(pca_dim=None).fit(X, Y, val_mask=sid == "S2")
    assert abs(np.nanmean(pearson_per_gene(Yt, h.predict(Xt)))) < 0.1


def test_mlp_learns_nonlinear():
    s = make_samples(nonlinear=True, seed=1, n=400)
    X, Y, sid, _ = _xy(s, ["S0", "S1", "S2"])
    Xt, Yt, _, _ = _xy(s, ["S3"])
    h = MLPHead(hidden=64, max_epochs=40, patience=5).fit(X, Y, val_mask=sid == "S2")
    assert np.nanmean(pearson_per_gene(Yt, h.predict(Xt))) > 0.4
    assert 1 <= h.epochs_ <= 40
