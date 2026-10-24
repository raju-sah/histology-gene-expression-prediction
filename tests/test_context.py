import numpy as np
import scipy.sparse as sp

from hestfm.context import AttentionContextHead, neighbor_lists, padded_neighbors, ring_mean_features
from hestfm.metrics import pearson_per_gene


def grid(nx, ny, s=365):
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.c_[xx.ravel(), yy.ravel()] * s


def test_ring_neighbors_and_mean():
    c = grid(5, 5)
    nb = neighbor_lists(c, 1)
    assert len(nb[12]) == 8 and len(nb[0]) == 3          # centre / corner
    assert len(neighbor_lists(c, 2)[12]) == 24
    X = np.arange(25, dtype=np.float32)[:, None]
    F = ring_mean_features(X, c, 1)
    assert F.shape == (25, 2) and np.isclose(F[12, 1], np.mean([6, 7, 8, 11, 13, 16, 17, 18]))
    lone = ring_mean_features(np.ones((1, 3), np.float32), np.zeros((1, 2)), 1)
    assert np.allclose(lone, 1)
    p = padded_neighbors(c, 1)
    assert p.shape == (25, 9) and (p[:, 0] == np.arange(25)).all() and (p[0] == -1).sum() == 5


W_SHARED = np.random.default_rng(42).normal(size=(16, 6))


def _smooth_sample(nx, ny, rng, d=16, g=6):
    c = grid(nx, ny)
    z = rng.normal(size=(len(c), d)).astype(np.float32)
    zr = ring_mean_features(z, c, 1)[:, d:]
    W = W_SHARED
    y = (zr @ W) + 0.2 * rng.normal(size=(len(c), g))   # expression depends on the NEIGHBOURHOOD, not the centre
    return c, z, y.astype(np.float32)


def test_attention_head_uses_context_and_not_other_samples():
    rng = np.random.default_rng(0)
    data = {s: _smooth_sample(20, 15, rng) for s in ["A", "B", "C"]}
    tr, te = ["A", "B"], ["C"]
    X = np.concatenate([data[s][1] for s in tr]); Y = np.concatenate([data[s][2] for s in tr])
    sid = np.concatenate([[s] * len(data[s][1]) for s in tr])
    coords = {s: data[s][0] for s in data}
    h = AttentionContextHead(k=1, pca_dim=16, d_attn=16, max_epochs=40, patience=6, lr=3e-3)
    h.fit(X, coords, sid, sp.csr_matrix(Y), val_mask=sid == "B")
    P = h.predict(data["C"][1], coords, np.array(["C"] * len(data["C"][1])))
    r_ctx = np.nanmean(pearson_per_gene(data["C"][2], P))
    from hestfm.heads import RidgeCVHead
    single = RidgeCVHead(pca_dim=None).fit(X, Y, val_mask=sid == "B")
    r_single = np.nanmean(pearson_per_gene(data["C"][2], single.predict(data["C"][1])))
    assert r_ctx > r_single + 0.3, (r_ctx, r_single)
    # neighbour indices never point into another sample's rows
    nbr = h._global_neighbors(coords, sid)
    owner = np.r_[np.zeros(300), np.ones(300)]
    valid = nbr >= 0
    assert (owner[np.where(valid, nbr, 0)][valid] == np.repeat(owner[:, None], nbr.shape[1], 1)[valid]).all()
