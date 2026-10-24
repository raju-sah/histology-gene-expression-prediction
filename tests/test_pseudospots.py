import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from hestfm.celltypes import annotate, composition, tau, type_profiles
from hestfm.pseudospots import TranscriptAggregator, orientation_check, pseudospot_counts, window_assignment


def fake_transcripts(n_cells=400, n_genes=30, seed=0, extent=4000):
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(0, extent, n_cells), rng.uniform(0, extent * 0.6, n_cells)
    genes = [f"G{i:02d}" for i in range(n_genes)]
    rows = []
    for c in range(n_cells):
        k = rng.integers(5, 60)
        rows.append(pd.DataFrame({"cell_id": f"c{c}-1", "feature_name": rng.choice(genes, k),
                                  "he_x": cx[c] + rng.normal(0, 3, k), "he_y": cy[c] + rng.normal(0, 3, k),
                                  "qv": 40.0, "is_gene": True}))
    df = pd.concat(rows, ignore_index=True)
    noise = pd.DataFrame({"cell_id": ["UNASSIGNED"] * 50 + ["c1-1"] * 50, "feature_name": "G00",
                          "he_x": 10.0, "he_y": 10.0, "qv": [40.0] * 50 + [5.0] * 50, "is_gene": True})
    ctrl = pd.DataFrame({"cell_id": "c2-1", "feature_name": "NegControlProbe_1", "he_x": 1.0, "he_y": 1.0,
                         "qv": 40.0, "is_gene": False}, index=range(20))
    df = pd.concat([df, noise, ctrl], ignore_index=True).sort_values("feature_name", kind="stable")  # gene-sorted like HEST
    return df.reset_index(drop=True), genes


def brute_force(df):
    d = df[(df.is_gene) & (df.qv >= 20) & (df.cell_id != "UNASSIGNED")]
    cells = d.groupby("cell_id")[["he_x", "he_y"]].mean()
    m = pd.crosstab(d.cell_id, d.feature_name)
    return cells, m


@pytest.mark.parametrize("chunk", [137, 10_000_000])
def test_streaming_equals_bruteforce(chunk):
    df, genes = fake_transcripts()
    agg = TranscriptAggregator()
    for i in range(0, len(df), chunk):
        agg.add(df.iloc[i:i + chunk])
    cm = agg.finalize()
    cells, m = brute_force(df)
    assert list(cm.genes) == sorted(m.columns)
    assert cm.counts.shape == m.shape
    # match cells via their centroids (row order is arbitrary)
    order = np.lexsort((cm.centroids[:, 1], cm.centroids[:, 0]))
    ref = cells.loc[m.index].values
    rorder = np.lexsort((ref[:, 1], ref[:, 0]))
    assert np.allclose(cm.centroids[order], ref[rorder])
    assert np.array_equal(cm.counts.toarray()[order], m.values[rorder])
    assert agg.n_kept == m.values.sum()


def test_window_assignment_matches_bruteforce_and_is_half_open():
    rng = np.random.default_rng(1)
    cent = rng.uniform(0, 1000, size=(500, 2))
    cent[0] = [100.0, 150.0]   # exactly on the right/bottom edge of patch 0 window -> excluded
    tl = np.array([[0.0, 50.0], [60.0, 60.0], [500, 500]])
    S = 100.0
    for scale in (0.5, 1.0, 2.0):
        A = window_assignment(cent, tl, S, scale).toarray()
        ctr = tl + S / 2
        ref = (np.abs(cent[None] - ctr[:, None]) < scale * S / 2).all(-1)
        assert np.array_equal(A.astype(bool), ref)
    assert window_assignment(cent, tl, S)[0, 0] == 0


def test_orientation_check_prefers_true_frame():
    rng = np.random.default_rng(2)
    tl = np.stack(np.meshgrid(np.arange(0, 3000, 100), np.arange(0, 500, 100)), -1).reshape(-1, 2).astype(float)
    cent = np.c_[rng.uniform(0, 3000, 2000), rng.uniform(0, 500, 2000)]
    o = orientation_check(cent, tl, 100.0)
    assert o["xy"] > 0.95 and o["swapped"] < 0.3


def test_pseudospot_sum_and_celltypes():
    genes = ["EPCAM", "KRT8", "COL1A1", "DCN", "CD3E", "X"]
    C = sp.csr_matrix(np.array([[30, 20, 0, 0, 0, 1],    # epithelial
                                [0, 0, 25, 30, 0, 1],    # fibroblast
                                [0, 0, 0, 0, 40, 0],     # t_nk
                                [1, 0, 0, 0, 0, 2]]))    # low quality
    labels, types, used = annotate(C, genes)
    assert list(labels) == ["epithelial", "fibroblast", "t_nk", "low_quality"]
    prof, n = type_profiles(C, labels, types)
    t = tau(prof)
    assert t[genes.index("EPCAM")] > 0.9
    A = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=np.float32))

    class CM:
        counts = C
    S = pseudospot_counts(A, CM)
    assert np.array_equal(S.toarray()[0], C.toarray()[:2].sum(0))
    assert np.array_equal(S.toarray()[1], C.toarray()[1:].sum(0))
    comp, cols = composition(A, labels, types)
    assert comp[1, cols.index("low_quality")] == 1 and comp[0, cols.index("epithelial")] == 1
