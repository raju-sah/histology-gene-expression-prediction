import numpy as np
from scipy.stats import pearsonr

from hestfm.metrics import bootstrap_ci, gene_bootstrap_ci, pearson_per_gene


def test_pearson_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(100, 5)), rng.normal(size=(100, 5))
    b[:, 0] += 2 * a[:, 0]
    r = pearson_per_gene(a, b)
    for k in range(5):
        assert np.isclose(r[k], pearsonr(a[:, k], b[:, k])[0])


def test_constant_column_is_nan():
    a = np.ones((10, 2)); a[:, 1] = np.arange(10)
    r = pearson_per_gene(a, np.arange(20).reshape(10, 2).astype(float))
    assert np.isnan(r[0]) and np.isclose(r[1], 1)


def test_bootstrap_ci_brackets_point():
    m = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    p, lo, hi = bootstrap_ci(m, n_boot=500)
    assert np.isclose(p, 0.35) and lo <= p <= hi
    p1, lo1, hi1 = bootstrap_ci(m[:1])
    assert np.isnan(lo1) and np.isnan(hi1)
    p2, lo2, hi2 = gene_bootstrap_ci(np.array([0.1, 0.2, np.nan, 0.3]))
    assert np.isclose(p2, 0.2) and lo2 <= p2 <= hi2
