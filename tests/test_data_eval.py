import numpy as np
import pytest
import scipy.sparse as sp

from hestfm.data import load_sample, write_embeddings, write_expression
from hestfm.evaluate import run_cv
from hestfm.heads import RidgeCVHead
from hestfm.splits import Fold, LeakageError, leave_one_group_out
from conftest import make_samples


@pytest.mark.parametrize("sparse", [False, True])
def test_roundtrip_and_barcode_alignment(tmp_path, sparse):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    Y = rng.poisson(1, size=(50, 4)).astype(np.float32)
    bc = np.array([f"b{i}" for i in range(50)])
    perm = rng.permutation(50)
    write_embeddings(tmp_path / "ds/enc/S.h5", X, bc, np.zeros((50, 2)))
    write_expression(tmp_path / "ds/expression/S.h5", sp.csr_matrix(Y[perm]) if sparse else Y[perm], bc[perm],
                     ["a", "b", "c", "d"], np.zeros((50, 2)))
    s = load_sample(tmp_path, "ds", "enc", "S", genes=["c", "a"])
    assert np.allclose(s.X, X.astype(np.float16).astype(np.float32))
    assert np.allclose(s.counts, Y[:, [2, 0]])


def test_run_cv_signal_vs_shuffled():
    s = make_samples()
    groups = {k: k for k in s}
    folds = leave_one_group_out(groups)
    res = run_cv(s, folds, lambda: RidgeCVHead(pca_dim=None), groups=groups)
    assert res.pearson_mean > 0.5
    assert set(res.sample_gene["sample"]) == set(s)
    p, lo, hi = res.sample_ci(n_boot=200)
    assert lo <= p <= hi
    rng = np.random.default_rng(0)
    for v in s.values():
        v.counts = v.counts[rng.permutation(len(v.counts))]
    res0 = run_cv(s, folds, lambda: RidgeCVHead(pca_dim=None), groups=groups)
    assert abs(res0.pearson_mean) < 0.1


def test_run_cv_refuses_leaky_folds():
    s = make_samples(n_samples=3)
    with pytest.raises(LeakageError):
        run_cv(s, [Fold("bad", ("S0", "S1"), ("S1",))], lambda: RidgeCVHead())
    with pytest.raises(LeakageError):
        run_cv(s, [Fold("bad", ("S0",), ("S1",))], lambda: RidgeCVHead(), groups={"S0": "p", "S1": "p", "S2": "q"})
