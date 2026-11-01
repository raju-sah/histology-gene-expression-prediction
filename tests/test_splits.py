from pathlib import Path

import numpy as np
import pytest

from hestfm import ROOT, load_config
from hestfm.splits import (Fold, LeakageError, assert_no_leakage, assert_spot_disjoint, folds_from_hest_bench,
                           inner_validation_mask, leave_one_group_out)


def test_logo_groups_never_cross():
    groups = load_config()["xenium5k_samples"]
    folds = leave_one_group_out(groups)
    assert len(folds) == len(set(groups.values()))
    assert_no_leakage(folds, groups)
    for f in folds:
        assert {groups[s] for s in f.test}.isdisjoint({groups[s] for s in f.train})
        assert set(f.train) | set(f.test) == set(groups)


def test_sample_overlap_raises():
    with pytest.raises(LeakageError):
        assert_no_leakage([Fold("bad", ("A", "B"), ("B",))])


def test_group_overlap_raises():
    groups = {"A": "p1", "B": "p1", "C": "p2"}
    with pytest.raises(LeakageError):
        assert_no_leakage([Fold("bad", ("A", "C"), ("B",))], groups)


def test_spot_level_check():
    with pytest.raises(LeakageError):
        assert_spot_disjoint(np.array(["A::x", "B::y"]), np.array(["B::z"]))
    assert_spot_disjoint(np.array(["A::x"]), np.array(["B::x"]))


@pytest.mark.skipif(not (ROOT / "data/meta/hest-bench/IDC/splits").exists(), reason="bench splits not downloaded")
@pytest.mark.parametrize("task", ["IDC", "LUNG", "COAD", "PRAD", "CCRCC"])
def test_hest_bench_splits_are_sample_disjoint(task):
    folds = folds_from_hest_bench(ROOT / "data/meta/hest-bench" / task)
    assert_no_leakage(folds)
    tested = [s for f in folds for s in f.test]
    assert len(tested) == len(set(tested))  # every sample is tested once


def test_inner_validation_uses_whole_unit():
    sid = np.array(["A"] * 50 + ["B"] * 20 + ["C"] * 30)
    m = inner_validation_mask(sid, None, frac=0.2)
    assert set(sid[m]) == {"B"}


def test_inner_validation_single_sample_spatial_block():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(1000, 2))
    m = inner_validation_mask(np.array(["A"] * 1000), coords, frac=0.2)
    assert 0.15 < m.mean() < 0.25
    assert coords[m, 0].min() >= coords[~m, 0].max()
