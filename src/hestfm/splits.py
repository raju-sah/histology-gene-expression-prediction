"""Sample/patient-level splits and leakage checks.

A fold is a set of training sample IDs and a disjoint set of test sample IDs. Samples that belong to the same
group (patient, or a conservative proxy such as "same TMA study") must never be split across train and test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    meta: dict = field(default_factory=dict, compare=False)


class LeakageError(AssertionError):
    pass


def folds_from_hest_bench(task_dir: str | Path) -> list[Fold]:
    """Read HEST-Benchmark `splits/train_i.csv` / `test_i.csv`."""
    split_dir = Path(task_dir) / "splits"
    n = len(list(split_dir.glob("test_*.csv")))
    if n == 0:
        raise FileNotFoundError(f"no splits in {split_dir}")
    folds = []
    for i in range(n):
        tr = pd.read_csv(split_dir / f"train_{i}.csv")["sample_id"].astype(str).tolist()
        te = pd.read_csv(split_dir / f"test_{i}.csv")["sample_id"].astype(str).tolist()
        folds.append(Fold(f"split{i}", tuple(tr), tuple(te)))
    return folds


def leave_one_group_out(sample_to_group: dict[str, str]) -> list[Fold]:
    groups = sorted(set(sample_to_group.values()))
    folds = []
    for g in groups:
        test = tuple(sorted(s for s, gg in sample_to_group.items() if gg == g))
        train = tuple(sorted(s for s, gg in sample_to_group.items() if gg != g))
        folds.append(Fold(f"logo_{g}", train, test, {"test_group": g}))
    return folds


def assert_no_leakage(folds: list[Fold], sample_to_group: dict[str, str] | None = None) -> None:
    """Raise LeakageError if any sample, or any group, appears on both sides of a fold."""
    for f in folds:
        if not f.train or not f.test:
            raise LeakageError(f"{f.name}: empty train or test")
        both = set(f.train) & set(f.test)
        if both:
            raise LeakageError(f"{f.name}: samples in train and test: {sorted(both)}")
        if sample_to_group is not None:
            missing = (set(f.train) | set(f.test)) - set(sample_to_group)
            if missing:
                raise LeakageError(f"{f.name}: samples without group: {sorted(missing)}")
            gtr = {sample_to_group[s] for s in f.train}
            gte = {sample_to_group[s] for s in f.test}
            if gtr & gte:
                raise LeakageError(f"{f.name}: groups in train and test: {sorted(gtr & gte)}")


def assert_spot_disjoint(train_keys: np.ndarray, test_keys: np.ndarray) -> None:
    """Row-level check on (sample_id, barcode) keys actually fed to a model."""
    tr_samples = {str(k).split("::")[0] for k in train_keys}
    te_samples = {str(k).split("::")[0] for k in test_keys}
    if tr_samples & te_samples:
        raise LeakageError(f"rows of samples {sorted(tr_samples & te_samples)} in both train and test")


def inner_validation_mask(sample_ids: np.ndarray, coords: np.ndarray | None, seed: int = 0,
                          frac: float = 0.2, groups: dict[str, str] | None = None) -> np.ndarray:
    """Boolean mask of rows used for validation inside a training set.

    With >= 2 training groups, one whole group (or sample) is held out (the one closest to `frac` of rows).
    With a single training sample, a contiguous spatial block (vertical stripe of x) is held out to limit the
    optimism from spatial autocorrelation. Only training rows are ever used.
    """
    sample_ids = np.asarray(sample_ids).astype(str)
    units = np.array([groups.get(s, s) for s in sample_ids]) if groups else sample_ids
    uniq = np.unique(units)
    if len(uniq) >= 2:
        rng = np.random.default_rng(seed)
        sizes = {u: (units == u).mean() for u in uniq}
        order = sorted(uniq, key=lambda u: (abs(sizes[u] - frac), rng.random()))
        return units == order[0]
    if coords is None:
        raise ValueError("single training sample: coords needed for a spatial validation block")
    x = np.asarray(coords)[:, 0].astype(float)
    lo, hi = np.quantile(x, [1 - frac, 1.0])
    return x >= lo
