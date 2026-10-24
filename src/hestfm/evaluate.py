"""Cross-validated evaluation of a head on frozen embeddings, plus experiment logging."""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import ROOT
from .data import SampleData, load_sample, stack
from .metrics import bootstrap_ci, pearson_per_gene, per_sample_pearson
from .splits import Fold, assert_no_leakage, assert_spot_disjoint, inner_validation_mask


@dataclass
class CVResult:
    fold_gene: pd.DataFrame      # fold, gene, pearson (pooled over the fold's test rows)
    sample_gene: pd.DataFrame    # fold, sample, gene, pearson (within one held-out sample)
    folds: pd.DataFrame          # fold, n_train, n_test, mean_pearson, extra head info

    @property
    def pearson_mean(self) -> float:
        """HEST-Benchmark score: mean over folds of the gene-averaged Pearson."""
        return float(self.folds["mean_pearson"].mean())

    def gene_table(self) -> pd.DataFrame:
        """Per-gene mean (and std) over folds."""
        return self.fold_gene.groupby("gene")["pearson"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)

    def sample_ci(self, n_boot: int = 2000, seed: int = 0):
        m = self.sample_gene.pivot_table(index="sample", columns="gene", values="pearson")
        return bootstrap_ci(m.values, n_boot=n_boot, seed=seed)


def run_cv(samples: dict[str, SampleData], folds: list[Fold], head_factory: Callable[[], object],
           groups: dict[str, str] | None = None, transform: Callable = np.log1p,
           use_val: bool = True, seed: int = 0, keep_predictions: bool = False) -> CVResult:
    assert_no_leakage(folds, groups)
    genes = next(iter(samples.values())).genes
    fg, sg, fr, preds = [], [], [], {}
    for fold in folds:
        Xtr, Ytr, sid_tr, bc_tr, c_tr = stack([samples[s] for s in fold.train])
        Xte, Yte, sid_te, bc_te, _ = stack([samples[s] for s in fold.test])
        assert_spot_disjoint(np.char.add(np.char.add(sid_tr, "::"), bc_tr),
                             np.char.add(np.char.add(sid_te, "::"), bc_te))
        Ytr, Yte = transform(Ytr), transform(Yte)
        val = inner_validation_mask(sid_tr, c_tr, seed=seed, groups=groups) if use_val else None
        head = head_factory()
        head.fit(Xtr, Ytr, val_mask=val)
        P = head.predict(Xte)
        r = pearson_per_gene(Yte, P)
        fg.append(pd.DataFrame({"fold": fold.name, "gene": genes, "pearson": r}))
        for s, rs in per_sample_pearson(Yte, P, sid_te).items():
            sg.append(pd.DataFrame({"fold": fold.name, "sample": s, "gene": genes, "pearson": rs}))
        info = {"fold": fold.name, "n_train": len(Xtr), "n_test": len(Xte), "mean_pearson": float(np.nanmean(r))}
        if hasattr(head, "alpha_"):
            info["alpha_median"] = float(np.median(head.alpha_))
        if hasattr(head, "epochs_"):
            info["epochs"] = head.epochs_
        fr.append(info)
        if keep_predictions:
            preds[fold.name] = (sid_te, bc_te, Yte, P)
    res = CVResult(pd.concat(fg, ignore_index=True), pd.concat(sg, ignore_index=True), pd.DataFrame(fr))
    res.predictions = preds
    return res


def load_samples(root, dataset, encoder, sample_ids, genes=None) -> dict[str, SampleData]:
    return {s: load_sample(root, dataset, encoder, s, genes) for s in sample_ids}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_experiment(encoder, task, split, head, metric, value, n_test="", notes="",
                   path: str | Path = ROOT / "experiments.csv"):
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "commit", "encoder", "task", "split", "head", "metric", "value", "n_test", "notes"])
        w.writerow([dt.date.today().isoformat(), git_commit(), encoder, task, split, head, metric,
                    f"{value:.4f}" if isinstance(value, float) else value, n_test, notes])
