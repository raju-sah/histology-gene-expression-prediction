"""Smoke test of notebooks/colab_phase1_benchmark.ipynb on fake HEST-bench files (CPU, dummy encoders)."""
import json
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from hestfm import ROOT
from hestfm.encoders import IMAGENET, Encoder
from notebook_harness import run_notebook

sys.path.insert(0, str(ROOT / "src" / "hestfm"))  # notebook does `from encoders import ...`
NB = ROOT / "notebooks" / "colab_phase1_benchmark.ipynb"


def _fake_bench(root, task, samples, genes, n=300):
    rng = np.random.default_rng(0)
    for i in range(len(samples)):
        te = [samples[i]]
        tr = [s for s in samples if s != samples[i]]
        os.makedirs(root / task / "splits", exist_ok=True)
        for name, ids in (("train", tr), ("test", te)):
            pd.DataFrame({"sample_id": ids, "patches_path": [f"patches/{s}.h5" for s in ids],
                          "expr_path": [f"adata/{s}.h5ad" for s in ids]}).to_csv(root / task / f"splits/{name}_{i}.csv", index=False)
    json.dump({"genes": genes}, open(root / task / "var_50genes.json", "w"))


def _fake_sample(root, task, sid, genes, n=300):
    rng = np.random.default_rng(abs(hash(sid)) % 2**32)
    os.makedirs(root / task / "patches", exist_ok=True)
    os.makedirs(root / task / "adata", exist_ok=True)
    imgs = rng.integers(0, 255, size=(n, 224, 224, 3), dtype=np.uint8)
    bc = np.array([f"{sid}_{j}".encode() for j in range(n)]).reshape(-1, 1)
    with h5py.File(root / task / f"patches/{sid}.h5", "w") as f:
        f["img"] = imgs
        f["coords"] = rng.integers(0, 9000, size=(n, 2))
        f["barcode"] = bc
    lam = np.exp(imgs.mean(axis=(1, 2, 3))[:, None] / 255 * np.linspace(0.2, 2, len(genes) + 3)[None])
    a = ad.AnnData(rng.poisson(lam).astype(np.float32))
    a.obs_names = [b[0].decode() for b in bc][::-1][: n][::-1]
    a.var_names = genes + ["X1", "X2", "X3"]
    a = a[rng.permutation(n)].copy()  # shuffled order: notebook must match on barcodes
    a.write_h5ad(root / task / f"adata/{sid}.h5ad")


@pytest.mark.skipif(not NB.exists(), reason="notebook not built")
def test_phase1_notebook_end_to_end(tmp_path):
    bench = tmp_path / "bench"
    genes = [f"G{k}" for k in range(50)]
    samples = ["A", "B", "C"]
    _fake_bench(bench, "IDC", samples, genes)
    for s in samples:
        _fake_sample(bench, "IDC", s, genes)

    def bench_file(task, rel):
        p = bench / task / rel
        if not p.exists():
            raise FileNotFoundError(p)
        cp = tmp_path / "scratch" / task / rel   # notebook deletes downloaded files; work on copies
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(p.read_bytes())
        return str(cp)

    def dummy(name, dim):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.AvgPool2d(14), torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, dim))
        return Encoder(name, m.eval(), m, 224, *IMAGENET, micro_batch=64, dim=dim, device="cpu")

    out = tmp_path / "drive" / "embeddings" / "phase1"
    cfg = {"TASKS": ["IDC"]}
    rep = {"/content/drive/MyDrive/hest_fm_project": str(tmp_path / "drive"), "/content/scratch": str(tmp_path / "scratch")}
    g = run_notebook(NB, {
        "TASKS = [": cfg,
        "from encoders import": {"build_encoder": lambda n: dummy(n, 300)},
        "BENCH = ": {"bench_file": bench_file},
    }, init={"os": os, "sys": sys}, replace=rep)
    sanity = json.load(open(out / "sanity_hest_ridge.json"))
    assert set(sanity) == {"IDC/resnet50", "IDC/uni2_h"} and len(sanity["IDC/uni2_h"]["folds"]) == 3
    for s in samples:
        for enc in ("resnet50", "uni2_h", "expression"):
            assert (out / "IDC" / enc / f"{s}.h5").exists()
    summ = json.load(open(out / "run_summary.json"))
    assert summ["tasks"]["IDC"]["samples"]["A"]["n_with_expr"] == 300
    # downloaded raw files were deleted
    assert not (tmp_path / "scratch" / "IDC" / "patches" / "A.h5").exists()
    # resumability: second run skips everything
    g2 = run_notebook(NB, {"TASKS = [": cfg, "from encoders import": {"build_encoder": lambda n: dummy(n, 300)},
                           "BENCH = ": {"bench_file": bench_file}},
                      init={"os": os, "sys": sys}, replace=rep, skip_containing=("PUBLISHED = {",))
    assert g2["summary"]["tasks"]["IDC"]["samples"] == {}
    # local loader reads the notebook output and barcodes line up with expression
    from hestfm.data import load_sample
    s = load_sample(out.parent, out.name + "/IDC", "uni2_h", "A", genes=genes)
    assert s.X.shape == (300, 300) and s.counts.shape == (300, 50)
    raw = ad.read_h5ad(bench / "IDC/adata/A.h5ad")
    assert np.allclose(s.counts, raw[s.barcodes, genes].X)
