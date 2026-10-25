"""Smoke test of notebooks/colab_phase2_xenium5k.ipynb on fake HEST files (CPU, dummy encoders, local 'HF')."""
import json
import os
import shutil
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from hestfm import ROOT
from hestfm.data import load_sample, read_expression
from hestfm.encoders import IMAGENET, Encoder
from notebook_harness import run_notebook

sys.path.insert(0, str(ROOT / "src" / "hestfm"))
import hestfm.data as _hd  # noqa: E402
sys.modules.setdefault("hestfm_data", _hd)  # notebook imports the inlined copy under this name
NB = ROOT / "notebooks" / "colab_phase2_xenium5k.ipynb"
GENES = ["EPCAM", "KRT8", "COL1A1", "DCN", "CD3E", "CD68", "PECAM1", "G1", "G2", "G3"]
S = 409


def make_xenium(root, sid, nx=6, ny=4, cells_per_patch=12, seed=0):
    rng = np.random.default_rng(seed)
    tl = np.stack(np.meshgrid(np.arange(nx) * 365, np.arange(ny) * 365), -1).reshape(-1, 2).astype(np.int64)
    bc = np.array([f"{r:03d}x{c:03d}" for c, r in tl // 365])
    n = len(tl)
    (root / "patches").mkdir(parents=True, exist_ok=True)
    with h5py.File(root / f"patches/{sid}.h5", "w") as f:
        d = f.create_dataset("img", data=rng.integers(0, 255, size=(n, 224, 224, 3), dtype=np.uint8))
        d.attrs["patch_size_src"] = S
        d.attrs["pixel_size"] = 0.2738
        f["coords"] = tl
        f["barcode"] = bc.astype("S").reshape(-1, 1)
    # cells on the tissue extent, each with a type-specific gene program
    ncell = n * cells_per_patch
    cxy = np.c_[rng.uniform(0, nx * 365 + 44, ncell), rng.uniform(0, ny * 365 + 44, ncell)]
    progs = [["EPCAM", "KRT8"], ["COL1A1", "DCN"], ["CD3E"], ["CD68"], ["PECAM1"]]
    rows = []
    for c in range(ncell):
        k = rng.integers(25, 80)
        g = rng.choice(progs[c % len(progs)] + ["G1", "G2", "G3"], k)
        rows.append(pd.DataFrame({"cell_id": f"cell{c}-1", "feature_name": g, "he_x": cxy[c, 0] + rng.normal(0, 2, k),
                                  "he_y": cxy[c, 1] + rng.normal(0, 2, k), "qv": 40.0, "is_gene": True}))
    t = pd.concat(rows + [pd.DataFrame({"cell_id": "UNASSIGNED", "feature_name": "G1", "he_x": 5.0, "he_y": 5.0,
                                        "qv": 40.0, "is_gene": True}, index=range(30)),
                          pd.DataFrame({"cell_id": "cell0-1", "feature_name": "NegControlProbe_A", "he_x": 5.0,
                                        "he_y": 5.0, "qv": 40.0, "is_gene": False}, index=range(30))])
    t = t.sort_values("feature_name", kind="stable").reset_index(drop=True)
    t["transcript_id"] = np.arange(len(t))
    (root / "transcripts").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), root / f"transcripts/{sid}_transcripts.parquet",
                   row_group_size=max(len(t) // 7, 1))
    # HEST-style st: 100 um squares at the same barcodes ~ our window
    from hestfm.pseudospots import TranscriptAggregator, window_assignment
    agg = TranscriptAggregator(); agg.add(t); cm = agg.finalize()
    X = (window_assignment(cm.centroids, tl, S, 100 / 112) @ cm.counts).toarray()
    a = ad.AnnData(X.astype(np.float32))
    a.obs_names, a.var_names = list(bc), list(cm.genes)
    (root / "st").mkdir(exist_ok=True)
    a.write_h5ad(root / f"st/{sid}.h5ad")
    return bc


def make_visium(root, sid, n=40, seed=1):
    rng = np.random.default_rng(seed)
    tl = np.c_[np.arange(n) * 200, np.zeros(n)].astype(np.int64)
    bc = np.array([f"v{i}" for i in range(n)])
    with h5py.File(root / f"patches/{sid}.h5", "w") as f:
        d = f.create_dataset("img", data=rng.integers(0, 255, size=(n, 224, 224, 3), dtype=np.uint8))
        d.attrs["patch_size_src"] = 200
        f["coords"] = tl
        f["barcode"] = bc.astype("S").reshape(-1, 1)
    a = ad.AnnData(rng.poisson(2, size=(n, 5)).astype(np.float32))
    a.obs_names, a.var_names = list(bc), ["EPCAM", "G1", "NOT_IN_5K", "CD3E", "MT-CO1"]
    a.write_h5ad(root / f"st/{sid}.h5ad")


@pytest.mark.skipif(not NB.exists(), reason="notebook not built")
def test_phase2_notebook_end_to_end(tmp_path):
    hest = tmp_path / "hest"
    make_xenium(hest, "X1", seed=0)
    make_xenium(hest, "X2", seed=1)
    make_visium(hest, "V1")
    scratch = tmp_path / "scratch"

    def hest_file(rel):
        src = hest / rel
        dst = scratch / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return str(dst)

    class FakeFS:
        def open(self, path, block_size=None):
            return open(hest / "transcripts" / os.path.basename(path), "rb")

    def dummy(name):
        if name == "hoptimus1":
            raise PermissionError("403 access pending")   # must be skipped gracefully
        torch.manual_seed(len(name))
        m = torch.nn.Sequential(torch.nn.AvgPool2d(28), torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
        return Encoder(name, m.eval(), m, 224, *IMAGENET, micro_batch=16, dim=16, device="cpu")

    rep = {"/content/drive/MyDrive/hest_fm_project": str(tmp_path / "drive"), "/content/scratch": str(scratch),
           "/content/lib": str(tmp_path / "lib")}
    cfg = {"SAMPLES": ["X1", "X2"], "VISIUM": ["V1"]}
    over = {"VISIUM = [": cfg, "from encoders import": {"build_encoder": dummy},
            "def hest_file": {"hest_file": hest_file, "HfFileSystem": FakeFS}}
    g = run_notebook(NB, over, init={"os": os, "sys": sys}, replace=rep)
    out = tmp_path / "drive/embeddings/phase2/xenium5k"
    summ = json.load(open(out / "run_summary.json"))
    assert "hoptimus1" in summ["skipped_encoders"]
    for r in summ["xenium5k"]:
        assert r["errors"] == 0, json.load(open(out / f"qc/{r['sample']}.json")).get("errors")
        assert r["swap_xy"] is False and r["inside_frac"] > 0.9
        assert r["hest_total_r"] > 0.7  # toy grid; HEST spots are 100 um vs our 112 um windows
    for enc in ("uni2_h", "virchow2", "conch_v15", "uni2_h_c56"):
        assert (out / enc / "X1.h5").exists()
    assert not (out / "hoptimus1").exists()
    X, bc, coords, genes = read_expression(out / "expression/X1.h5")
    assert "NegControlProbe_A" not in genes and X.shape == (24, 10)
    big, *_ = read_expression(out / "expression_w200/X1.h5")
    small, *_ = read_expression(out / "expression_w050/X1.h5")
    assert small.sum() < X.sum() < big.sum()
    with h5py.File(out / "expression/X1.h5") as f:
        comp = f["composition"][:]
        assert np.array_equal(comp.sum(1), f["n_cells"][:])
    with h5py.File(out / "cells/X1.h5") as f:
        types = f["types"][:].astype(str)
        assert "epithelial" in types and np.nanmax(f["tau"][:]) > 0.5
    s = load_sample(tmp_path / "drive/embeddings", "phase2/xenium5k", "uni2_h", "X2")
    assert s.X.shape == (24, 16)
    vis = tmp_path / "drive/embeddings/phase4/visium"
    Xv, bv, _, gv = read_expression(vis / "expression/V1.h5")
    assert set(gv) == {"EPCAM", "G1", "CD3E"} and Xv.shape == (40, 3)
    assert not (scratch / "patches/X1.h5").exists() and not (scratch / "st/X1.h5ad").exists()
    assert not (scratch / "transcripts/X1_transcripts.parquet").exists()
    # resumable: a second run does nothing and raises nothing
    before = {p: p.stat().st_mtime for p in out.rglob("*.h5")}
    run_notebook(NB, over, init={"os": os, "sys": sys}, replace=rep)
    assert all(p.stat().st_mtime == t for p, t in before.items())  # nothing recomputed
