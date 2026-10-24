"""On-disk format shared by the Colab notebooks (writers) and local analysis (readers).

    <root>/<dataset>/<encoder>/<sample>.h5      embeddings (N, D) float16, barcodes (N,), coords (N, 2)
    <root>/<dataset>/expression/<sample>.h5     barcodes (N,), genes (G,), coords (N, 2), and either
                                                counts (N, G) dense, or a CSR group `counts_csr`
                                                {data, indices, indptr, shape}; optional extra datasets.

`dataset` is e.g. "phase1/IDC" or "phase2/xenium5k". Rows are matched on barcodes, never on position.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def _str(a) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


@dataclass
class SampleData:
    sample_id: str
    X: np.ndarray            # (N, D) float32 embeddings
    counts: np.ndarray       # (N, G) float32 raw counts
    barcodes: np.ndarray
    coords: np.ndarray
    genes: np.ndarray


def read_embeddings(path: str | Path):
    with h5py.File(path, "r") as f:
        emb = f["embeddings"][:]
        bc = _str(f["barcodes"][:])
        coords = f["coords"][:] if "coords" in f else np.zeros((len(bc), 2))
    return emb, bc, coords


def read_expression(path: str | Path, genes: list[str] | None = None):
    with h5py.File(path, "r") as f:
        bc = _str(f["barcodes"][:])
        all_genes = _str(f["genes"][:])
        coords = f["coords"][:] if "coords" in f else np.zeros((len(bc), 2))
        if "counts_csr" in f:
            g = f["counts_csr"]
            M = sp.csr_matrix((g["data"][:].astype(np.float32), g["indices"][:].astype(np.int32),
                               g["indptr"][:].astype(np.int64)), shape=tuple(g["shape"][:]))
        else:
            M = f["counts"][:]
    if genes is not None:
        pos = {g: i for i, g in enumerate(all_genes)}
        missing = [g for g in genes if g not in pos]
        if missing:
            raise KeyError(f"{path}: genes missing: {missing[:5]}... ({len(missing)})")
        idx = np.array([pos[g] for g in genes])
        M = M[:, idx]
        all_genes = np.asarray(genes)
    return M, bc, coords, all_genes


def load_sample(root: str | Path, dataset: str, encoder: str, sample_id: str,
                genes: list[str] | None = None) -> SampleData:
    root = Path(root) / dataset
    emb, ebc, coords = read_embeddings(root / encoder / f"{sample_id}.h5")
    M, xbc, _, g = read_expression(root / "expression" / f"{sample_id}.h5", genes)
    pos = {b: i for i, b in enumerate(xbc)}
    keep = np.array([b in pos for b in ebc])
    if keep.sum() == 0:
        raise ValueError(f"{sample_id}: no barcode overlap between embeddings and expression")
    rows = np.array([pos[b] for b in ebc[keep]])
    Y = M[rows]
    Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    return SampleData(sample_id, emb[keep].astype(np.float32), Y.astype(np.float32), ebc[keep], coords[keep], g)


def stack(samples: list[SampleData]):
    """Concatenate samples; returns X, counts, sample_ids, barcodes, coords."""
    X = np.concatenate([s.X for s in samples])
    Y = np.concatenate([s.counts for s in samples])
    sid = np.concatenate([[s.sample_id] * len(s.X) for s in samples])
    bc = np.concatenate([s.barcodes for s in samples])
    coords = np.concatenate([s.coords for s in samples])
    return X, Y, sid, bc, coords


def write_embeddings(path, embeddings, barcodes, coords):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=np.asarray(embeddings, dtype=np.float16))
        f.create_dataset("barcodes", data=np.asarray(barcodes).astype("S"))
        f.create_dataset("coords", data=np.asarray(coords))


def write_expression(path, counts, barcodes, genes, coords, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if sp.issparse(counts):
            c = sp.csr_matrix(counts)
            c.sum_duplicates()
            data = c.data
            if np.allclose(data, np.round(data)) and (len(data) == 0 or data.max() < 65535):
                data = data.astype(np.uint16)   # integer counts: compact
            idx_dtype = np.int16 if c.shape[1] < 32767 else np.int32
            g = f.create_group("counts_csr")
            kw = dict(compression="gzip", compression_opts=4)
            g.create_dataset("data", data=data, **kw)
            g.create_dataset("indices", data=c.indices.astype(idx_dtype), **kw)
            g.create_dataset("indptr", data=c.indptr.astype(np.int64), **kw)
            g["shape"] = np.array(c.shape)
        else:
            f.create_dataset("counts", data=np.asarray(counts, dtype=np.float32))
        f.create_dataset("barcodes", data=np.asarray(barcodes).astype("S"))
        f.create_dataset("genes", data=np.asarray(genes).astype("S"))
        f.create_dataset("coords", data=np.asarray(coords))
        for k, v in extra.items():
            f.create_dataset(k, data=v)
