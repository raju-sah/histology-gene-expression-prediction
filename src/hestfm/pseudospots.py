"""Patch-aligned pseudo-spots from Xenium transcripts (inlined into the Phase 2 Colab notebook).

Definition (documented in PROGRESS.md / paper):
- A transcript counts if `is_gene`, `qv >= 20`, and it is assigned to a cell (`cell_id != "UNASSIGNED"`).
- A cell's position is the mean H&E position (`he_x`, `he_y`, full-resolution px) of its counted transcripts.
- A pseudo-spot is the square window of side `scale * S` centred on a HEST patch, where S is the patch footprint in
  full-res px (`patch_size_src`, 112 um at 0.5 um/px after resizing to 224 px). Its counts are the summed counts of
  all cells whose position falls inside the window. scale=1 is the primary, patch-aligned definition; 0.5 and 2 are
  used for the window-size sensitivity analysis. Windows of neighbouring patches overlap (HEST grid spacing 100 um).

Transcript files are streamed one row group at a time; nothing is held per transcript.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

UNASSIGNED = {"UNASSIGNED", "-1", "", "None", "nan"}
COLUMNS = ["cell_id", "feature_name", "he_x", "he_y", "qv", "is_gene"]


@dataclass
class CellMatrix:
    counts: sp.csr_matrix   # (n_cells, n_genes) int
    genes: np.ndarray
    centroids: np.ndarray   # (n_cells, 2) mean he_x, he_y
    n_transcripts: np.ndarray
    cell_hash: np.ndarray   # uint64 hash of cell_id


def _hash(strings) -> np.ndarray:
    return pd.util.hash_array(np.asarray(strings, dtype=object))


class TranscriptAggregator:
    """Streaming cell x gene counts + centroids."""

    def __init__(self, min_qv: float = 20.0):
        self.min_qv = min_qv
        self.cell_index = pd.Index(np.array([], dtype=np.uint64))
        self.gene_pos: dict[str, int] = {}
        self.sum_x = np.zeros(0)
        self.sum_y = np.zeros(0)
        self.n = np.zeros(0, dtype=np.int64)
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.n_seen = self.n_kept = 0

    def _grow(self, size):
        if size > len(self.n):
            new = max(size, int(len(self.n) * 1.5) + 1024)
            for k in ("sum_x", "sum_y"):
                a = np.zeros(new); a[: len(getattr(self, k))] = getattr(self, k); setattr(self, k, a)
            a = np.zeros(new, dtype=np.int64); a[: len(self.n)] = self.n; self.n = a

    def add(self, table) -> None:
        """`table`: pyarrow Table or pandas DataFrame with COLUMNS."""
        import pyarrow as pa
        import pyarrow.compute as pc
        if isinstance(table, pd.DataFrame):
            table = pa.Table.from_pandas(table, preserve_index=False)
        self.n_seen += table.num_rows
        cid = pc.cast(table["cell_id"], pa.string())
        mask = pc.and_(pc.and_(pc.cast(table["is_gene"], pa.bool_()), pc.greater_equal(table["qv"], self.min_qv)),
                       pc.invert(pc.is_in(cid, value_set=pa.array(sorted(UNASSIGNED)))))
        mask = pc.fill_null(mask, False)
        t = table.filter(mask)
        if t.num_rows == 0:
            return
        self.n_kept += t.num_rows
        # cells: dictionary-encode within the chunk, hash only the unique ids
        cd = pc.dictionary_encode(pc.cast(t["cell_id"], pa.string())).combine_chunks()
        h = _hash(cd.dictionary.to_numpy(zero_copy_only=False))
        pos = self.cell_index.get_indexer(h)
        new = pos < 0
        if new.any():
            start = len(self.cell_index)
            self.cell_index = self.cell_index.append(pd.Index(h[new]))
            pos[new] = np.arange(start, start + new.sum())
        cell = pos[cd.indices.to_numpy(zero_copy_only=False)].astype(np.int64)
        # genes
        gd = pc.dictionary_encode(pc.cast(t["feature_name"], pa.string())).combine_chunks()
        gnames = gd.dictionary.to_pylist()
        gmap = np.array([self.gene_pos.setdefault(g, len(self.gene_pos)) for g in gnames], dtype=np.int64)
        gene = gmap[gd.indices.to_numpy(zero_copy_only=False)]
        x = t["he_x"].to_numpy().astype(np.float64)
        y = t["he_y"].to_numpy().astype(np.float64)
        self._grow(len(self.cell_index))
        m = len(self.cell_index)
        self.sum_x[:m] += np.bincount(cell, weights=x, minlength=m)
        self.sum_y[:m] += np.bincount(cell, weights=y, minlength=m)
        self.n[:m] += np.bincount(cell, minlength=m)
        key = cell * (1 << 20) + gene
        u, c = np.unique(key, return_counts=True)
        self.parts.append(((u >> 20).astype(np.uint32), (u & ((1 << 20) - 1)).astype(np.uint32), c.astype(np.uint32)))

    def finalize(self) -> CellMatrix:
        n_cells, n_genes = len(self.cell_index), len(self.gene_pos)
        if self.parts:
            r = np.concatenate([p[0] for p in self.parts])
            c = np.concatenate([p[1] for p in self.parts])
            v = np.concatenate([p[2] for p in self.parts])
        else:
            r = c = v = np.zeros(0, dtype=np.uint32)
        self.parts = []
        M = sp.coo_matrix((v, (r, c)), shape=(n_cells, n_genes)).tocsr()
        M.sum_duplicates()
        names = np.array(sorted(self.gene_pos, key=self.gene_pos.get))
        order = np.argsort(names)
        n = self.n[:n_cells]
        cent = np.c_[self.sum_x[:n_cells] / np.maximum(n, 1), self.sum_y[:n_cells] / np.maximum(n, 1)]
        return CellMatrix(M[:, order].tocsr(), names[order], cent, n, self.cell_index.to_numpy())


def window_assignment(centroids: np.ndarray, patch_topleft: np.ndarray, patch_size_px: float,
                      scale: float = 1.0, swap_xy: bool = False) -> sp.csr_matrix:
    """(n_patches, n_cells) 0/1 matrix: cell centroid inside the square window of side scale*S at the patch centre."""
    from scipy.spatial import cKDTree
    c = centroids[:, ::-1] if swap_xy else centroids
    centers = np.asarray(patch_topleft, dtype=float) + patch_size_px / 2.0
    half = scale * patch_size_px / 2.0
    tree = cKDTree(c)
    lists = tree.query_ball_point(centers, r=half, p=np.inf)
    rows = np.repeat(np.arange(len(centers)), [len(l) for l in lists])
    cols = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if len(rows) else np.zeros(0, dtype=np.int64)
    # query_ball_point is inclusive (<= half); make the window half-open like a pixel footprint
    if len(cols):
        d = np.abs(c[cols] - centers[rows])
        ok = (d < half).all(1)
        rows, cols = rows[ok], cols[ok]
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(centers), len(c)))


def orientation_check(centroids, patch_topleft, patch_size_px, max_cells=200_000, seed=0) -> dict:
    """Fraction of cells inside any patch footprint for (x, y) and swapped (y, x). The larger one is the right frame."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(centroids), size=min(max_cells, len(centroids)), replace=False)
    out = {}
    for swap in (False, True):
        A = window_assignment(centroids[idx], patch_topleft, patch_size_px, 1.0, swap)
        out["swapped" if swap else "xy"] = float((np.asarray(A.sum(0)).ravel() > 0).mean())
    return out


def pseudospot_counts(A: sp.csr_matrix, cells: CellMatrix) -> sp.csr_matrix:
    return (A @ cells.counts).tocsr().astype(np.float32)
