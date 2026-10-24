"""Coarse marker-based cell typing and gene cell-type specificity (inlined into the Phase 2 notebook).

This is deliberately simple: each cell gets the type whose marker set has the highest mean log-normalised
expression, if that score is above a floor; otherwise "unassigned". Cells with few transcripts are "low_quality".
It is used to (1) describe pseudo-spot composition and (2) compute per-gene specificity (tau) across types.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MARKERS = {
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "KDR"],
    "smooth_muscle_pericyte": ["ACTA2", "MYH11", "TAGLN", "DES", "RGS5"],
    "t_nk": ["CD3E", "CD3D", "CD2", "CD8A", "CD247", "NKG7"],
    "b": ["MS4A1", "CD79A", "CD19", "CD22", "BANK1"],
    "plasma": ["MZB1", "JCHAIN", "TNFRSF17", "DERL3"],
    "myeloid": ["CD68", "CD163", "CD14", "LYZ", "CSF1R", "C1QA", "C1QB"],
    "mast": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
}
MIN_TRANSCRIPTS = 20
SCORE_FLOOR = 0.3


def lognorm(counts: sp.csr_matrix, target: float = 100.0) -> sp.csr_matrix:
    tot = np.asarray(counts.sum(1)).ravel()
    scale = target / np.maximum(tot, 1)
    X = sp.diags(scale) @ counts.astype(np.float32)
    X = X.tocsr()
    X.data = np.log1p(X.data)
    return X


def annotate(counts: sp.csr_matrix, genes, markers=MARKERS, min_transcripts=MIN_TRANSCRIPTS, floor=SCORE_FLOOR):
    """Returns (labels array of str, type names, used markers dict)."""
    genes = list(genes)
    pos = {g: i for i, g in enumerate(genes)}
    X = lognorm(counts)
    types, scores, used = [], [], {}
    for t, ms in markers.items():
        idx = [pos[m] for m in ms if m in pos]
        if not idx:
            continue
        used[t] = [genes[i] for i in idx]
        types.append(t)
        scores.append(np.asarray(X[:, idx].mean(1)).ravel())
    S = np.vstack(scores).T
    best = S.argmax(1)
    labels = np.array(types, dtype=object)[best]
    labels[S.max(1) < floor] = "unassigned"
    labels[np.asarray(counts.sum(1)).ravel() < min_transcripts] = "low_quality"
    return labels.astype(str), types, used


def type_profiles(counts: sp.csr_matrix, labels, types):
    """(n_types, n_genes) mean log-normalised expression per type, and cells per type."""
    X = lognorm(counts)
    prof, n = [], []
    for t in types:
        m = labels == t
        n.append(int(m.sum()))
        prof.append(np.asarray(X[m].mean(0)).ravel() if m.any() else np.zeros(X.shape[1]))
    return np.vstack(prof), np.array(n)


def tau(profiles: np.ndarray) -> np.ndarray:
    """Tissue-specificity index (Yanai et al. 2005) across types: 0 = uniform, 1 = one type only. NaN if all zero."""
    P = np.asarray(profiles, dtype=float).T          # genes x types
    mx = P.max(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (1 - P / mx).sum(1) / (P.shape[1] - 1)
    t[mx.ravel() <= 0] = np.nan
    return t


def composition(A: sp.csr_matrix, labels, types) -> np.ndarray:
    """(n_spots, n_types + 2) cell counts per type in each pseudo-spot; last two columns unassigned, low_quality."""
    cols = list(types) + ["unassigned", "low_quality"]
    onehot = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)),
                            [cols.index(l) for l in labels])), shape=(len(labels), len(cols)))
    return np.asarray((A @ onehot).todense()), cols
