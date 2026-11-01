"""Unsupervised clustering-based cell composition baseline for Step 2.

Addresses the weakness of coarse marker typing (where 60-70% of cells are unassigned).
Performs unsupervised clustering of single cells:
  1. Log-normalise expression: log1p(100 * counts / total).
  2. Fit PCA (50 components) on training cells only.
  3. Build kNN graph (k=15) on training cells and run Leiden community detection.
  4. Label clusters by their top marker genes.
  5. Compute training cluster centroids in PCA space.
  6. Assign test cells to nearest training centroid (strictly zero leakage).
  7. Compute per-spot cluster composition: count of each cluster's cells within the spot window.
     This achieves 0% unassigned cells.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import igraph
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from hestfm import ROOT
from hestfm.pseudospots import window_assignment


@dataclass
class ClusterPipeline:
    svd: TruncatedSVD
    centroids: np.ndarray          # (K, n_components)
    cluster_labels: np.ndarray     # training cluster labels
    cluster_markers: list[list[str]]  # top marker genes per cluster
    n_clusters: int
    common_genes: list[str]


def read_sample_cells(root: Path | str, sid: str, genes: list[str] | None = None) -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    """Loads (cell_counts_csr, cell_centroids, gene_names) for a sample."""
    p = Path(root) / "phase2/xenium5k/cells" / f"{sid}.h5"
    with h5py.File(p, "r") as f:
        all_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["genes"][:]]
        centroids = f["cell_centroids"][:]
        g = f["cell_counts_csr"]
        counts = sp.csr_matrix((g["data"][:].astype(np.float32), g["indices"][:].astype(np.int32),
                                g["indptr"][:].astype(np.int64)), shape=tuple(g["shape"][:]))
    if genes is not None:
        pos = {x: i for i, x in enumerate(all_genes)}
        keep = [pos[x] for x in genes if x in pos]
        assert len(keep) == len(genes), f"Some genes missing in {sid}"
        counts = counts[:, keep].tocsr()
        all_genes = list(genes)
    return counts, centroids, all_genes


def lognorm_cells(counts: sp.csr_matrix, target: float = 100.0) -> sp.csr_matrix:
    """Standard single-cell library size normalisation."""
    tot = np.asarray(counts.sum(1)).ravel()
    scale = target / np.maximum(tot, 1.0)
    X = sp.diags(scale) @ counts.astype(np.float32)
    X = X.tocsr()
    X.data = np.log1p(X.data)
    return X


def fit_unsupervised_clustering(
    train_counts_list: Sequence[sp.csr_matrix],
    genes: list[str],
    n_components: int = 50,
    resolution: float = 0.8,
    k_neighbors: int = 15,
    max_train_cells: int = 30000,
    seed: int = 42,
) -> ClusterPipeline:
    """Fits unsupervised PCA and Leiden clustering strictly on training cells."""
    rng = np.random.default_rng(seed)
    
    # Subsample training cells if large to ensure fast, deterministic clustering
    subsampled = []
    for counts in train_counts_list:
        n_c = counts.shape[0]
        per_sample_budget = max(1000, max_train_cells // len(train_counts_list))
        if n_c > per_sample_budget:
            idx = rng.choice(n_c, size=per_sample_budget, replace=False)
            idx.sort()
            subsampled.append(counts[idx])
        else:
            subsampled.append(counts)
    
    X_train_raw = sp.vstack(subsampled).tocsr()
    X_norm = lognorm_cells(X_train_raw)

    # 1. PCA (50 components) on training cells
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z_train = svd.fit_transform(X_norm)

    # 2. kNN graph on PCA embeddings
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto").fit(Z_train)
    adj = nn.kneighbors_graph(Z_train, mode="connectivity")
    
    # 3. Leiden community detection
    sources, targets = adj.nonzero()
    g = igraph.Graph(directed=False)
    g.add_vertices(len(Z_train))
    g.add_edges(list(zip(sources, targets)))
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    train_labels = np.array(part.membership)
    n_clusters = int(train_labels.max() + 1)

    # 4. Compute cluster centroids in PCA space
    centroids = np.zeros((n_clusters, n_components), dtype=np.float32)
    for k in range(n_clusters):
        mask = train_labels == k
        centroids[k] = Z_train[mask].mean(0)

    # 5. Find top marker genes per cluster
    markers = []
    # Convert training subset mean expression
    mean_all = np.asarray(X_norm.mean(0)).ravel()
    for k in range(n_clusters):
        mask = train_labels == k
        if mask.any():
            mean_k = np.asarray(X_norm[mask].mean(0)).ravel()
            diff = mean_k - mean_all
            top_idx = np.argsort(diff)[::-1][:5]
            markers.append([genes[i] for i in top_idx])
        else:
            markers.append([])

    return ClusterPipeline(
        svd=svd,
        centroids=centroids,
        cluster_labels=train_labels,
        cluster_markers=markers,
        n_clusters=n_clusters,
        common_genes=genes,
    )


def assign_test_cells(
    test_counts: sp.csr_matrix,
    pipeline: ClusterPipeline,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Assigns test cells to nearest training cluster centroid without leakage (chunked to bound memory)."""
    n_cells = test_counts.shape[0]
    labels = np.zeros(n_cells, dtype=np.int32)
    mu_sq = np.sum(pipeline.centroids ** 2, axis=1)
    for i in range(0, n_cells, chunk_size):
        chunk = test_counts[i : i + chunk_size]
        X_norm = lognorm_cells(chunk)
        Z_test = pipeline.svd.transform(X_norm)
        dists = -2.0 * (Z_test @ pipeline.centroids.T) + mu_sq
        labels[i : i + chunk.shape[0]] = dists.argmin(axis=1)
    return labels


def spot_cluster_composition(
    centroids: np.ndarray,
    cell_labels: np.ndarray,
    n_clusters: int,
    patch_topleft: np.ndarray,
    patch_size_px: float,
    swap_xy: bool = False,
) -> np.ndarray:
    """Builds (n_spots, n_clusters) matrix of cell cluster counts per pseudo-spot.
    
    Notice: 100% of cells assigned to spot windows have a valid cluster label; unassigned = 0.
    """
    A = window_assignment(centroids, patch_topleft, patch_size_px, scale=1.0, swap_xy=swap_xy)
    onehot = sp.csr_matrix(
        (np.ones(len(cell_labels), dtype=np.float32), (np.arange(len(cell_labels)), cell_labels)),
        shape=(len(cell_labels), n_clusters)
    )
    comp = np.asarray((A @ onehot).todense())
    return comp
