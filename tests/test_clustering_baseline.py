"""Tests for unsupervised clustering baseline (Step 2)."""
import numpy as np
import scipy.sparse as sp

from hestfm.cluster_baseline import (
    assign_test_cells,
    fit_unsupervised_clustering,
    spot_cluster_composition,
)


def test_clustering_baseline_no_leakage():
    rng = np.random.default_rng(42)
    n_genes = 20
    genes = [f"G{i}" for i in range(n_genes)]
    
    # 2 training samples
    counts_tr1 = sp.csr_matrix(rng.poisson(2, size=(200, n_genes)).astype(np.float32))
    counts_tr2 = sp.csr_matrix(rng.poisson(2, size=(300, n_genes)).astype(np.float32))
    
    # 1 test sample
    counts_te = sp.csr_matrix(rng.poisson(5, size=(150, n_genes)).astype(np.float32))
    
    # Fit strictly on training samples
    pipe1 = fit_unsupervised_clustering([counts_tr1, counts_tr2], genes, n_components=10, seed=42)
    assert pipe1.n_clusters >= 2
    assert pipe1.centroids.shape == (pipe1.n_clusters, 10)
    
    # Assign test cells to nearest training centroid
    labels_te1 = assign_test_cells(counts_te, pipe1)
    assert len(labels_te1) == 150
    assert labels_te1.min() >= 0 and labels_te1.max() < pipe1.n_clusters
    
    # Adversarial test: modify test counts drastically, verify pipe1 is 100% invariant
    counts_te_corrupted = sp.csr_matrix(rng.poisson(500, size=(150, n_genes)).astype(np.float32))
    pipe2 = fit_unsupervised_clustering([counts_tr1, counts_tr2], genes, n_components=10, seed=42)
    
    np.testing.assert_array_equal(pipe1.centroids, pipe2.centroids)
    np.testing.assert_array_equal(pipe1.cluster_labels, pipe2.cluster_labels)
    assert pipe1.cluster_markers == pipe2.cluster_markers


def test_spot_cluster_composition_unassigned_zero():
    rng = np.random.default_rng(123)
    n_cells = 500
    n_clusters = 5
    cell_labels = rng.integers(0, n_clusters, size=n_cells)
    
    # Grid of spots
    patch_topleft = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    patch_size_px = 80.0
    
    # Cell centroids mostly inside spots
    centroids = rng.uniform(0, 150, size=(n_cells, 2))
    
    comp = spot_cluster_composition(centroids, cell_labels, n_clusters, patch_topleft, patch_size_px)
    assert comp.shape == (len(patch_topleft), n_clusters)
    
    # Every cell captured in a spot belongs to a cluster (no unassigned column)
    assert comp.min() >= 0
    # Spot total matches sum of assigned clusters
    from hestfm.pseudospots import window_assignment
    A = window_assignment(centroids, patch_topleft, patch_size_px)
    expected_cells_per_spot = np.asarray(A.sum(1)).ravel()
    np.testing.assert_array_equal(comp.sum(1), expected_cells_per_spot)
