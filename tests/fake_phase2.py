"""Write a fake data/embeddings/phase2/xenium5k tree in the notebook's output format."""
import h5py
import numpy as np
import scipy.sparse as sp

from hestfm import load_config
from hestfm.data import write_embeddings, write_expression

TYPES = ["epithelial", "fibroblast", "t_nk"]


def make_fake_phase2(root, encoders=("uni2_h", "virchow2"), n_genes=40, grid=(12, 10), seed=0, dataset="phase2/xenium5k",
                     samples=None):
    rng = np.random.default_rng(seed)
    samples = samples or list(load_config()["xenium5k_samples"])
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    W = {e: rng.normal(size=(6, 32 + 8 * k)) for k, e in enumerate(encoders)}
    B = rng.normal(size=(6, n_genes)) * np.r_[np.ones(n_genes // 2), np.zeros(n_genes - n_genes // 2)] * 0.7
    for s in samples:
        xx, yy = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]))
        coords = np.c_[xx.ravel(), yy.ravel()] * 365
        n = len(coords)
        z = np.c_[np.sin(coords[:, :1] / 900 + rng.uniform(0, 6)), np.cos(coords[:, 1:] / 700), rng.normal(size=(n, 4))]
        bc = np.array([f"{r:03d}x{c:03d}" for c, r in coords // 365])
        for e in encoders:
            write_embeddings(f"{root}/{dataset}/{e}/{s}.h5", z @ W[e] + rng.normal(0, 0.7, (n, W[e].shape[1])), bc, coords)
        lam = np.exp(np.clip(z @ B, -3, 3) + 0.3)
        Y = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
        comp = rng.poisson(3, size=(n, len(TYPES) + 2))
        for sub, f in (("expression_w050", 0.5), ("expression_w200", 2.0)):
            write_expression(f"{root}/{dataset}/{sub}/{s}.h5", sp.csr_matrix(rng.poisson(lam * f).astype(np.float32)),
                             bc, genes, coords)
        write_expression(f"{root}/{dataset}/expression/{s}.h5", Y, bc, genes, coords,
                         n_cells=comp.sum(1).astype(np.int32), composition=comp.astype(np.int32),
                         composition_cols=np.array(TYPES + ["unassigned", "low_quality"]).astype("S"))
        with h5py.File(_mk(f"{root}/{dataset}/cells/{s}.h5"), "w") as f:
            f["type_profiles"] = rng.gamma(1.0, 1.0, size=(len(TYPES), n_genes)).astype(np.float32)
            f["types"] = np.array(TYPES).astype("S")
            f["n_per_type"] = rng.integers(50, 500, len(TYPES))
            f["genes"] = genes.astype("S")
    return samples, list(genes)


def _mk(p):
    import os
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
