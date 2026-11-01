"""Step 2: Stronger composition baseline via unsupervised single-cell clustering.

Memory-lean: loads single-cell matrices on-demand and chunks test assignment,
staying well below the 3.8 GB RSS limit.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.atlas import common_genes, DEFAULT_ALPHAS  # noqa: E402
from hestfm.cluster_baseline import (  # noqa: E402
    assign_test_cells,
    fit_unsupervised_clustering,
    read_sample_cells,
    spot_cluster_composition,
)
from hestfm.heads import RidgeCVHead  # noqa: E402
from hestfm.metrics import pearson_per_gene  # noqa: E402
from hestfm.splits import Fold, assert_no_leakage, inner_validation_mask  # noqa: E402


def load_subsample_cells(root: Path, sid: str, genes: list[str], max_cells: int = 15000, seed: int = 42):
    """Loads a random subsample of cells for a training sample to keep memory lean."""
    counts, centroids, _ = read_sample_cells(root, sid, genes)
    n = counts.shape[0]
    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
        counts = counts[idx]
        centroids = centroids[idx]
    gc.collect()
    return counts, centroids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase2"))
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=0.8)
    ap.add_argument("--max-train-per-sample", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    root = Path(a.root)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    all_groups = cfg["xenium5k_samples"]

    # Samples with full single-cell expression matrices
    cells_dir = root / "phase2/xenium5k/cells"
    cell_samples = []
    for sid in all_groups:
        cp = cells_dir / f"{sid}.h5"
        if cp.exists():
            with h5py.File(cp, "r") as f:
                if "cell_counts_csr" in f and "cell_centroids" in f:
                    cell_samples.append(sid)

    print(f"Samples with full per-cell data: {cell_samples}")
    if len(cell_samples) < 2:
        print("Need at least 2 samples with per-cell expression for cross-validation.")
        return

    groups = {s: all_groups[s] for s in cell_samples}
    cgenes = common_genes(root, "phase2/xenium5k", cell_samples)
    print(f"Common genes across cell samples: {len(cgenes)}")

    # Load spot data (coords, qc, original composition, expression, n_cells)
    spot_data = {}
    for sid in cell_samples:
        expr_p = root / "phase2/xenium5k/expression" / f"{sid}.h5"
        qc_p = root / "phase2/xenium5k/qc" / f"{sid}.json"
        with h5py.File(expr_p, "r") as f:
            coords = f["coords"][:]
            bc = f["barcodes"][:].astype(str)
            all_g = [x.decode() if isinstance(x, bytes) else str(x) for x in f["genes"][:]]
            gpos = {g: i for i, g in enumerate(all_g)}
            gi = np.array([gpos[g] for g in cgenes])
            g = f["counts_csr"]
            M = sp.csr_matrix((g["data"][:].astype(np.float32), g["indices"][:].astype(np.int32),
                               g["indptr"][:].astype(np.int64)), shape=tuple(g["shape"][:]))
            Y_raw = M[:, gi].toarray()
            orig_comp = f["composition"][:] if "composition" in f else None
            n_cells = f["n_cells"][:] if "n_cells" in f else None
            comp_cols = [x.decode() if isinstance(x, bytes) else str(x) for x in f["composition_cols"][:]] if "composition_cols" in f else []
        qc = json.load(open(qc_p))
        patch_size = float(qc.get("patch_size_src", 409.0))
        swap_xy = bool(qc.get("swap_xy", False))
        
        # Calculate coarse unassigned fraction
        if orig_comp is not None and "unassigned" in comp_cols:
            u_idx = comp_cols.index("unassigned")
            unassigned_frac = float(orig_comp[:, u_idx].sum() / max(orig_comp.sum(), 1))
        else:
            unassigned_frac = np.nan

        spot_data[sid] = {
            "coords": coords, "barcodes": bc, "Y_raw": Y_raw,
            "patch_size": patch_size, "swap_xy": swap_xy,
            "orig_comp": orig_comp, "n_cells": n_cells,
            "unassigned_frac": unassigned_frac,
        }

    # Build leave-one-group-out folds over cell samples
    unique_groups = sorted(set(groups.values()))
    folds = []
    for g in unique_groups:
        te = tuple(s for s in cell_samples if groups[s] == g)
        tr = tuple(s for s in cell_samples if groups[s] != g)
        folds.append(Fold(f"logo_{g}", tr, te, {"test_group": g}))
    assert_no_leakage(folds, groups)

    records = []
    # Pre-compute cluster compositions for each fold to avoid recomputing across targets
    fold_compositions = {}
    for fold in folds:
        tr, te = fold.train, fold.test
        print(f"\n--- Fitting unsupervised clustering for {fold.name} (train: {tr}, test: {te}) ---")
        train_counts = []
        for s in tr:
            cnt, _ = load_subsample_cells(root, s, cgenes, max_cells=a.max_train_per_sample, seed=a.seed)
            train_counts.append(cnt)
        pipeline = fit_unsupervised_clustering(
            train_counts, cgenes,
            n_components=a.n_components,
            resolution=a.resolution,
            max_train_cells=len(tr) * a.max_train_per_sample,
            seed=a.seed,
        )
        del train_counts; gc.collect()
        print(f"[{fold.name}] Discovered {pipeline.n_clusters} clusters. Top marker preview: cluster 0 -> {pipeline.cluster_markers[0][:3]}")

        # Compute spot cluster composition for training samples
        X_tr_cluster_dict = {}
        for s in tr:
            counts, centroids, _ = read_sample_cells(root, s, cgenes)
            c_labels = assign_test_cells(counts, pipeline, chunk_size=50000)
            comp = spot_cluster_composition(
                centroids, c_labels, pipeline.n_clusters,
                spot_data[s]["coords"], spot_data[s]["patch_size"], spot_data[s]["swap_xy"]
            )
            X_tr_cluster_dict[s] = comp
            del counts, centroids, c_labels; gc.collect()

        # Compute spot cluster composition for test sample (strictly nearest centroid)
        X_te_cluster_dict = {}
        for s in te:
            counts, centroids, _ = read_sample_cells(root, s, cgenes)
            c_labels = assign_test_cells(counts, pipeline, chunk_size=50000)
            comp = spot_cluster_composition(
                centroids, c_labels, pipeline.n_clusters,
                spot_data[s]["coords"], spot_data[s]["patch_size"], spot_data[s]["swap_xy"]
            )
            X_te_cluster_dict[s] = comp
            del counts, centroids, c_labels; gc.collect()

        fold_compositions[fold.name] = {
            "pipeline": pipeline,
            "X_tr": X_tr_cluster_dict,
            "X_te": X_te_cluster_dict,
        }

    # Evaluate across targets: log1p and lognorm
    for y_kind in ("log1p", "lognorm"):
        print(f"\n=== Evaluating Target: {y_kind} ===")
        # Prepare spot target expression Y
        Ys = {}
        for sid in cell_samples:
            Y_raw = spot_data[sid]["Y_raw"]
            if y_kind == "log1p":
                Ys[sid] = np.log1p(Y_raw)
            elif y_kind == "lognorm":
                tot = np.maximum(Y_raw.sum(1, keepdims=True), 1.0)
                scale = 10000.0 / tot
                Ys[sid] = np.log1p(scale * Y_raw)

        for fold in folds:
            tr, te = fold.train, fold.test
            comps = fold_compositions[fold.name]
            n_clusters = comps["pipeline"].n_clusters

            X_tr_cluster = np.vstack([np.log1p(comps["X_tr"][s].astype(np.float32)) for s in tr])
            X_tr_coarse = np.vstack([np.log1p(spot_data[s]["orig_comp"].astype(np.float32)) for s in tr])
            X_tr_ncells = np.vstack([np.log1p(spot_data[s]["n_cells"].astype(np.float32))[:, None] for s in tr])
            Y_tr = np.vstack([Ys[s] for s in tr])
            sid_tr = np.concatenate([[s] * len(Ys[s]) for s in tr])

            X_te_cluster = np.vstack([np.log1p(comps["X_te"][s].astype(np.float32)) for s in te])
            X_te_coarse = np.vstack([np.log1p(spot_data[s]["orig_comp"].astype(np.float32)) for s in te])
            X_te_ncells = np.vstack([np.log1p(spot_data[s]["n_cells"].astype(np.float32))[:, None] for s in te])
            Y_te = np.vstack([Ys[s] for s in te])

            c_tr = np.concatenate([spot_data[s]["coords"] for s in tr])
            val_mask = inner_validation_mask(sid_tr, c_tr, seed=a.seed, groups=groups)

            # Fit RidgeCV on cluster composition
            head_cluster = RidgeCVHead(alphas=DEFAULT_ALPHAS)
            head_cluster.fit(X_tr_cluster, Y_tr, val_mask)
            pred_cluster = head_cluster.predict(X_te_cluster)
            r_cluster = pearson_per_gene(Y_te, pred_cluster)

            # Fit RidgeCV on coarse composition
            head_coarse = RidgeCVHead(alphas=DEFAULT_ALPHAS)
            head_coarse.fit(X_tr_coarse, Y_tr, val_mask)
            pred_coarse = head_coarse.predict(X_te_coarse)
            r_coarse = pearson_per_gene(Y_te, pred_coarse)

            # Fit RidgeCV on n_cells
            head_ncells = RidgeCVHead(alphas=DEFAULT_ALPHAS)
            head_ncells.fit(X_tr_ncells, Y_tr, val_mask)
            pred_ncells = head_ncells.predict(X_te_ncells)
            r_ncells = pearson_per_gene(Y_te, pred_ncells)

            m_cluster = float(np.nanmean(r_cluster))
            m_coarse = float(np.nanmean(r_coarse))
            m_ncells = float(np.nanmean(r_ncells))

            print(f"  {fold.name} ({y_kind}): Cluster-Comp r={m_cluster:.4f} | Coarse-Comp r={m_coarse:.4f} | n_cells r={m_ncells:.4f}")

            records.append({
                "y_kind": y_kind,
                "fold": fold.name,
                "test_group": fold.meta["test_group"],
                "n_clusters": n_clusters,
                "r_cluster_composition": m_cluster,
                "r_coarse_composition": m_coarse,
                "r_ncells": m_ncells,
            })

    res_df = pd.DataFrame(records)
    out_csv = out_dir / "clustering_baseline.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved Step 2 clustering baseline results to {out_csv}")
    
    # Summary report
    summary = res_df.groupby("y_kind").agg({
        "r_cluster_composition": ["mean", "std"],
        "r_coarse_composition": ["mean", "std"],
        "r_ncells": ["mean", "std"],
    })
    print("\n=== Summary Table ===")
    print(summary.to_string())

    # Calculate average unassigned fractions
    coarse_unassigned = float(np.nanmean([spot_data[s]["unassigned_frac"] for s in cell_samples]))
    cluster_unassigned = 0.0
    print(f"\nUnassigned cell fraction: Coarse marker = {coarse_unassigned*100:.1f}%, Unsupervised clustering = {cluster_unassigned*100:.1f}%")

    json.dump({
        "coarse_unassigned_pct": round(coarse_unassigned * 100, 2),
        "cluster_unassigned_pct": 0.0,
        "n_samples_evaluated": len(cell_samples),
        "summary": res_df.to_dict(orient="records"),
    }, open(out_dir / "clustering_baseline_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
