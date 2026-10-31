"""Phase 5: robustness of the Phase 2 atlas.

    .venv/bin/python scripts/run_phase5_robustness.py [--seeds 0 1 2] [--mlp-genes 1000]

1. Seeds: re-run the atlas ridge with different seeds (PCA randomisation + validation tie-breaks), per encoder.
2. Expression window: pseudo-spots of 0.5x / 1x / 2x the 112 um patch footprint (`expression_w050`, `expression`,
   `expression_w200`), same embeddings. Only if those folders were downloaded.
3. Image field of view: UNI2-h on the full 112 um patch vs the central 56 um (`uni2_h_c56`), same targets.
4. Head: ridge vs one-hidden-layer MLP (3 seeds) on a fixed random gene subset, best encoder.
All on the LOGO folds. Outputs: reports/phase5/*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.atlas import shared_fit_cache, common_genes, load_sparse_samples, run_atlas_chunked  # noqa: E402
from hestfm.data import SampleData  # noqa: E402
from hestfm.evaluate import log_experiment, run_cv  # noqa: E402
from hestfm.heads import MLPHead, RidgeCVHead  # noqa: E402
from hestfm.metrics import bootstrap_ci  # noqa: E402
from hestfm.splits import leave_one_group_out  # noqa: E402

DATASET = "phase2/xenium5k"
WINDOWS = {"expression_w050": 56, "expression": 112, "expression_w200": 224}
LEAN = dict(x_dtype=np.float16, keep_y=False)   # expression loaded per gene chunk (memory)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _device()


def score(gf, sg, enc):
    per_fold = gf[gf.encoder == enc].groupby("fold").pearson.mean()
    m = sg[sg.encoder == enc].pivot_table(index="sample", columns="gene_idx", values="pearson")
    p, lo, hi = bootstrap_ci(m.values)
    return float(per_fold.mean()), lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--encoders", nargs="+", default=["uni2_h", "virchow2", "conch_v15", "hoptimus1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mlp-genes", type=int, default=1000)
    ap.add_argument("--mlp-encoder", default="uni2_h")
    ap.add_argument("--y-kind", default="log1p")
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=250, help="genes loaded per pass (memory bound)")
    ap.add_argument("--seed-genes", type=int, default=1000, help="fixed random gene subset for the seed analysis; 0 = all")
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase5"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["seeds", "window", "fov", "mlp"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    groups = {s: g for s, g in load_config()["xenium5k_samples"].items()
              if (root / DATASET / "expression" / f"{s}.h5").exists()}
    ids = list(groups)
    folds = leave_one_group_out(groups)
    encs = [e for e in a.encoders if all((root / DATASET / e / f"{s}.h5").exists() for s in ids)]
    genes = common_genes(root, DATASET, ids)
    log = (lambda *x, **k: None) if a.no_log else log_experiment
    results = {}

    if "seeds" not in a.skip:
        rng = np.random.default_rng(0)
        gseed = genes if a.seed_genes <= 0 or a.seed_genes >= len(genes) else \
            sorted(rng.choice(genes, size=a.seed_genes, replace=False))   # fixed subset: CPU budget
        samples = load_sparse_samples(root, DATASET, encs, ids, genes, **LEAN)
        rows = []
        for seed in a.seeds:
            gf, sg, _, _ = run_atlas_chunked(samples, folds, encs, groups, root, DATASET, gseed, chunk=a.chunk, fit_cache=shared_fit_cache(root),
                                             y_kind=a.y_kind, block=a.block, seed=seed, verbose=False)
            for e in encs:
                m, lo, hi = score(gf, sg, e)
                rows.append({"encoder": e, "seed": seed, "pearson_mean": m, "ci_lo": lo, "ci_hi": hi, "n_genes": len(gseed)})
                log(e, "xenium5k_atlas", "logo_5groups", f"ridge_seed{seed}", "pearson_mean", m, notes=f"y={a.y_kind}")
        df = pd.DataFrame(rows)
        df.to_csv(out / "seeds.csv", index=False)
        agg = df.groupby("encoder").pearson_mean.agg(["mean", "std", "min", "max"])
        agg.to_csv(out / "seeds_summary.csv")
        results["seeds"] = agg
        print(agg.to_string())
        del samples, gf, sg
        import gc
        gc.collect()

    if "window" not in a.skip:
        have = {w: um for w, um in WINDOWS.items() if all((root / DATASET / w / f"{s}.h5").exists() for s in ids)}
        rows = []
        if len(have) > 1:
            for w, um in have.items():
                g_w = common_genes(root, DATASET, ids, subdir=w)
                g_w = sorted(set(g_w) & set(genes))
                samples = load_sparse_samples(root, DATASET, encs[:1], ids, g_w, subdir=w, **LEAN)
                gf, sg, _, _ = run_atlas_chunked(samples, folds, encs[:1], groups, root, DATASET, g_w, chunk=a.chunk, fit_cache=shared_fit_cache(root),
                                                 subdir=w, y_kind=a.y_kind, block=a.block, verbose=False)
                m, lo, hi = score(gf, sg, encs[0])
                rows.append({"encoder": encs[0], "window_um": um, "pearson_mean": m, "ci_lo": lo, "ci_hi": hi,
                             "n_genes": len(g_w)})
                log(encs[0], "xenium5k_atlas", "logo_5groups", f"ridge_window{um}um", "pearson_mean", m, notes=f"y={a.y_kind}")
                del samples, gf, sg
                import gc
                gc.collect()
            results["window"] = pd.DataFrame(rows)
            results["window"].to_csv(out / "window_size.csv", index=False)
            print(results["window"].to_string())
        else:
            print("window sensitivity skipped: expression_w050 / expression_w200 not downloaded")

    if "fov" not in a.skip:
        if all((root / DATASET / "uni2_h_c56" / f"{s}.h5").exists() for s in ids) and "uni2_h" in encs:
            samples = load_sparse_samples(root, DATASET, ["uni2_h", "uni2_h_c56"], ids, genes, **LEAN)
            gf, sg, _, _ = run_atlas_chunked(samples, folds, ["uni2_h", "uni2_h_c56"], groups, root, DATASET, genes,
                                             chunk=a.chunk, fit_cache=shared_fit_cache(root), y_kind=a.y_kind, block=a.block, verbose=False)
            rows = []
            for e, um in (("uni2_h", 112), ("uni2_h_c56", 56)):
                m, lo, hi = score(gf, sg, e)
                rows.append({"encoder": e, "fov_um": um, "pearson_mean": m, "ci_lo": lo, "ci_hi": hi})
                log(e, "xenium5k_atlas", "logo_5groups", f"ridge_fov{um}um", "pearson_mean", m, notes=f"y={a.y_kind}")
            results["fov"] = pd.DataFrame(rows)
            results["fov"].to_csv(out / "field_of_view.csv", index=False)
            g = gf.pivot_table(index="gene_idx", columns="encoder", values="pearson", aggfunc="mean")
            pd.DataFrame({"gene": np.asarray(genes)[g.index], "r_112um": g["uni2_h"], "r_56um": g["uni2_h_c56"]}).to_csv(
                out / "field_of_view_genes.csv", index=False)
            print(results["fov"].to_string())
            del samples, gf, sg
            import gc
            gc.collect()
        else:
            print("field-of-view sensitivity skipped: uni2_h_c56 not downloaded")

    if "mlp" not in a.skip and a.mlp_encoder in encs:
        rng = np.random.default_rng(0)
        gsub = sorted(rng.choice(genes, size=min(a.mlp_genes, len(genes)), replace=False))
        sp_samples = load_sparse_samples(root, DATASET, [a.mlp_encoder], ids, gsub, x_dtype=np.float16)
        dense = {s: SampleData(s, v.X[a.mlp_encoder].astype(np.float32), v.Y.toarray(), v.barcodes, v.coords,
                               np.asarray(gsub))
                 for s, v in sp_samples.items()}
        del sp_samples
        import gc
        gc.collect()
        rows = []
        heads = {"ridge": lambda seed: RidgeCVHead(pca_dim=256, per_gene_alpha=True, seed=seed),
                 "mlp": lambda seed: MLPHead(hidden=512, pca_dim=256, seed=seed, max_epochs=60, patience=8, device=DEVICE)}
        for name, mk in heads.items():
            for seed in (a.seeds if name == "mlp" else a.seeds[:1]):
                res = run_cv(dense, folds, lambda: mk(seed), groups=groups, seed=seed)
                p, lo, hi = res.sample_ci()
                rows.append({"head": name, "seed": seed, "pearson_mean": res.pearson_mean, "ci_lo": lo, "ci_hi": hi,
                             "n_genes": len(gsub)})
                log(a.mlp_encoder, "xenium5k_atlas_subset", "logo_5groups", f"{name}_seed{seed}", "pearson_mean",
                    res.pearson_mean, notes=f"genes={len(gsub)} random subset")
                print(rows[-1])
        results["mlp"] = pd.DataFrame(rows)
        results["mlp"].to_csv(out / "heads.csv", index=False)
    return results


if __name__ == "__main__":
    main()
