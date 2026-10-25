"""Phase 4: generalisation across samples and across platforms.

    .venv/bin/python scripts/run_phase4_generalization.py [--encoders uni2_h virchow2 conch_v15]

For every Xenium 5k test sample, on the genes shared by all Xenium 5k and same-organ Visium samples, compare:
  xenium_logo        : train on all other Xenium 5k groups (the Phase 2 setting, any organ)
  xenium_same_organ  : train only on other-group Xenium 5k samples of the same organ (lung, ovary; none for cervix)
  visium_same_organ  : train on HEST Visium samples of the same organ, test on the Xenium pseudo-spots
Targets are library-size normalised (`lognorm`) by default, because Visium and Xenium depths differ by orders of
magnitude. The drop is reported per test sample and gene as (setting - xenium_logo).
No sample/group is ever on both sides; Visium samples are separate studies (see configs/project.yaml).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.atlas import common_genes, load_sparse_samples, run_atlas_chunked  # noqa: E402
from hestfm.evaluate import log_experiment  # noqa: E402
from hestfm.splits import Fold, assert_no_leakage  # noqa: E402

XEN, VIS = "phase2/xenium5k", "phase4/visium"


def build_folds(xgroups, xorgan, vorgan, settings):
    folds = []
    for t in sorted(xgroups):
        g = xgroups[t]
        test = tuple(sorted(s for s in xgroups if xgroups[s] == g))
        if t != test[0]:
            continue   # one fold per test group
        org = xorgan[t]
        cand = {
            "xenium_logo": [s for s in xgroups if xgroups[s] != g],
            "xenium_same_organ": [s for s in xgroups if xgroups[s] != g and xorgan[s] == org],
            "visium_same_organ": [s for s in vorgan if vorgan[s] == org],
        }
        for name in settings:
            if cand[name]:
                folds.append(Fold(f"{name}|{g}", tuple(sorted(cand[name])), test, {"setting": name, "organ": org}))
    return folds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--encoders", nargs="+", default=["uni2_h", "virchow2", "conch_v15"])
    ap.add_argument("--settings", nargs="+", default=["xenium_logo", "xenium_same_organ", "visium_same_organ"])
    ap.add_argument("--y-kind", default="lognorm")
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=250, help="genes loaded per pass (memory bound)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase4"))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    root = Path(a.root)
    xg = {s: g for s, g in cfg["xenium5k_samples"].items() if (root / XEN / "expression" / f"{s}.h5").exists()}
    vo = {s: o for s, o in cfg["visium_organ"].items() if (root / VIS / "expression" / f"{s}.h5").exists()}
    xo = cfg["xenium5k_organ"]
    vo = {s: o for s, o in vo.items() if o in {xo[x] for x in xg}}
    encs = [e for e in a.encoders if all((root / XEN / e / f"{s}.h5").exists() for s in xg)
            and all((root / VIS / e / f"{s}.h5").exists() for s in vo)]
    print("xenium", list(xg), "| visium", list(vo), "| encoders", encs)

    genes = sorted(set(common_genes(root, XEN, list(xg))) & (set(common_genes(root, VIS, list(vo))) if vo else set()))
    print(len(genes), "shared genes")
    lean = dict(x_dtype=np.float16, keep_y=False)   # expression is loaded per gene chunk (memory)
    samples = load_sparse_samples(root, XEN, encs, list(xg), genes, **lean)
    if vo:
        samples.update(load_sparse_samples(root, VIS, encs, list(vo), genes, **lean))
    ds = {s: XEN for s in xg} | {s: VIS for s in vo}
    groups = dict(xg) | {s: f"visium_{s}" for s in vo}
    folds = build_folds(xg, xo, vo, a.settings)
    assert_no_leakage(folds, groups)

    for f in folds:
        print(f"== {f.name}: train {len(f.train)} samples -> test {f.test}")
    _, sg, _, _ = run_atlas_chunked(samples, folds, encs, groups, root, ds, genes, chunk=a.chunk,
                                    y_kind=a.y_kind, block=a.block, verbose=False)
    meta = {f.name: f for f in folds}
    sg["setting"] = sg.fold.map(lambda n: meta[n].meta["setting"])
    sg["organ"] = sg.fold.map(lambda n: meta[n].meta["organ"])
    sg["n_train_samples"] = sg.fold.map(lambda n: len(meta[n].train))
    sg["gene"] = np.asarray(genes)[sg.gene_idx.values]
    sg.to_csv(out / "sample_gene.csv.gz", index=False)

    per = sg.groupby(["setting", "encoder", "organ", "sample"]).pearson.mean().rename("mean_r").reset_index()
    ref = per[per.setting == "xenium_logo"].set_index(["encoder", "sample"]).mean_r
    per["drop_vs_logo"] = per.apply(lambda r: r.mean_r - ref.get((r.encoder, r["sample"]), np.nan), axis=1)
    per.to_csv(out / "per_sample.csv", index=False)
    wide = sg.pivot_table(index=["encoder", "sample", "gene"], columns="setting", values="pearson")
    gene_drop = pd.DataFrame({f"drop_{s}": wide[s] - wide["xenium_logo"] for s in wide.columns if s != "xenium_logo"})
    gene_drop.groupby(level=["encoder", "gene"]).mean().to_csv(out / "gene_drop.csv")
    summ = per.groupby(["setting", "encoder"]).agg(mean_r=("mean_r", "mean"), mean_drop_vs_logo=("drop_vs_logo", "mean"),
                                                   n_test_samples=("sample", "nunique")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    print(per.to_string()); print(summ.to_string())
    if not a.no_log:
        for r in summ.itertuples():
            log_experiment(r.encoder, "xenium5k_generalization", r.setting, "ridge_pergene_alpha_pca256",
                           "mean_per_sample_r", float(r.mean_r), n_test=r.n_test_samples,
                           notes=f"y={a.y_kind};genes={len(genes)};drop_vs_logo={r.mean_drop_vs_logo:.4f}")
    return per, summ


if __name__ == "__main__":
    main()
