"""Phase 1: evaluate downloaded HEST-Benchmark embeddings and compare with the published table.

    .venv/bin/python scripts/run_phase1_eval.py [--root data/embeddings] [--heads hest_ridge ridge_cv mlp]

Writes reports/phase1/{benchmark_table.csv,benchmark_table.md,per_gene.csv} and appends to experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402
from hestfm.evaluate import load_samples, log_experiment, run_cv  # noqa: E402
from hestfm.heads import make_head  # noqa: E402
from hestfm.splits import folds_from_hest_bench  # noqa: E402

PUBLISHED = {  # HEST README table dated 03.04.26 (RECON.md section 3)
    "resnet50": {"IDC": 0.4739, "PRAD": 0.3044, "PAAD": 0.3880, "SKCM": 0.4821, "COAD": 0.2500, "READ": 0.0783,
                 "CCRCC": 0.2252, "LUNG": 0.4949, "LYMPH_IDC": 0.2305},
    "uni2_h": {"IDC": 0.5898, "PRAD": 0.3569, "PAAD": 0.5001, "SKCM": 0.6606, "COAD": 0.3015, "READ": 0.2223,
               "CCRCC": 0.2640, "LUNG": 0.5587, "LYMPH_IDC": 0.2727},
    "virchow2": {"IDC": 0.5971, "LUNG": 0.5685, "COAD": 0.2581},
    "conch_v15": {"IDC": 0.5440, "LUNG": 0.5513, "COAD": 0.2802},
}
DIFF_TOL = 0.03          # |ours - published| above this on the exact protocol => investigate
TOO_GOOD = 0.05          # ours above published by this much => check for leakage first


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "embeddings"))
    ap.add_argument("--bench-meta", default=str(ROOT / "data" / "meta" / "hest-bench"))
    ap.add_argument("--tasks", nargs="+", default=load_config()["phase1_tasks"])
    ap.add_argument("--encoders", nargs="+", default=["resnet50", "uni2_h"])
    ap.add_argument("--heads", nargs="+", default=["hest_ridge", "ridge_cv", "mlp"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase1"))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, genes_rows, flags = [], [], []
    for task in a.tasks:
        folds = folds_from_hest_bench(Path(a.bench_meta) / task)
        genes = json.load(open(Path(a.bench_meta) / task / "var_50genes.json"))["genes"]
        ids = sorted({s for f in folds for s in f.train + f.test})
        for enc in a.encoders:
            if not (Path(a.root) / "phase1" / task / enc).exists():
                print(f"skip {task}/{enc}: no embeddings")
                continue
            samples = load_samples(a.root, f"phase1/{task}", enc, ids, genes)
            for head in a.heads:
                kw = {"latent_dim": 256, "seed": 1} if head == "hest_ridge" else {}
                res = run_cv(samples, folds, lambda: make_head(head, **kw), use_val=head != "hest_ridge")
                p, lo, hi = res.sample_ci()
                pub = PUBLISHED.get(enc, {}).get(task, np.nan)
                rows.append({"task": task, "encoder": enc, "head": head, "pearson": res.pearson_mean,
                             "ci_lo": lo, "ci_hi": hi, "published": pub, "diff": res.pearson_mean - pub,
                             "folds": ";".join(f"{x:.4f}" for x in res.folds["mean_pearson"]),
                             "n_test": int(res.folds["n_test"].sum())})
                gt = res.gene_table().reset_index().assign(task=task, encoder=enc, head=head)
                genes_rows.append(gt)
                if head == "hest_ridge" and abs(res.pearson_mean - pub) > DIFF_TOL:
                    flags.append(f"{task}/{enc}: ours {res.pearson_mean:.4f} vs published {pub:.4f} (> {DIFF_TOL})")
                if res.pearson_mean - pub > TOO_GOOD:
                    flags.append(f"{task}/{enc}/{head}: {res.pearson_mean:.4f} exceeds published {pub:.4f} by > {TOO_GOOD}: check leakage")
                if not a.no_log:
                    log_experiment(enc, task, "hest_bench_folds", head, "pearson_mean", res.pearson_mean,
                                   n_test=rows[-1]["n_test"], notes=f"ci95=[{lo:.3f},{hi:.3f}] published={pub}")
                print(f"{task:6s} {enc:10s} {head:11s} r={res.pearson_mean:.4f} (pub {pub:.4f}) folds={rows[-1]['folds']}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results")
        return df, flags
    df.to_csv(out / "benchmark_table.csv", index=False)
    pd.concat(genes_rows).to_csv(out / "per_gene.csv", index=False)
    with open(out / "benchmark_table.md", "w") as f:
        f.write("| task | encoder | head | Pearson (ours) | 95% CI (samples) | published | diff |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['task']} | {r['encoder']} | {r['head']} | {r['pearson']:.4f} | "
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['published']:.4f} | {r['diff']:+.4f} |\n")
        if flags:
            f.write("\n**Flags**\n\n" + "\n".join(f"- {x}" for x in flags) + "\n")
    print("\n".join(["FLAGS:"] + flags) if flags else "no flags")
    return df, flags


if __name__ == "__main__":
    main()
