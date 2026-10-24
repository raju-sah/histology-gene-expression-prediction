import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_phase2

STEMS = ["fig_r_distribution", "fig_r_vs_mean", "fig_r_vs_morans",
         "fig_r_by_celltype", "fig_encoder_agreement", "fig_image_vs_composition"]


def _make_table(n_genes=300, n_encoders=3, seed=0):
    rng = np.random.default_rng(seed)
    encoders = ["uni2_h", "virchow2", "conch_v15"][:n_encoders]
    mean_log1p = rng.gamma(2.0, 1.0, n_genes)
    var_log1p = mean_log1p * rng.uniform(0.5, 2.0, n_genes)
    frac_zero = np.clip(rng.normal(0.5, 0.2, n_genes), 0, 1)
    morans_i = rng.uniform(-0.1, 0.9, n_genes)
    tau = np.clip(rng.normal(0.4, 0.15, n_genes), 0, 1)
    top_type = rng.choice(["epithelial", "fibroblast", "t_nk", "myeloid", "endothelial"],
                          n_genes)
    agreement_mean = rng.uniform(0.5, 1.0, n_genes)
    data = {
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "var_log1p": var_log1p,
        "frac_zero": frac_zero,
        "morans_i": morans_i,
        "tau": tau,
        "top_type": top_type,
        "agreement_mean": agreement_mean,
        "mean_log1p": mean_log1p,
    }
    for k, enc in enumerate(encoders):
        r = 0.25 + k * 0.1 + 0.45 * (mean_log1p / mean_log1p.max()) \
            + rng.normal(0, 0.15, n_genes)
        col = "r_" + enc
        data[col] = np.clip(r, -1, 1)
    data["r_composition"] = np.clip(0.5 * (mean_log1p / mean_log1p.max())
                                    + rng.normal(0, 0.2, n_genes), -1, 1)
    data["r_ncells"] = np.clip(rng.normal(0.05, 0.1, n_genes), -1, 1)
    return pd.DataFrame(data)


def test_full_table_produces_all_files(tmp_path):
    table = tmp_path / "gene_table.csv"
    _make_table().to_csv(table, index=False)
    out = tmp_path / "figs"
    paths = plot_phase2.main(["--table", str(table), "--out", str(out)])
    expected = {s + ext for s in STEMS for ext in (".png", ".pdf")} | {"fig_summary.csv"}
    written = {Path(p).name for p in paths}
    assert written == expected
    for s in STEMS:
        assert (out / f"{s}.png").exists()
        assert (out / f"{s}.pdf").exists()
    summary = pd.read_csv(out / "fig_summary.csv")
    assert len(summary) == 5  # 3 encoders + 2 baselines
    assert list(summary.columns) == ["model", "n_genes", "mean_r", "median_r",
                                     "frac_r_gt_0_3", "frac_r_gt_0_5"]
    assert set(summary["model"]) == {"uni2_h", "virchow2", "conch_v15",
                                     "composition", "ncells"}
    assert summary["n_genes"].eq(300).all()


def test_single_encoder_and_few_genes_with_nan(tmp_path):
    table = tmp_path / "gene_table.csv"
    df = _make_table(n_genes=8, n_encoders=1)
    df.loc[0:2, "r_uni2_h"] = np.nan  # some NaN r values
    df.loc[3:4, "tau"] = np.nan
    df.loc[1:2, "agreement_mean"] = np.nan
    df.to_csv(table, index=False)
    out = tmp_path / "figs"
    paths = plot_phase2.main(["--table", str(table), "--out", str(out)])
    expected = {s + ext for s in STEMS for ext in (".png", ".pdf")} | {"fig_summary.csv"}
    assert {Path(p).name for p in paths} == expected
    summary = pd.read_csv(out / "fig_summary.csv")
    assert len(summary) == 3
