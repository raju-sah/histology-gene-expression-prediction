import sys

import pandas as pd

from hestfm import ROOT
from fake_phase2 import make_fake_phase2

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase2_atlas  # noqa: E402


def test_atlas_script_end_to_end(tmp_path):
    make_fake_phase2(tmp_path / "emb")
    summ, flags = run_phase2_atlas.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"),
                                         "--encoders", "uni2_h", "virchow2", "hoptimus1", "--block", "16",
                                         "--null-genes", "20", "--no-log"])
    assert set(summ.model) == {"uni2_h", "virchow2", "composition", "ncells", "composition_ncells"}   # hoptimus1 absent -> skipped
    assert not flags, flags
    s = summ[summ.y_kind == "log1p"].set_index("model")
    assert s.loc["uni2_h", "pearson_mean"] > 0.2 > abs(s.loc["ncells", "pearson_mean"])
    t = pd.read_csv(tmp_path / "rep/log1p/gene_table.csv", index_col=0)
    for c in ["r_uni2_h", "r_virchow2", "r_composition", "r_ncells", "mean_log1p", "frac_zero", "morans_i", "tau",
              "top_type", "agreement_mean", "n_folds_evaluable"]:
        assert c in t.columns
    assert len(t) == 40
    # informative genes (first half) beat uninformative ones
    assert t.loc[[f"G{i:03d}" for i in range(20)], "r_uni2_h"].mean() > t.loc[[f"G{i:03d}" for i in range(20, 40)], "r_uni2_h"].mean() + 0.2


def test_explain_script(tmp_path):
    import analyze_predictability
    make_fake_phase2(tmp_path / "emb")
    run_phase2_atlas.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"), "--encoders", "uni2_h",
                           "virchow2", "--block", "16", "--null-genes", "10", "--no-log", "--y-kind", "log1p"])
    sp_df, ols = analyze_predictability.main(["--table", str(tmp_path / "rep/log1p/gene_table.csv")])
    assert {"mean_log1p", "morans_i", "agreement_mean"} <= set(sp_df.feature)
    assert (tmp_path / "rep/log1p/explain.md").read_text().startswith("# What explains")


def test_plot_on_real_atlas_output(tmp_path):
    import plot_phase2
    make_fake_phase2(tmp_path / "emb")
    run_phase2_atlas.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"), "--encoders", "uni2_h",
                           "virchow2", "--block", "16", "--null-genes", "10", "--no-log", "--y-kind", "log1p"])
    plot_phase2.main(["--table", str(tmp_path / "rep/log1p/gene_table.csv"), "--out", str(tmp_path / "fig")])
    s = pd.read_csv(tmp_path / "fig/fig_summary.csv")
    assert set(s.model) == {"uni2_h", "virchow2", "composition", "ncells", "composition_ncells"}
