"""Run all analysis scripts on fake data and check every \\num{...} macro used by paper/main.tex gets a value."""
import re
import sys

from hestfm import ROOT, load_config
from fake_phase2 import make_fake_phase2

sys.path.insert(0, str(ROOT / "scripts"))


def test_all_paper_macros_resolve(tmp_path):
    import analyze_predictability, make_paper_numbers, plot_phase2, run_phase2_atlas, run_phase3_context
    import run_phase4_generalization, run_phase5_robustness
    emb, rep = tmp_path / "emb", tmp_path / "reports"
    make_fake_phase2(emb, encoders=("uni2_h", "virchow2", "uni2_h_c56"), n_genes=12)
    make_fake_phase2(emb, encoders=("uni2_h", "virchow2"), n_genes=12, dataset="phase4/visium",
                     samples=list(load_config()["visium_organ"]))
    common = ["--root", str(emb), "--no-log", "--block", "6"]
    run_phase2_atlas.main(common + ["--out", str(rep / "phase2"), "--null-genes", "6"])
    analyze_predictability.main(["--table", str(rep / "phase2/log1p/gene_table.csv")])
    plot_phase2.main(["--table", str(rep / "phase2/log1p/gene_table.csv"), "--out", str(rep / "phase2/figures")])
    run_phase3_context.main(common + ["--out", str(rep / "phase3"), "--encoders", "uni2_h", "--attention-epochs", "2",
                                      "--phase2-table", str(rep / "phase2/log1p/gene_table.csv")])
    run_phase4_generalization.main(common + ["--out", str(rep / "phase4"), "--encoders", "uni2_h", "virchow2"])
    run_phase5_robustness.main(common + ["--out", str(rep / "phase5"), "--encoders", "uni2_h", "--seeds", "0", "1",
                                         "--mlp-genes", "6"])
    # fake Phase 1 table
    (rep / "phase1").mkdir(parents=True)
    (rep / "phase1/benchmark_table.csv").write_text(
        "task,encoder,head,pearson,ci_lo,ci_hi,published,diff,folds,n_test\nIDC,uni2_h,hest_ridge,0.58,0.5,0.6,0.5898,-0.01,0.5,100\n")
    m = make_paper_numbers.main(["--reports", str(rep), "--out", str(tmp_path / "gen")])
    used = set(re.findall(r"\\num\{(\w+)\}", (ROOT / "paper/main.tex").read_text()))
    missing = sorted(u for u in used if u not in m or m[u] == make_paper_numbers.PENDING)
    assert not missing, missing
