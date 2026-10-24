import sys

from hestfm import ROOT, load_config
from hestfm.splits import assert_no_leakage
from fake_phase2 import make_fake_phase2

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase4_generalization as p4  # noqa: E402


def test_folds_are_leak_free_and_organ_matched():
    cfg = load_config()
    xg, xo, vo = cfg["xenium5k_samples"], cfg["xenium5k_organ"], cfg["visium_organ"]
    folds = p4.build_folds(xg, xo, vo, ["xenium_logo", "xenium_same_organ", "visium_same_organ"])
    groups = dict(xg) | {s: f"visium_{s}" for s in vo}
    assert_no_leakage(folds, groups)
    for f in folds:
        org = f.meta["organ"]
        assert all(xo[s] == org for s in f.test)
        if f.meta["setting"] != "xenium_logo":
            assert all((xo.get(s) or vo.get(s)) == org for s in f.train)
    names = {f.name for f in folds}
    assert "xenium_same_organ|cervix_10x" not in names        # no other cervix Xenium sample
    assert "visium_same_organ|cervix_10x" in names
    nc = [f for f in folds if f.name == "xenium_same_organ|lung_luad_10x"][0]
    assert set(nc.train) == {"NCBI887", "NCBI888"}


def test_phase4_script(tmp_path):
    make_fake_phase2(tmp_path / "emb", n_genes=10)
    make_fake_phase2(tmp_path / "emb", n_genes=10, dataset="phase4/visium", samples=list(load_config()["visium_organ"]), seed=0)
    per, summ = p4.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"), "--encoders", "uni2_h",
                         "--block", "5", "--no-log"])
    assert set(summ.setting) == {"xenium_logo", "xenium_same_organ", "visium_same_organ"}
    assert per[per.setting == "xenium_logo"].drop_vs_logo.abs().max() < 1e-12
    assert (tmp_path / "rep/gene_drop.csv").exists()
