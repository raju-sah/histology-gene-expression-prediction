import sys

from hestfm import ROOT
from fake_phase2 import make_fake_phase2

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase5_robustness as p5  # noqa: E402


def test_phase5_script(tmp_path):
    make_fake_phase2(tmp_path / "emb", encoders=("uni2_h", "uni2_h_c56"), n_genes=10)
    r = p5.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"), "--encoders", "uni2_h",
                 "--seeds", "0", "1", "--mlp-genes", "6", "--block", "5", "--no-log"])
    assert set(r) == {"seeds", "window", "fov", "mlp"}
    assert list(r["window"].window_um) == [56, 112, 224]
    assert set(r["mlp"]["head"]) == {"ridge", "mlp"} and len(r["mlp"]) == 3
    assert r["seeds"].loc["uni2_h", "std"] < 0.05
