import sys

import pandas as pd

from hestfm import ROOT
from fake_phase2 import make_fake_phase2

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase3_context  # noqa: E402


def test_phase3_script(tmp_path):
    make_fake_phase2(tmp_path / "emb", n_genes=12)
    summ, gsum = run_phase3_context.main(["--root", str(tmp_path / "emb"), "--out", str(tmp_path / "rep"),
                                          "--encoders", "uni2_h", "--attention-encoder", "uni2_h",
                                          "--attention-epochs", "3", "--block", "6", "--no-log",
                                          "--phase2-table", str(tmp_path / "none.csv")])
    assert set(summ.index) == {"uni2_h", "uni2_h_k1", "uni2_h_k2", "uni2_h_attn_k1"}
    assert set(gsum.index) == {"uni2_h_k1", "uni2_h_k2", "uni2_h_attn_k1"}
    t = pd.read_csv(tmp_path / "rep/context_gene_table.csv", index_col=0)
    assert {"gain_uni2_h_k1", "npos_uni2_h_k1", "r_uni2_h_attn_k1"} <= set(t.columns) and len(t) == 12
