import json
import sys

import numpy as np
import pandas as pd

from hestfm import ROOT
from hestfm.data import write_embeddings, write_expression
from conftest import make_samples

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase1_eval  # noqa: E402


def test_phase1_eval_on_fake_embeddings(tmp_path):
    meta = tmp_path / "meta"
    s = make_samples(n_samples=3, n=300, d=300, g=50)
    genes = list(next(iter(s.values())).genes)
    (meta / "T" / "splits").mkdir(parents=True)
    json.dump({"genes": genes}, open(meta / "T" / "var_50genes.json", "w"))
    ids = list(s)
    for i, sid in enumerate(ids):
        pd.DataFrame({"sample_id": [x for x in ids if x != sid]}).to_csv(meta / f"T/splits/train_{i}.csv", index=False)
        pd.DataFrame({"sample_id": [sid]}).to_csv(meta / f"T/splits/test_{i}.csv", index=False)
    for sid, v in s.items():
        write_embeddings(tmp_path / f"emb/phase1/T/uni2_h/{sid}.h5", v.X, v.barcodes, v.coords)
        write_expression(tmp_path / f"emb/phase1/T/expression/{sid}.h5", v.counts, v.barcodes, v.genes, v.coords)
    df, flags = run_phase1_eval.main(["--root", str(tmp_path / "emb"), "--bench-meta", str(meta), "--tasks", "T",
                                      "--encoders", "uni2_h", "resnet50", "--heads", "hest_ridge", "ridge_cv",
                                      "--out", str(tmp_path / "rep"), "--no-log"])
    assert len(df) == 2 and (df.pearson > 0.3).all()
    assert (tmp_path / "rep/benchmark_table.md").exists()
