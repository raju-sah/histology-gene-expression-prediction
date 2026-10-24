"""Render the Xenium / Xenium 5k sample list (from HEST metadata) as a markdown table."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/meta/xenium_samples.csv", index_col=0)
cols = ["st_technology", "organ", "disease_state", "oncotree_code", "species", "patient",
        "spots_under_tissue", "nb_genes", "preservation_method", "patches", "transcripts"]
out = df[cols].rename(columns={"spots_under_tissue": "spots", "patches": "patches_GB",
                                "transcripts": "transcripts_GB", "st_technology": "tech"})
out["patient"] = out["patient"].fillna("n/a")
out["oncotree_code"] = out["oncotree_code"].fillna("-")
print(out.to_markdown())
