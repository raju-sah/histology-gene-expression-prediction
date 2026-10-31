"""Build release artifacts for Hugging Face Dataset and Hugging Face Space.

Ensures:
- Only per-gene and per-model summary tables are copied.
- No barcodes, spot coordinates, or per-spot expression matrices are copied.
- Copies are placed into release/hf_dataset/ and release/hf_space/data/.
- Paper figures (PNG only) are copied to release/hf_space/figures/.
"""
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HF_DATASET = ROOT / "release" / "hf_dataset"
HF_SPACE = ROOT / "release" / "hf_space"

FORBIDDEN_COLUMNS = {"barcode", "barcodes", "he_x", "he_y", "x", "y", "coords", "spot_id", "cell_id"}

def verify_and_copy(df: pd.DataFrame, out_path: Path):
    cols = set(df.columns)
    forbidden = cols & FORBIDDEN_COLUMNS
    assert not forbidden, f"Forbidden columns found in {out_path.name}: {forbidden}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path.name}: {df.shape[0]} rows, {df.shape[1]} cols")

def main():
    HF_DATASET.mkdir(parents=True, exist_ok=True)
    (HF_SPACE / "data").mkdir(parents=True, exist_ok=True)
    (HF_SPACE / "figures" / "log1p").mkdir(parents=True, exist_ok=True)
    (HF_SPACE / "figures" / "lognorm").mkdir(parents=True, exist_ok=True)

    # 1. Atlas gene tables
    for t in ["log1p", "lognorm"]:
        df = pd.read_csv(REPORTS / f"phase2/{t}/gene_table.csv")
        verify_and_copy(df, HF_DATASET / f"atlas_{t}.csv")

    # 2. Phase 2 summary table combining log1p and lognorm
    s_log1p = pd.read_csv(REPORTS / "phase2/log1p/summary.csv")
    s_lognorm = pd.read_csv(REPORTS / "phase2/lognorm/summary.csv")
    s_combined = pd.concat([s_log1p, s_lognorm], ignore_index=True)
    verify_and_copy(s_combined, HF_DATASET / "summary_phase2.csv")

    # 3. Context gain gene tables and phase 3 summary
    for t in ["log1p", "lognorm"]:
        df = pd.read_csv(REPORTS / f"phase3/{t}/context_gene_table.csv")
        verify_and_copy(df, HF_DATASET / f"context_gain_{t}.csv")

    p3_s_log1p = pd.read_csv(REPORTS / "phase3/log1p/summary.csv")
    p3_s_log1p["target"] = "log1p"
    p3_s_lognorm = pd.read_csv(REPORTS / "phase3/lognorm/summary.csv")
    p3_s_lognorm["target"] = "lognorm"
    p3_combined = pd.concat([p3_s_log1p, p3_s_lognorm], ignore_index=True)
    verify_and_copy(p3_combined, HF_DATASET / "summary_phase3.csv")

    # 4. Generalization summary
    df_gen = pd.read_csv(REPORTS / "phase4/summary.csv")
    verify_and_copy(df_gen, HF_DATASET / "generalization_summary.csv")

    # 5. Robustness tables
    verify_and_copy(pd.read_csv(REPORTS / "phase5/seeds_summary.csv"), HF_DATASET / "robustness_seeds.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/heads.csv"), HF_DATASET / "robustness_heads.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/window_size.csv"), HF_DATASET / "robustness_window_log1p.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/lognorm/window_size.csv"), HF_DATASET / "robustness_window_lognorm.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/field_of_view.csv"), HF_DATASET / "robustness_fov_log1p.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/lognorm/field_of_view.csv"), HF_DATASET / "robustness_fov_lognorm.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/field_of_view_genes.csv"), HF_DATASET / "robustness_fov_genes_log1p.csv")
    verify_and_copy(pd.read_csv(REPORTS / "phase5/lognorm/field_of_view_genes.csv"), HF_DATASET / "robustness_fov_genes_lognorm.csv")

    # Bundle copies to HF Space data directory
    for f in HF_DATASET.glob("*.csv"):
        shutil.copy(f, HF_SPACE / "data" / f.name)
    print("All dataset CSVs copied to HF Space data directory.")

    # Bundle PNG figures from reports/phase2/figures
    for target in ["log1p", "lognorm"]:
        fig_dir = REPORTS / f"phase2/figures/{target}"
        if fig_dir.exists():
            for png in fig_dir.glob("*.png"):
                shutil.copy(png, HF_SPACE / "figures" / target / png.name)
    print("All paper PNG figures copied to HF Space figures directory.")

if __name__ == "__main__":
    main()
