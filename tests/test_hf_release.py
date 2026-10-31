r"""Tests for Hugging Face Dataset and Space release artifacts.

Verifies:
- All required CSVs exist in release/hf_dataset/ and release/hf_space/data/.
- Required columns exist in every dataset.
- No per-spot, coordinate, or barcode columns exist.
- Per-gene table row counts match \AtlasNGenes (5,001) from paper/generated/numbers.tex.
- Space app callbacks and figures exist and operate correctly.
"""
from pathlib import Path
import re
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = ROOT / "release" / "hf_dataset"
HF_SPACE = ROOT / "release" / "hf_space"
NUMBERS_TEX = ROOT / "paper" / "generated" / "numbers.tex"

FORBIDDEN_COLUMNS = {
    "barcode", "barcodes", "he_x", "he_y", "x", "y", "coords",
    "spot_id", "cell_id", "pseudospot_id", "spot"
}

EXPECTED_FILES = [
    "atlas_log1p.csv",
    "atlas_lognorm.csv",
    "summary_phase2.csv",
    "context_gain_log1p.csv",
    "context_gain_lognorm.csv",
    "summary_phase3.csv",
    "generalization_summary.csv",
    "robustness_seeds.csv",
    "robustness_heads.csv",
    "robustness_window_log1p.csv",
    "robustness_window_lognorm.csv",
    "robustness_fov_log1p.csv",
    "robustness_fov_lognorm.csv",
    "robustness_fov_genes_log1p.csv",
    "robustness_fov_genes_lognorm.csv",
]

def get_expected_ngenes():
    text = NUMBERS_TEX.read_text()
    m = re.search(r"\\newcommand\{\\AtlasNGenes\}\{(\d+)\}", text)
    assert m is not None, "Could not find \\AtlasNGenes in numbers.tex"
    return int(m.group(1))

@pytest.mark.parametrize("folder_name", ["hf_dataset", "hf_space/data"])
def test_all_release_csvs_exist(folder_name):
    folder = ROOT / "release" / folder_name
    assert folder.exists(), f"Release folder {folder} does not exist"
    for fname in EXPECTED_FILES:
        path = folder / fname
        assert path.exists(), f"Missing expected release file: {path}"

def test_no_forbidden_columns_in_release_csvs():
    for f in HF_DATASET.glob("*.csv"):
        df = pd.read_csv(f, nrows=5)
        forbidden = set(df.columns) & FORBIDDEN_COLUMNS
        assert not forbidden, f"Forbidden per-spot column(s) {forbidden} found in {f.name}"

def test_gene_counts_match_numbers_tex():
    expected_ngenes = get_expected_ngenes()
    gene_files = [
        "atlas_log1p.csv",
        "atlas_lognorm.csv",
        "context_gain_log1p.csv",
        "context_gain_lognorm.csv",
        "robustness_fov_genes_log1p.csv",
        "robustness_fov_genes_lognorm.csv",
    ]
    for fname in gene_files:
        df = pd.read_csv(HF_DATASET / fname)
        assert len(df) == expected_ngenes, (
            f"{fname} has {len(df)} rows, expected {expected_ngenes} matching \\AtlasNGenes"
        )
        assert "gene" in df.columns, f"{fname} missing 'gene' column"

def test_atlas_required_columns():
    for target in ["log1p", "lognorm"]:
        df = pd.read_csv(HF_DATASET / f"atlas_{target}.csv", nrows=5)
        required = [
            "gene", "r_virchow2", "r_uni2_h", "r_conch_v15",
            "r_composition", "r_ncells", "r_image_minus_composition",
            "morans_i", "tau", "top_type"
        ]
        for col in required:
            assert col in df.columns, f"Missing required column {col} in atlas_{target}.csv"

def test_summary_phase2_columns():
    df = pd.read_csv(HF_DATASET / "summary_phase2.csv")
    required = ["model", "pearson_mean", "ci_lo", "ci_hi", "y_kind"]
    for col in required:
        assert col in df.columns, f"Missing {col} in summary_phase2.csv"
    assert set(df["y_kind"].unique()) == {"log1p", "lognorm"}

def test_space_app_callbacks():
    import sys
    sys.path.insert(0, str(HF_SPACE))
    import app

    # Search gene test
    card = app.search_gene("F11R", "lognorm")
    assert "F11R" in card
    assert "Virchow2" in card
    assert "Cell Composition" in card

    # Filter table test
    tbl = app.filter_gene_table("lognorm", "All Genes", "All", "")
    assert len(tbl) == get_expected_ngenes()

    tbl_beats = app.filter_gene_table("lognorm", "Image beats composition", "All", "")
    assert len(tbl_beats) > 0

    # Scatter plot test
    fig = app.plot_scatter("lognorm")
    assert fig is not None

    # Paper figures test
    for target in ["log1p", "lognorm"]:
        figs = app.get_figures(target)
        assert len(figs) == 6
        for fpath in figs:
            assert fpath is not None and Path(fpath).exists(), f"Figure {fpath} does not exist"
