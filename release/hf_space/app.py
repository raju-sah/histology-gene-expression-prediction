"""Gradio application for HEST Xenium 5k Gene Expression Predictability Atlas Explorer.

Benchmark exploring pathology foundation models (Virchow2, UNI2-h, CONCH v1.5)
predicting spatial gene expression across 5,001 genes in 6 Xenium 5k samples.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
def load_atlas(target: str) -> pd.DataFrame:
    """Load atlas and merge context gains."""
    atlas = pd.read_csv(DATA_DIR / f"atlas_{target}.csv")
    ctx_file = DATA_DIR / f"context_gain_{target}.csv"
    if ctx_file.exists():
        ctx = pd.read_csv(ctx_file)[["gene", "gain_virchow2_k1", "gain_virchow2_k2"]]
        df = pd.merge(atlas, ctx, on="gene", how="left")
    else:
        df = atlas.copy()
        df["gain_virchow2_k1"] = np.nan
        df["gain_virchow2_k2"] = np.nan
    return df

ATLAS_DATA = {
    "lognorm": load_atlas("lognorm"),
    "log1p": load_atlas("log1p"),
}

SUMMARY_PHASE2 = pd.read_csv(DATA_DIR / "summary_phase2.csv") if (DATA_DIR / "summary_phase2.csv").exists() else pd.DataFrame()
SUMMARY_PHASE3 = pd.read_csv(DATA_DIR / "summary_phase3.csv") if (DATA_DIR / "summary_phase3.csv").exists() else pd.DataFrame()
GEN_SUMMARY = pd.read_csv(DATA_DIR / "generalization_summary.csv") if (DATA_DIR / "generalization_summary.csv").exists() else pd.DataFrame()
ROBUST_SEEDS = pd.read_csv(DATA_DIR / "robustness_seeds.csv") if (DATA_DIR / "robustness_seeds.csv").exists() else pd.DataFrame()
ROBUST_HEADS = pd.read_csv(DATA_DIR / "robustness_heads.csv") if (DATA_DIR / "robustness_heads.csv").exists() else pd.DataFrame()

# ---------------------------------------------------------
# Callback Handlers
# ---------------------------------------------------------
def search_gene(gene_symbol: str, target: str) -> str:
    """Return markdown summary for a searched gene."""
    if not gene_symbol or not gene_symbol.strip():
        return "Please enter a gene symbol (e.g. `F11R`, `EPCAM`, `CD3D`, `ERBB2`, `MKI67`)."
    
    symbol = gene_symbol.strip().upper()
    df = ATLAS_DATA[target]
    row = df[df["gene"].str.upper() == symbol]
    
    if row.empty:
        # Suggest close matches
        matches = df[df["gene"].str.contains(symbol, case=False, na=False)]["gene"].head(5).tolist()
        hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
        return f"Gene **{symbol}** not found in the 5,001 Xenium 5k panel.{hint}"
    
    r = row.iloc[0]
    
    v2 = f"{r['r_virchow2']:.3f}" if pd.notna(r.get('r_virchow2')) else "N/A"
    uni = f"{r['r_uni2_h']:.3f}" if pd.notna(r.get('r_uni2_h')) else "N/A"
    conch = f"{r['r_conch_v15']:.3f}" if pd.notna(r.get('r_conch_v15')) else "N/A"
    comp = f"{r['r_composition']:.3f}" if pd.notna(r.get('r_composition')) else "N/A"
    ncells = f"{r['r_ncells']:.3f}" if pd.notna(r.get('r_ncells')) else "N/A"
    diff = f"{r['r_image_minus_composition']:+.3f}" if pd.notna(r.get('r_image_minus_composition')) else "N/A"
    moran = f"{r['morans_i']:.3f}" if pd.notna(r.get('morans_i')) else "N/A"
    mean_exp = f"{r['mean_log1p']:.3f}" if pd.notna(r.get('mean_log1p')) else "N/A"
    tau = f"{r['tau']:.3f}" if pd.notna(r.get('tau')) else "N/A"
    cell_type = str(r.get('top_type', 'Unknown'))
    k1 = f"{r['gain_virchow2_k1']:+.3f}" if pd.notna(r.get('gain_virchow2_k1')) else "N/A"
    k2 = f"{r['gain_virchow2_k2']:+.3f}" if pd.notna(r.get('gain_virchow2_k2')) else "N/A"
    
    card = f"""
### Gene: **{r['gene']}** *(Target: `{target}`)*

| Model / Baseline | Pearson $r$ | Details |
| :--- | :--- | :--- |
| **Virchow2** (Image FM) | **`{v2}`** | Paige AI 632M parameter pathology ViT |
| **UNI2-h** (Image FM) | **`{uni}`** | Mahmood Lab ViT-Giant encoder |
| **CONCH v1.5** (V-L FM) | **`{conch}`** | Vision-Language pathology model |
| **Cell Composition** | **`{comp}`** | 8-cell-type reference proportion oracle |
| **Cell Count Baseline** | **`{ncells}`** | Univariate total nuclear count |
| **Image − Composition** | **`{diff}`** | Foundation model margin over composition |

#### Biological & Context Properties:
- **Top Cell Type**: `{cell_type}`
- **Spatial Autocorrelation (Moran's $I$)**: `{moran}`
- **Mean Expression ($\\\\log(1+p)$)**: `{mean_exp}`
- **Cell Specificity ($\\tau$)**: `{tau}`
- **Context Gain ($k=1$ ring)**: `{k1}`
- **Context Gain ($k=2$ rings)**: `{k2}`
"""
    return card

def filter_gene_table(target: str, preset: str, cell_type: str, query: str) -> pd.DataFrame:
    """Filter and sort the atlas table."""
    df = ATLAS_DATA[target].copy()
    
    # 1. Filter by cell type
    if cell_type and cell_type != "All":
        df = df[df["top_type"] == cell_type]
        
    # 2. Filter by query
    if query and query.strip():
        q = query.strip().upper()
        df = df[df["gene"].str.contains(q, case=False, na=False)]
        
    # 3. Apply preset
    if preset == "Image beats composition":
        df = df[df["r_image_minus_composition"] > 0]
        df = df.sort_values(by="r_image_minus_composition", ascending=False)
    elif preset == "Top predictable":
        df = df.sort_values(by="r_virchow2", ascending=False)
    elif preset == "Gains from spatial context":
        df = df[df["gain_virchow2_k1"] > 0]
        df = df.sort_values(by="gain_virchow2_k1", ascending=False)
    else:
        df = df.sort_values(by="r_virchow2", ascending=False)
        
    # Select key display columns
    display_cols = [
        "gene", "r_virchow2", "r_uni2_h", "r_conch_v15", 
        "r_composition", "r_ncells", "r_image_minus_composition",
        "morans_i", "tau", "top_type", "gain_virchow2_k1"
    ]
    cols = [c for c in display_cols if c in df.columns]
    res = df[cols].copy()
    
    # Round numerical columns
    num_cols = res.select_dtypes(include=[np.number]).columns
    res[num_cols] = res[num_cols].round(3)
    return res

def plot_scatter(target: str) -> plt.Figure:
    """Generate Pearson r vs Moran's I scatter plot colored by top cell type."""
    df = ATLAS_DATA[target].dropna(subset=["morans_i", "r_virchow2", "top_type"])
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    types = sorted(df["top_type"].unique())
    cmap = plt.get_cmap("tab10")
    
    for i, ct in enumerate(types):
        sub = df[df["top_type"] == ct]
        ax.scatter(
            sub["morans_i"], 
            sub["r_virchow2"], 
            s=14, 
            alpha=0.65, 
            label=ct, 
            color=cmap(i % 10),
            edgecolors="none"
        )
        
    ax.set_xlabel("Moran's I (Spatial Autocorrelation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson $r$ (Virchow2)", fontsize=11, fontweight="bold")
    ax.set_title(f"Predictability vs Spatial Autocorrelation ({target})", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Annotate Spearman rho
    rho = df["r_virchow2"].corr(df["morans_i"], method="spearman")
    ax.text(
        0.03, 0.93, 
        f"Spearman $\\rho = {rho:.3f}$\n$N = {len(df):,}$ genes", 
        transform=ax.transAxes,
        fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray")
    )
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9, title="Top Cell Type")
    plt.tight_layout()
    return fig

def get_figures(target: str):
    """Retrieve paper figures for the chosen target."""
    folder = FIG_DIR / target
    names = [
        "fig_encoder_agreement.png",
        "fig_image_vs_composition.png",
        "fig_r_distribution.png",
        "fig_r_by_celltype.png",
        "fig_r_vs_morans.png",
        "fig_r_vs_mean.png"
    ]
    paths = []
    for name in names:
        p = folder / name
        paths.append(str(p) if p.exists() else None)
    return tuple(paths)

# ---------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------
custom_css = """
.caveat-box {
    border-left: 5px solid #d97706;
    background-color: #fef3c7;
    color: #92400e;
    padding: 14px 18px;
    margin-bottom: 20px;
    border-radius: 4px;
}
"""

with gr.Blocks(title="HEST Xenium 5k Atlas Explorer") as demo:
    gr.Markdown("""
# 🔬 HEST Xenium 5k Gene Expression Predictability Atlas Explorer

> ⚠️ **Key Finding & Normalisation Caveat:**
> While histopathology foundation models achieve seemingly high correlation ($r \approx 0.51$) under raw `log1p` targets, this predictability is almost entirely driven by spot cellularity (cell density). After standard library-size normalisation (`lognorm`), foundation models ($r \approx 0.167$) **barely outperform a scalar cell-count baseline ($r \approx 0.158$)** and an 8-cell-type composition oracle ($r \approx 0.137$).

**Code Repository:** [GitHub Link (TODO)](https://github.com/TODO) | **Dataset Card:** [raju-ai/hest-xenium5k-predictability-atlas](https://huggingface.co/datasets/raju-ai/hest-xenium5k-predictability-atlas) | **Reference:** Jaume et al., NeurIPS 2024 (HEST-1k)
""")

    # Target toggle
    with gr.Row():
        target_radio = gr.Radio(
            choices=["lognorm", "log1p"],
            value="lognorm",
            label="Expression Target",
            info="lognorm: library-size normalised (controls for cellularity) | log1p: raw count transform (confounded by cellularity)"
        )

    with gr.Tabs():
        # Tab 1: Gene Search
        with gr.TabItem("🔍 Gene Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    gene_input = gr.Textbox(
                        label="Search Gene Symbol",
                        placeholder="e.g. F11R, EPCAM, CD3D, ERBB2, MKI67",
                        value="F11R"
                    )
                    search_btn = gr.Button("Search", variant="primary")
                    gr.Markdown("Click a quick sample:")
                    with gr.Row():
                        btn_f11r = gr.Button("F11R", size="sm")
                        btn_epcam = gr.Button("EPCAM", size="sm")
                        btn_cd3d = gr.Button("CD3D", size="sm")
                        btn_erbb2 = gr.Button("ERBB2", size="sm")
                with gr.Column(scale=2):
                    gene_output = gr.Markdown()

        # Tab 2: Gene Predictability Table
        with gr.TabItem("📊 Filterable Atlas Table"):
            with gr.Row():
                preset_radio = gr.Radio(
                    choices=["All Genes", "Image beats composition", "Top predictable", "Gains from spatial context"],
                    value="All Genes",
                    label="Preset Filter"
                )
                cell_type_dropdown = gr.Dropdown(
                    choices=["All"] + sorted(ATLAS_DATA["lognorm"]["top_type"].dropna().unique().tolist()),
                    value="All",
                    label="Filter by Top Cell Type"
                )
                query_text = gr.Textbox(
                    label="Search Filter (Symbol substring)",
                    placeholder="Filter by name..."
                )
            table_output = gr.DataFrame(
                label="Per-Gene Predictability Atlas (5,001 Genes)",
                wrap=True
            )

        # Tab 3: Predictability vs Moran's I
        with gr.TabItem("📈 Scatter: r vs Moran's I"):
            gr.Markdown("Global spatial autocorrelation (Moran's $I$) is a primary predictor of gene expression predictability from morphology.")
            scatter_plot = gr.Plot()

        # Tab 4: Phase 3 Spatial Context
        with gr.TabItem("🌐 Phase 3: Spatial Context"):
            gr.Markdown(r"""
### Spatial Context Ablation
Evaluating whether pooling spatial neighbor rings around the $112\,\mu\text{m}$ patch improves prediction accuracy.
- **Single Patch**: $112\,\mu\text{m}$ central patch embedding.
- **+k1**: Central patch + 6 immediate hexagonal neighbors (radius $\approx 112\,\mu\text{m}$).
- **+k2**: Central patch + 2 neighbor rings (18 neighbors total).
- **Attention (k1)**: Learned attention pooling over neighbor patches.
""")
            gr.DataFrame(value=SUMMARY_PHASE3, label="Context Ablation Summary")
            gr.Markdown("""
**Takeaway:** Under `log1p`, spatial context adds minimal gain ($0.510 \to 0.514$) because total cell density is largely captured locally. Under `lognorm`, broader context yields modest improvements ($0.167 \to 0.178 \to 0.184$), helping resolve cell states through wider tissue architecture.
""")

        # Tab 5: Phase 4 Generalization
        with gr.TabItem("🚀 Phase 4: Generalization"):
            gr.Markdown("""
### Generalization & Cross-Platform Transfer
Evaluating models trained on Visium 10x spatial transcriptomics transferred to single-cell resolution Xenium 5k on 4,634 shared genes under `lognorm`.
""")
            gr.DataFrame(value=GEN_SUMMARY, label="Cross-Platform Transfer Summary")
            gr.Markdown("""
**Takeaway:** 
- **Visium $\to$ Xenium**: Modest cross-platform drop of $-0.021$ (Virchow2) and $-0.026$ (UNI2-h).
- **Within-Organ Xenium**: Models trained across patient donors within the same organ improve performance to $r = 0.205$ (Virchow2) and $r = 0.173$ (UNI2-h).
""")

        # Tab 6: Phase 5 Robustness
        with gr.TabItem("🛡️ Phase 5: Robustness"):
            gr.Markdown("""
### Robustness Checks & Ablations
""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Pipeline Determinism Across Random Seeds")
                    gr.DataFrame(value=ROBUST_SEEDS, label="Seed Stability (s.d. across seeds)")
                    gr.Markdown("*Note: Ridge regression with exact PCA is fully deterministic (s.d. $< 10^{-12}$).*")
                with gr.Column():
                    gr.Markdown("#### Prediction Head: Ridge vs MLP")
                    gr.DataFrame(value=ROBUST_HEADS, label="Ridge vs MLP Comparison")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Window Size Ablation")
                    gr.Markdown(r"Virchow2 evaluated at $56\,\mu\text{m}$, $112\,\mu\text{m}$, and $224\,\mu\text{m}$ expression window sizes.")
                    w_lognorm = pd.read_csv(DATA_DIR / "robustness_window_lognorm.csv") if (DATA_DIR / "robustness_window_lognorm.csv").exists() else pd.DataFrame()
                    w_log1p = pd.read_csv(DATA_DIR / "robustness_window_log1p.csv") if (DATA_DIR / "robustness_window_log1p.csv").exists() else pd.DataFrame()
                    gr.DataFrame(value=w_lognorm.round(3), label="Window Size (lognorm)")
                    gr.DataFrame(value=w_log1p.round(3), label="Window Size (log1p)")
                with gr.Column():
                    gr.Markdown("#### Field of View Ablation")
                    gr.Markdown(r"UNI2-h evaluated at full $112\,\mu\text{m}$ patch vs central $56\,\mu\text{m}$ sub-patch.")
                    fov_lognorm = pd.read_csv(DATA_DIR / "robustness_fov_lognorm.csv") if (DATA_DIR / "robustness_fov_lognorm.csv").exists() else pd.DataFrame()
                    fov_log1p = pd.read_csv(DATA_DIR / "robustness_fov_log1p.csv") if (DATA_DIR / "robustness_fov_log1p.csv").exists() else pd.DataFrame()
                    gr.DataFrame(value=fov_lognorm.round(3), label="Field of View (lognorm)")
                    gr.DataFrame(value=fov_log1p.round(3), label="Field of View (log1p)")

        # Tab 7: Paper Figures
        with gr.TabItem("🖼️ Paper Figures"):
            gr.Markdown("Key figures from the paper corresponding to the active target.")
            with gr.Row():
                fig1 = gr.Image(label="Foundation Model Agreement")
                fig2 = gr.Image(label="Image vs Composition Baseline")
            with gr.Row():
                fig3 = gr.Image(label="Predictability Distribution")
                fig4 = gr.Image(label="Predictability by Cell Type")
            with gr.Row():
                fig5 = gr.Image(label="Predictability vs Moran's I")
                fig6 = gr.Image(label="Predictability vs Mean Expression")

    # ---------------------------------------------------------
    # Interactions & Events
    # ---------------------------------------------------------
    # Initial load / target toggle
    def update_all(target: str, preset: str, cell_type: str, query: str, gene: str):
        card = search_gene(gene, target)
        tbl = filter_gene_table(target, preset, cell_type, query)
        plt_fig = plot_scatter(target)
        figs = get_figures(target)
        return (card, tbl, plt_fig) + figs

    target_radio.change(
        fn=update_all,
        inputs=[target_radio, preset_radio, cell_type_dropdown, query_text, gene_input],
        outputs=[gene_output, table_output, scatter_plot, fig1, fig2, fig3, fig4, fig5, fig6]
    )

    # Search gene handlers
    search_btn.click(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)
    gene_input.submit(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)

    btn_f11r.click(lambda: "F11R", outputs=gene_input).then(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)
    btn_epcam.click(lambda: "EPCAM", outputs=gene_input).then(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)
    btn_cd3d.click(lambda: "CD3D", outputs=gene_input).then(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)
    btn_erbb2.click(lambda: "ERBB2", outputs=gene_input).then(fn=search_gene, inputs=[gene_input, target_radio], outputs=gene_output)

    # Table filter handlers
    for trigger in [preset_radio.change, cell_type_dropdown.change, query_text.change]:
        trigger(
            fn=filter_gene_table,
            inputs=[target_radio, preset_radio, cell_type_dropdown, query_text],
            outputs=table_output
        )

    # App load event
    demo.load(
        fn=update_all,
        inputs=[target_radio, preset_radio, cell_type_dropdown, query_text, gene_input],
        outputs=[gene_output, table_output, scatter_plot, fig1, fig2, fig3, fig4, fig5, fig6]
    )

if __name__ == "__main__":
    demo.launch()
