---
license: cc-by-nc-sa-4.0
tags:
  - spatial-transcriptomics
  - histopathology
  - gene-expression
  - pathology-foundation-models
  - xenium-5k
  - hest-1k
pretty_name: HEST Xenium 5k Gene Expression Predictability Atlas
size_categories:
  - 1K<n<10K
---

# HEST Xenium 5k Gene Expression Predictability Atlas

This dataset contains per-gene and per-model summary statistics benchmarking the predictability of spatial gene expression from H&E histopathology image patches across 5,001 genes in 6 Xenium 5k samples from the HEST-1k benchmark.

**Important Note on Data Privacy & Scope:**
No proprietary model weights, embeddings, raw image patches, spot coordinates, barcodes, or single-spot expression matrices are contained in this repository. Only aggregated per-gene summary metrics and evaluation summaries across cross-validation folds are shared.

---

## Dataset Structure & Column Dictionary

### 1. `atlas_log1p.csv` and `atlas_lognorm.csv`
Per-gene predictability metrics across 5,001 genes under raw count target (`log1p`) and library-size normalised target (`lognorm`).
- `gene`: Gene symbol (HGNC name, 5,001 targeted genes).
- `r_composition`: Pearson correlation $r$ achieved by the cell-type composition oracle baseline.
- `r_conch_v15`: Pearson correlation $r$ achieved by CONCH v1.5 image encoder.
- `r_ncells`: Pearson correlation $r$ achieved by the total cell count baseline.
- `r_uni2_h`: Pearson correlation $r$ achieved by UNI2-h image encoder.
- `r_virchow2`: Pearson correlation $r$ achieved by Virchow2 image encoder.
- `n_folds_evaluable`: Number of cross-validation folds in which the gene had non-zero variance and was evaluable (max 5).
- `agreement_mean`: Pairwise Pearson correlation of predictions between image encoders (Virchow2, UNI2-h, CONCH v1.5).
- `mean_log1p`: Mean log(1+count) of the gene across all spots.
- `var_log1p`: Variance of log(1+count) across spots.
- `mean_counts`: Mean raw transcript count per spot.
- `frac_zero`: Fraction of spots with zero observed counts.
- `morans_i`: Global spatial autocorrelation (Moran's $I$) computed on target expression.
- `tau`: Tissue specificity index ($\tau$) indicating cell-type specificity.
- `top_type`: Highest-expressing cell type for this gene according to single-cell reference annotations.
- `r_best_encoder`: Maximum $r$ across Virchow2, UNI2-h, and CONCH v1.5.
- `r_encoder_sd`: Standard deviation of $r$ across the three foundation models.
- `r_image_minus_composition`: Difference between best image encoder $r$ and the cell composition baseline ($r_{\text{best}} - r_{\text{composition}}$).

### 2. `summary_phase2.csv`
Aggregate benchmark performance comparing models and baselines under both `log1p` and `lognorm` targets.
- `model`: Evaluated model or baseline (`virchow2`, `uni2_h`, `conch_v15`, `composition`, `ncells`).
- `pearson_mean`: Mean Pearson correlation $r$ across all 5,001 genes and evaluable folds.
- `ci_lo`, `ci_hi`: 95% bootstrap confidence intervals for mean Pearson $r$.
- `per_sample_mean`: Macro-average Pearson $r$ across samples.
- `median_gene_r`: Median per-gene Pearson $r$.
- `frac_gt_0_3`: Fraction of genes with mean $r > 0.3$.
- `frac_gt_0_5`: Fraction of genes with mean $r > 0.5$.
- `n_genes_evaluable`: Number of evaluated genes (5,001).
- `folds`: Semicolon-delimited list of mean $r$ values for each cross-validation fold.
- `y_kind`: Target transformation (`log1p` or `lognorm`).
- `null_permuted_r`: Mean Pearson $r$ obtained under target permutation (null baseline).

### 3. `context_gain_log1p.csv` and `context_gain_lognorm.csv`
Per-gene spatial context ablation statistics comparing single-patch features against spatial neighborhood aggregations.
- `gene`: Gene symbol.
- `r_virchow2`: Pearson $r$ with single center patch ($112\,\mu\text{m}$).
- `r_virchow2_k1`: Pearson $r$ with center patch + immediate ring of 6 spatial neighbors (mean-pooled).
- `r_virchow2_k2`: Pearson $r$ with center patch + 2 rings of spatial neighbors (18 neighbors, mean-pooled).
- `r_virchow2_attn_k1`: Pearson $r$ with attention-weighted neighborhood pooling (evaluated on 500 representative genes).
- `mean_log1p`, `frac_zero`, `morans_i`, `tau`, `top_type`: Gene-level biological properties.
- `gain_virchow2_k1`: Gain in $r$ from 1st ring context ($r_{k1} - r_{\text{single}}$).
- `npos_virchow2_k1`: Number of folds where $k=1$ context improved prediction over single patch.
- `gain_virchow2_k2`: Gain in $r$ from 2 rings context ($r_{k2} - r_{\text{single}}$).
- `npos_virchow2_k2`: Number of folds where $k=2$ context improved prediction.
- `gain_virchow2_attn_k1`, `npos_virchow2_attn_k1`: Gain and fold count for attention pooling.

### 4. `summary_phase3.csv`
Macro-summary of context ablation across all evaluated models for both targets.
- `encoder`: Context variant (`virchow2`, `virchow2_k1`, `virchow2_k2`, `virchow2_attn_k1`).
- `pearson_mean`: Mean Pearson $r$ across genes.
- `fold_sd`: Standard deviation of Pearson $r$ across cross-validation folds.
- `target`: Target transformation (`log1p` or `lognorm`).

### 5. `generalization_summary.csv`
Cross-platform and cross-sample transfer evaluation (Phase 4).
- `setting`: Evaluation setting (`xenium_logo` within-platform LOGO, `xenium_same_organ` within-organ transfer, `visium_same_organ` cross-platform transfer from Visium to Xenium on 4,634 shared genes).
- `encoder`: Evaluated encoder (`uni2_h`, `virchow2`).
- `mean_r`: Mean Pearson $r$ on the test split.
- `mean_drop_vs_logo`: Change in mean $r$ relative to the standard within-platform LOGO benchmark.
- `n_test_samples`: Number of test samples evaluated.

### 6. Robustness Files
- `robustness_seeds.csv`: Determinism check across random seeds for frozen PCA + ridge regression.
- `robustness_heads.csv`: Comparison of linear ridge regression versus multi-layer perceptron (MLP) prediction heads on PCA-256 embeddings.
- `robustness_window_log1p.csv` and `robustness_window_lognorm.csv`: Expression window sizing ablation ($56\,\mu\text{m}$, $112\,\mu\text{m}$, $224\,\mu\text{m}$).
- `robustness_fov_log1p.csv` and `robustness_fov_lognorm.csv`: Optical field of view ablation comparing $112\,\mu\text{m}$ patch against central $56\,\mu\text{m}$ sub-patch.
- `robustness_fov_genes_log1p.csv` and `robustness_fov_genes_lognorm.csv`: Per-gene correlation for $112\,\mu\text{m}$ vs $56\,\mu\text{m}$ field of view.

---

## Target Definitions: Raw (`log1p`) vs Normalised (`lognorm`)

A key finding of this research is that evaluation target definition drastically alters scientific conclusions:

1. **Raw `log1p` ($\log(1 + y)$):**
   - Directly transforms raw transcript counts.
   - Because raw spot counts scale with total cellularity (the number of cells present under the patch), models predicting `log1p` primarily exploit tissue density.
   - Foundation models achieve seemingly strong performance (Virchow2 $r = 0.510$, UNI2-h $r = 0.474$, CONCH v1.5 $r = 0.463$).
   - However, a scalar cell-count baseline alone achieves $r = 0.425$, and a cell composition baseline achieves $r = 0.497$.

2. **Library-Size Normalised `lognorm` ($\log(1 + 10^4 \cdot y / \sum y)$):**
   - Standard spatial transcriptomics normalisation adjusting for spot cellularity and sequencing depth.
   - Foundation model performance drops dramatically: Virchow2 $r = 0.167$ [0.138, 0.194], UNI2-h $r = 0.148$, CONCH v1.5 $r = 0.123$.
   - The cell-count baseline achieves $r = 0.158$, and cell composition achieves $r = 0.137$.
   - **Conclusion:** After controlling for spot cellularity, histopathology image foundation models barely outperform a simple cell count.

---

## Experimental Protocol

- **Dataset:** 6 Xenium 5k FFPE tissue sections from HEST-1k (breast cancer, colorectal cancer, non-small cell lung cancer).
- **Split:** Leave-one-group-out (LOGO) cross-validation across 5 biologically independent donor groups.
- **Image Features:** Pathology foundation models (Virchow2, UNI2-h, CONCH v1.5) extract embeddings from $112\,\mu\text{m} \times 112\,\mu\text{m}$ patches ($224 \times 224$ pixels at $0.5\,\mu\text{m}/\text{px}$ resolution). Embeddings are reduced to 256 dimensions using exact PCA fitted on the training split.
- **Predictor Head:** Per-gene $L_2$-regularised Ridge regression with regularization parameter $\alpha \in [10^{-3}, 10^5]$ tuned via inner cross-validation.
- **Baselines:**
  - *Cell Count:* Univariate ridge regression from the scalar count of segmented cell nuclei.
  - *Cell Composition:* Ridge regression from 8-dimensional reference cell-type proportion vectors.
  - *Permuted Null:* Empirical null distribution generated by permuting target expression across spots.

---

## Summary of Findings

- **Predictability & Confounding:** Under `log1p`, linear regression explains $R^2 = 0.939$ of variation in per-gene predictability, dominated by Moran's $I$ ($\rho = 0.927$) and cell composition correlation ($\rho = 0.983$). Under `lognorm`, the model explains $R^2 = 0.649$, where spatial autocorrelation ($\rho = 0.506$) and cell composition ($\rho = 0.688$) remain significant but cell density confounds are mitigated.
- **Spatial Context:** Mean-pooling features from spatial neighbor rings yields slight gains under `lognorm` ($r=0.167 \to 0.178$ for $k=1$, and $0.184$ for $k=2$), showing that broader morphological context aids expression prediction when cellularity is controlled.
- **Generalization:** Cross-platform transfer (Visium $\to$ Xenium on 4,634 shared genes under `lognorm`) incurs modest degradation (Virchow2 drop $-0.021$, UNI2-h drop $-0.026$). Within-organ transfer shows modest improvement ($+0.034$ for Virchow2).
- **Ablations:** Expression window of $112\,\mu\text{m}$ outperforms $56\,\mu\text{m}$ ($r = 0.143$) and $224\,\mu\text{m}$ ($r = 0.151$). Restricting optical FOV to $56\,\mu\text{m}$ reduces UNI2-h performance from $0.148$ to $0.117$.

---

## Limitations

1. **Cell Segmentation & Assignment:** Approximately 60–70% of segmented transcripts in the Xenium 5k dataset cannot be unambiguously assigned to segmented cell bodies, introducing baseline noise into the cell-count and composition baselines.
2. **Cohort Diversity:** The benchmark comprises 6 samples across 5 patient groups; broader validation across larger pan-cancer cohorts is warranted.
3. **Model Availability:** H-optimus-0 and H-optimus-1 foundation models were excluded from this benchmark due to licensing and gating restrictions.
4. **Cross-Platform Scope:** Phase 4 cross-platform evaluation (Visium $\to$ Xenium) was evaluated on lung tissue samples where matched high-resolution ST and Visium data were available.

---

## Licensing & Data Attribution

- This predictability atlas derives from the [HEST-1k dataset](https://github.com/mahmoodlab/HEST) and is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.
- Foundation model weights (Virchow2, UNI2-h, CONCH v1.5) must be obtained directly from their respective creators in accordance with their licenses.

### Citations

If you use this atlas or its derived statistics, please cite HEST-1k:

```bibtex
@inproceedings{jaume2024hest,
    author = {Guillaume Jaume and Paul Doucet and Andrew H. Song and Ming Y. Lu and Cristina Almagro-Perez and Sophia J. Wagner and Anurag J. Vaidya and Richard J. Chen and Drew F. K. Williamson and Ahrong Kim and Faisal Mahmood},
    title = {HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024},
    month = dec,
}
```

Please also cite this study:

```bibtex
@article{todo2026spatialhistologyatlas,
    author = {TODO},
    title = {Benchmarking Pathology Foundation Models for Spatial Gene Expression Prediction: The Cellularity Confound in H&E-ST},
    journal = {TODO},
    year = {2026},
    note = {Preprint / In review}
}
```
