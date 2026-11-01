---
license: cc-by-nc-sa-4.0
tags:
  - spatial-transcriptomics
  - histopathology
  - gene-expression
  - pathology-foundation-models
  - xenium
  - hest-1k
pretty_name: HEST Xenium 5K Gene Predictability Atlas
size_categories:
  - 1K<n<10K
---

# HEST Xenium 5K Gene Predictability Atlas

Per-gene and per-model summary statistics from the study *Which genes can a pathology foundation model see?
Cellularity confounding and transcriptome-wide predictability from H&E* (Raju Sah, 2026).

- Code: https://github.com/raju-sah/histology-gene-expression-prediction
- Paper: preprint link to be added.

The study asks how well frozen pathology foundation models (Virchow2, UNI2-h, CONCH v1.5) predict the expression of
each gene in the Xenium Prime 5K panel from H&E image patches, and compares them with baselines that use the number
of cells in each spot and the cell-type composition of the spot.

**Contents.** Only aggregated summary statistics: per gene, per model, per target. No model weights, embeddings,
image patches, spot coordinates, barcodes or spot-level expression are included.

## Data and protocol

- **Samples:** 9 human Xenium Prime 5K samples from HEST-1k v1.3.0 (lung, ovary, cervix, skin melanoma, prostate,
  reactive lymph node). TENX190 (a Xenium v1 run) is excluded.
- **Splits:** leave-one-group-out over 8 groups. NCBI887 and NCBI888 come from the same study and share a group,
  because patient identifiers are not available.
- **Genes:** 4,624 genes shared by all 9 samples (main atlas and spatial-context files).
- **Pseudo-spots:** transcripts (QV >= 20) of all cells whose centroid lies within each 112 um H&E patch are summed.
- **Targets:** `log1p` = log(1 + count). `lognorm` = counts divided by the spot's total count (library-size
  normalisation), then log-transformed; this removes most of the effect of how many cells a spot contains.
- **Model:** frozen encoder embedding, standardised, PCA to 256 dimensions (fitted on training data only), then one
  ridge regression per gene with the penalty chosen on held-out training groups.
- **Metric:** Pearson r between predicted and measured expression per gene on the held-out group, averaged over folds.
- **Baselines:**
  - `ncells`: number of cells in the spot from the **Xenium cell segmentation** (molecular data, not the image). It is a
    reference point, not an image-only competitor.
  - `composition`: fractions of coarse marker-based cell types in the spot (many cells stay unassigned).
  - `composition_ncells`: both together.
  - A permutation control (expression shuffled within each sample) gives r close to 0.

## Files

| File | Content |
|---|---|
| `atlas_log1p.csv`, `atlas_lognorm.csv` | One row per gene: `r_<model>` for each encoder and baseline, `n_folds_evaluable`, `agreement_mean` (correlation of predictions between encoders), `mean_log1p`, `var_log1p`, `mean_counts`, `frac_zero`, `morans_i` (spatial autocorrelation), `tau` (cell-type specificity), `top_type`, `r_best_encoder`, `r_encoder_sd`, and `r_image_minus_<baseline>` differences |
| `summary_phase2.csv` | One row per model and target: mean r with 95% bootstrap interval, per-sample mean, median gene r, fraction of genes above 0.3 / 0.5, per-fold values, permutation-control r |
| `context_gain_log1p.csv`, `context_gain_lognorm.csv` | Virchow2 with neighbouring patches: single patch, +1 ring, +2 rings (mean-pooled) and attention pooling (gene subset); gains and the number of folds with a positive gain |
| `summary_phase3.csv` | Mean r per context variant and target |
| `generalization_summary.csv` | Visium lung -> Xenium transfer and same-organ training vs. leave-one-group-out (normalised targets) |
| `robustness_*.csv` | Seeds (the pipeline is deterministic, so seeds change nothing), ridge vs. MLP head, expression window 56/112/224 um, image field of view 112 vs. 56 um |

**Important:** `generalization_summary.csv` and all `robustness_*.csv` files come from the **earlier 6-sample run**
(5,001 genes, 5 groups), before 3 samples were added. They were not repeated on the 9-sample set.

## Main finding

On raw counts, the per-spot cell count alone reaches a large part of the encoders' performance. After library-size
normalisation, all models drop sharply; the best encoder (Virchow2) is statistically tied with the cell-count baseline
and above composition, while UNI2-h and CONCH v1.5 fall below the cell count. Some genes (for example fibroblast-
associated CTHRC1, THY1, ADAM12) are predicted well above both baselines. Exact numbers are in `summary_phase2.csv`
and in the paper.

## Limitations

- 9 samples in 8 groups: confidence intervals are wide and per-organ conclusions are weak.
- Coarse marker-based cell typing leaves most cells unassigned, which weakens the composition baseline (the paper adds
  an unsupervised clustering baseline that does not change the conclusion).
- Linear probes on frozen embeddings only; fine-tuned models were not tested. H-optimus models were not available.
- Cross-platform transfer was tested on lung only.

## Licence and attribution

Derived from HEST-1k (CC BY-NC-SA 4.0); this dataset is released under the same licence (non-commercial,
share-alike). Encoder weights are not included and must be obtained from their owners under their own terms.

Please cite HEST-1k:

```bibtex
@inproceedings{jaume2024hest,
  author    = {Guillaume Jaume and Paul Doucet and Andrew H. Song and Ming Y. Lu and Cristina Almagro-Perez and Sophia J. Wagner and Anurag J. Vaidya and Richard J. Chen and Drew F. K. Williamson and Ahrong Kim and Faisal Mahmood},
  title     = {HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
  note      = {arXiv:2406.16192}
}
```

and this work:

```bibtex
@misc{sah2026cellularity,
  author = {Raju Sah},
  title  = {Which genes can a pathology foundation model see? Cellularity confounding and transcriptome-wide predictability from {H\&E}},
  year   = {2026},
  note   = {Preprint; code at https://github.com/raju-sah/histology-gene-expression-prediction}
}
```
