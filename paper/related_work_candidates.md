# Related Work Candidates & Novelty Assessment

This document catalogs candidate literature for the spatial transcriptomics from histology preprint, verified DOIs/URLs, and a formal assessment of novelty regarding the cellularity confound.

---

## 1. Deep Learning for Histology-to-Expression Prediction

### ST-Net
- **Title**: Integrating spatial gene expression and breast tumour morphology via deep learning
- **Authors**: Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, José Fernández Navarro, Henrik Pettersson, Joakim Lundeberg, Zoubin Ghahramani
- **Year**: 2020
- **Venue**: *Nature Biomedical Engineering*, 4(8), 827–834
- **DOI**: [10.1038/s41551-020-0578-x](https://doi.org/10.1038/s41551-020-0578-x)
- **Relevance**: First deep learning framework using DenseNet to predict spatial gene expression from H&E image patches on 10x Visium/spatial transcriptomics. Evaluated on raw log-transformed counts without controlling for local cellularity.

### HisToGene
- **Title**: Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors
- **Authors**: Minxing Pang, Kenong Su, Mingyao Li
- **Year**: 2021
- **Venue**: *bioRxiv* / *Genome Biology*, 2021.11.28.470212
- **DOI**: [10.1101/2021.11.28.470212](https://doi.org/10.1101/2021.11.28.470212)
- **Relevance**: Vision Transformer (ViT) architecture incorporating self-attention across neighbouring spots to capture spatial context; demonstrated that spatial neighbouring patches inform expression, but focused on un-normalised targets.

### BLEEP
- **Title**: Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning
- **Authors**: Ronald Xie, Kexin Ding, Gary D. Bader, Bo Wang
- **Year**: 2023
- **Venue**: *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Vol. 36
- **arXiv**: [arXiv:2306.01257](https://arxiv.org/abs/2306.01257)
- **Relevance**: Formulates spatial expression prediction as a bi-modal contrastive learning task (CLIP-style), embedding histology patches and spot expression into a shared latent space for non-parametric expression imputation.

### DeepPT / ENLIGHT-DeepPT
- **Title**: A deep-learning framework to predict cancer treatment response from histopathology images through imputed transcriptomics
- **Authors**: Danh-Tai Hoang, Gal Dinstag, et al., Eytan Ruppin
- **Year**: 2024
- **Venue**: *Nature Cancer* / *Nature Medicine*, 2024
- **DOI**: [10.1038/s43018-024-00793-2](https://doi.org/10.1038/s43018-024-00793-2)
- **Relevance**: Translates patch-level expression predictions from standard whole-slide images to downstream clinical therapeutic response (ENLIGHT), emphasizing the translational promise of histology-to-transcriptomics imputation.

---

## 2. Spatial Transcriptomics Benchmarking

### HEST-1k
- **Title**: HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis
- **Authors**: Guillaume Jaume, Paul Doucet, Andrew H. Song, Ming Y. Lu, Cristina Almagro-Perez, Sophia J. Wagner, Anurag J. Vaidya, Richard J. Chen, Drew F. K. Williamson, Ahrong Kim, Faisal Mahmood
- **Year**: 2024
- **Venue**: *Advances in Neural Information Processing Systems (NeurIPS 2024)* Datasets & Benchmarks Track
- **URL**: [https://github.com/MahmoodLab/HEST](https://github.com/MahmoodLab/HEST)
- **Relevance**: Largest standardized benchmark combining 1,200+ histology-ST pairs across Visium and Xenium; introduces the linear probe protocol on frozen foundation models that serves as our baseline experimental setup.

---

## 3. Pathology Vision Foundation Models

### UNI
- **Title**: Towards a general-purpose foundation model for computational pathology
- **Authors**: Richard J. Chen, Tong Ding, Ming Y. Lu, Drew F. K. Williamson, Guillaume Jaume, Andrew H. Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al., Faisal Mahmood
- **Year**: 2024
- **Venue**: *Nature Medicine*, 30, 850–862
- **DOI**: [10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3)
- **Relevance**: Large-scale self-supervised Vision Transformer (ViT-L) trained on 100M+ tissue patches across 20+ disease types; benchmarked in our atlas.

### UNI2-h
- **Title**: A General-Purpose Pathology Foundation Model with Giant Vision Transformers
- **Authors**: Richard J. Chen, et al., Faisal Mahmood
- **Year**: 2024
- **Venue**: *arXiv:2407.03613*
- **URL**: [https://arxiv.org/abs/2407.03613](https://arxiv.org/abs/2407.03613)
- **Relevance**: Giant SwiGLU-based foundation model with register tokens; represents the state-of-the-art vision encoder evaluated in Phase 2 across Xenium 5k.

### Virchow / Virchow2
- **Title**: A foundation model for clinical-grade computational pathology and rare cancers detection
- **Authors**: Eugene Vorontsov, Alican Bozkurt, Adam Casson, et al., Paige AI
- **Year**: 2024
- **Venue**: *Nature Medicine*, 30, 2924–2935
- **DOI**: [10.1038/s41591-024-03141-0](https://doi.org/10.1038/s41591-024-03141-0)
- **Relevance**: Trained on 1.5M whole-slide images (3B+ patches) from Paige AI; top-performing visual feature extractor in our atlas.

### CONCH / TITAN
- **Title**: A visual-language foundation model for computational pathology / Multimodal Whole Slide Foundation Model for Pathology
- **Authors**: Ming Y. Lu, Bowen Chen, et al. / Tong Ding, Sophia J. Wagner, Andrew H. Song, et al., Faisal Mahmood
- **Year**: 2024 / 2025
- **Venue**: *Nature Medicine*, 30, 863–874 (2024) / *Nature Medicine*, 31, 3749–3761 (2025)
- **DOI**: [10.1038/s41591-024-02856-4](https://doi.org/10.1038/s41591-024-02856-4) / [10.1038/s41591-025-03982-3](https://doi.org/10.1038/s41591-025-03982-3)
- **Relevance**: Multimodal vision-language foundation models incorporating biomedical text and morphological features.

---

## 4. Normalisation, Cellularity, and Spatial Confounders

### SCTransform
- **Title**: Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression
- **Authors**: Christoph Hafemeister, Rahul Satija
- **Year**: 2019
- **Venue**: *Genome Biology*, 20, 296
- **DOI**: [10.1186/s13059-019-1874-1](https://doi.org/10.1186/s13059-019-1874-1)
- **Relevance**: Foundational methodology showing that sequencing depth / total counts acts as a dominating technical confound in single-cell transcriptomics, requiring rigorous decoupling of library size from biological variance.

### Cell2location
- **Title**: Comprehensive cell-type mapping of spatial transcriptomics data with cell2location
- **Authors**: Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, et al., Oliver Stegle
- **Year**: 2022
- **Venue**: *Nature Biotechnology*, 40, 661–671
- **DOI**: [10.1038/s41587-021-01139-4](https://doi.org/10.1038/s41587-021-01139-4)
- **Relevance**: Formal Bayesian model demonstrating that multi-cell spatial transcriptomics spots are superpositions of cell-type abundances and cell density.

---

## 5. Novelty Assessment Note: The Cellularity Confound in H&E Expression Prediction

### Has anyone pointed out that H&E-to-expression predictability on raw counts is mostly cellularity?

**Short Answer**: **No.** While cellularity is known in spatial biology to affect total read counts, the deep learning and computational pathology literature has **consistently overlooked cellularity as the primary driver of reported high predictability scores**.

### Detailed Literature Assessment:
1. **The Status Quo in the Field**:
   - The pioneering and widely-cited models (ST-Net, HisToGene, BLEEP, DeepPT, and the HEST benchmark) evaluate primarily on raw or un-normalized log-transformed counts ($\log(1 + \text{count})$).
   - They report mean Pearson correlations of $r \approx 0.35$--$0.60$ across predictable genes, attributing this performance to complex neural network representations capturing microscopic tissue architecture, cellular differentiation, and tumor heterogeneity.
   - Crucially, **none of these papers included a simple non-imaging scalar cell count baseline ($n\_cells$) or a cellular composition baseline** to benchmark how much of that predictive power originates merely from the density of nuclei under the patch.

2. **What This Work Demonstrates (The Core Novel Contribution)**:
   - **Trivial Baseline Match**: A scalar cell count ($n\_cells$) alone yields $r = 0.461$ on raw counts, virtually matching multi-billion-parameter foundation models like Virchow2 ($r = 0.456$) and UNI2-h ($r = 0.448$).
   - **Variance Explained**: OLS regression on gene predictability reveals that cellularity alone explains **~94% of the cross-gene variance** in model performance ($R^2 = 0.939$).
   - **The Collapse on Normalized Data**: Once standard library-size normalization (`lognorm`) is applied, apparent predictability collapses by 60–70% across all foundation models ($r \approx 0.15$--$0.20$), where image foundation models barely outperform scalar cell count or coarse composition.
   - **Robustness to Unsupervised Clustering (0% Unassigned Cells)**: Even when 100% of cells are assigned to data-driven unsupervised single-cell clusters (reducing unassigned cells from 72% down to 0%), cluster composition does not exceed cellularity on normalized targets ($r = 0.207$ vs $r = 0.223$).

3. **Conclusion on Novelty**:
   - This paper represents the **first systematic empirical demonstration and dissection** of the cellularity confound in spatial transcriptomics prediction from histology images.
   - It re-frames the entire task: predicting expression from H&E is largely a surrogate for estimating local cell density and broad lineage composition, rather than inferring nuanced cell-intrinsic transcriptional regulation.
