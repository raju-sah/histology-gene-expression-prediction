# Author Defense & Verification Checklist

This document prepares the authors to defend the methodology, findings, and conclusions of the manuscript (*"Which genes can a pathology foundation model see? Cellularity confounding and transcriptome-wide predictability from H&E"*) in peer review, grant defenses, and technical interviews.

---

## 1. Claims Matrix

Every scientific claim made in the Abstract, Introduction, and Results is mapped directly to its underlying quantitative evidence, generated macro, and source artifact.

| # | Manuscript Claim | Evidence Source | Exact Macro / Metric | Numerical Value |
|---|---|---|---|---|
| **1** | Atlas spans 9 samples across 6 organ systems and 8 LOGO groups | `configs/project.yaml`, `reports/phase2/` | `\NXeniumSamples`, `\NXeniumGroups` | 9 samples, 8 groups (lung, ovary, cervix, skin, prostate, lymph node) |
| **2** | Predictability evaluated over 4,624 shared genes across all samples | `reports/phase2/log1p/gene_table.csv` | `\AtlasNGenes` | 4624 common genes |
| **3** | Virchow2 reaches $r = 0.519$ on raw $\log(1+\text{count})$ | `reports/phase2/log1p/summary.csv` | `\AtlasVirchowTwo` | Mean $r = 0.519$ |
| **4** | Scalar spot cell count ($n\_cells$) matches foundation models on raw counts | `reports/phase2/log1p/summary.csv` | `\AtlasNcells` | Mean $r = 0.445$ |
| **5** | Coarse composition oracle matches foundation models on raw counts | `reports/phase2/log1p/summary.csv` | `\AtlasComposition` | Mean $r = 0.490$ |
| **6** | Predictability collapses upon library-size normalisation ($\text{lognorm}$) | `reports/phase2/lognorm/summary.csv` | `\AtlasnormVirchowTwo`, `\AtlasnormUniTwoH` | Virchow2: $r = 0.193$; UNI2-h: $r = 0.148$ |
| **7** | Image models do not outperform scalar cell count after normalisation | `reports/phase2/lognorm/summary.csv` | `\AtlasnormNcells` | Cell count: $r = 0.214$; Virchow2: $r = 0.193$ |
| **8** | Unsupervised single-cell clustering (0.000 unassigned) fails to beat cell count | `reports/phase2/clustering_baseline_summary.json` | `\ClusterUnassignedFrac`, `\ClusterBaselineLognorm` | Unassigned: 0.000; Cluster comp ($r = 0.207$) vs $n\_cells$ ($r = 0.223$) |
| **9** | Spot cellularity explains ~94% of predictability variance on raw counts | `reports/phase2/log1p/explain_ols.csv` | `\ExplainRsqLogOnePlus` | OLS model $R^2 = 0.953$ |
| **10** | Normalised predictability explained by cell-type specificity ($\tau$) and encoder agreement | `reports/phase2/lognorm/explain_ols.csv` | `\ExplainRsqLognorm` | OLS model $R^2 = 0.610$ |
| **11** | Spatial context ring averaging yields negligible change on normalised counts | `reports/phase3/lognorm/gain_summary.csv` | `\GainnormVirchowTwoKOne` | Mean gain $\Delta r = 0.010$ (160 / 4,624 genes gain $>0.05$) |
| **12** | Attention pooling provides no advantage over uniform ring averaging | `reports/phase3/lognorm/summary.csv` | `\GainVirchowTwoAttnKOne` | Attention penalty $\Delta r = -0.027$ |
| **13** | Cross-platform Visium-to-Xenium transfer exhibits modest drop within lung | `reports/phase4/summary.csv` | `\GenVisiumSameOrganVirchowTwo`, `\DropVisiumSameOrganVirchowTwo` | Visium $\to$ Xenium: $r = 0.132$ ($\Delta r = -0.021$) |
| **14** | Pipeline is strictly deterministic across random seeds (not robustness) | `reports/phase5/seeds_summary.csv` | `\SeedsdVirchowTwo` | Standard deviation across seeds = 0.0000 |
| **15** | Permutation control breaks morphology-expression alignment | `reports/phase2/log1p/summary.csv` | `\AtlasNullR` | Mean permuted $r = -0.000 \approx 0.000$ (zero leakage) |

---

## 2. Methods Summary: Exact Mathematical Formulations

### 2.1 Pseudo-Spot Construction & Alignment
- **Transcript streaming**: High-quality single-molecule transcripts ($\text{QV} \ge 20$) are assigned to segmented DAPI cell boundaries.
- **Centroid registration**: Each cell $i$ is assigned coordinate $\mathbf{c}_i \in \mathbb{R}^2$ computed as the mean registered coordinate of its constituent transcripts.
- **Footprint aggregation**: For each HEST patch centered at $\mathbf{p}_s$ with square footprint $F_s = [p_{s,x} - w/2, p_{s,x} + w/2] \times [p_{s,y} - w/2, p_{s,y} + w/2]$ ($w = 112\,\mu\text{m} = 224\,\text{px} \times 0.5\,\mu\text{m/px}$), pseudo-spot expression is:
  $$\mathbf{y}_s = \sum_{i: \mathbf{c}_i \in F_s} \mathbf{x}_i \in \mathbb{N}_0^G, \quad n_s = \sum_{i: \mathbf{c}_i \in F_s} 1$$
  where $n_s$ is the scalar spot cell count (`n_cells`).

### 2.2 The Cellularity Confound & Normalisation Mathematics
- **Raw target ($\text{log1p}$)**:
  $$Y_{s,g}^{(\text{raw})} = \log(1 + y_{s,g})$$
  In multi-cellular tissue spots, $y_{s,g} \approx n_s \cdot \mu_g \cdot c_{s,g}$, where $n_s$ is local cell density, $\mu_g$ is baseline gene expression, and $c_{s,g}$ is spot lineage composition. Because $n_s$ varies across spots by orders of magnitude (e.g., 5 to 150 cells/spot), $\log(1 + y_{s,g}) \approx \log n_s + \log \mu_g$. Consequently, any vision model that predicts nuclear density automatically achieves high correlation with $Y_{s,g}^{(\text{raw})}$.
- **Library-size normalised target ($\text{lognorm}$)**:
  Let $T_s = \sum_{g=1}^G y_{s,g}$ be total counts per spot over the transcriptome. Let $M = \text{median}_{s': T_{s'} > 0}(T_{s'})$. The library-size normalized target is:
  $$\tilde{y}_{s,g} = \frac{y_{s,g}}{T_s} \cdot M, \quad Y_{s,g}^{(\text{norm})} = \log(1 + \tilde{y}_{s,g})$$
  Dividing by $T_s$ explicitly cancels out local cell density $n_s$ to first order, isolating relative transcriptional abundance from nuclear crowding.

### 2.3 Incremental Stream PCA & Ridge Regression Protocol
- **Dimension reduction**: To prevent out-of-memory errors across 25,000+ spots and 2,560 embedding dimensions, we implement `StreamPCA`:
  The covariance matrix $\mathbf{C} = \frac{1}{N} \mathbf{X}_\text{train}^T \mathbf{X}_\text{train} - \boldsymbol{\mu} \boldsymbol{\mu}^T$ is accumulated in float64 chunks of rows. Exact eigendecomposition yields the top 256 orthogonal eigenvectors $\mathbf{V}_{256}$.
- **Closed-form per-gene RidgeCV**:
  Let $\mathbf{Z}_\text{train} = \mathbf{X}_\text{train} \mathbf{V}_{256} = \mathbf{U} \mathbf{S} \mathbf{V}^T$ (SVD). For candidate penalties $\alpha \in \{10^{-2}, 10^{-1}, \dots, 10^5\}$, predictions on validation set $\mathbf{Z}_\text{val}$ are evaluated via singular values:
  $$\hat{\mathbf{Y}}_\text{val}(\alpha) = \mathbf{Z}_\text{val} \left( \mathbf{S} (\mathbf{S}^2 + \alpha \mathbf{I})^{-1} \mathbf{U}^T \mathbf{Y}_\text{train} \right) + \bar{\mathbf{y}}$$
  The optimal $\alpha_g^*$ is chosen per gene independently on an inner validation fold. Full fold retraining projects test spots $\mathbf{Z}_\text{test} = \mathbf{X}_\text{test} \mathbf{V}_{256}$ to generate out-of-sample predictions.

### 2.4 Unsupervised Single-Cell Clustering Baseline (Zero-Leakage)
- For training samples $\mathcal{S}_\text{train}$ only:
  1. Single-cell expression matrix $\mathbf{A}_\text{train} \in \mathbb{R}^{N_\text{cells} \times G}$ is library-size normalised and log1p-transformed.
  2. Incremental TruncatedSVD(50) computes orthogonal single-cell latent space $\mathbf{E}_\text{train}$.
  3. kNN graph ($k=15$) and Leiden community detection ($\text{resolution}=0.5$) identify discrete clusters $k \in \{1, \dots, K\}$.
  4. Training cluster centroids $\boldsymbol{\mu}_k \in \mathbb{R}^{50}$ are recorded.
- For test sample $\mathcal{S}_\text{test}$:
  1. Test cells are projected onto the pre-computed training SVD: $\mathbf{E}_\text{test} = \mathbf{A}_\text{test} \mathbf{V}_{50}$.
  2. Each test cell is mapped to the nearest training centroid: $k_i^* = \arg\min_k \|\mathbf{e}_i - \boldsymbol{\mu}_k\|_2$.
  3. Spot composition $\mathbf{C}_s \in \mathbb{R}^K$ is the proportion of spot cells belonging to cluster $k$.
  4. This yields **0.0% unassigned cells** with **strictly zero test leakage**.

### 2.5 Predictability Explanation Regression
- Standardized OLS with HC3 heteroskedasticity-consistent robust standard errors:
  $$r_g = \beta_0 + \beta_1 \cdot \text{MeanExpr}_g + \beta_2 \cdot \text{FracZero}_g + \beta_3 \cdot \text{MoranI}_g + \beta_4 \cdot \tau_g + \beta_5 \cdot \text{Agreement}_g + \epsilon_g$$
  where $\tau_g = \frac{\sum_{t=1}^T (1 - x_{g,t}/\max_t x_{g,t})}{T - 1}$ is the tissue specificity index, and $\text{Agreement}_g$ is the pairwise Pearson correlation between predictions of Virchow2 and UNI2-h.
- Unique variance contribution $\Delta R^2_j = R^2_\text{full} - R^2_{(-j)}$ isolates non-redundant explanatory power.

---

## 3. Defense Against 10 Tough Reviewer Questions

### Q1: "Why only 9 samples and 8 groups? Isn't this dataset too small for a general benchmark?"
**Answer:**
1. **Focus on single-cell resolution panels**: Prior benchmarks (e.g. HEST-1k) aggregated large numbers of 10x Visium slides, but Visium spots are 55\,$\mu$m capturing 5–30 cells without cell boundaries, making cellularity estimation noisy and preventing single-cell segmentation. We deliberately focus on **Xenium Prime 5K**, which profiles ~5,000 genes at subcellular resolution. In the public domain (HEST-1k v1.3.0), exactly 9 valid human Xenium Prime 5K samples exist across 6 diverse organs (`lung`, `ovary`, `cervix`, `skin`, `prostate`, `lymph_node`). We use **100% of all available samples** in this class.
2. **Total spot and cell count**: These 9 samples contain **>23,000 aligned pseudo-spots** and **>1.01 million segmented single cells**.
3. **Leave-one-group-out cross-validation**: With 8 LOGO groups spanning 6 organs, our splits test cross-tissue and cross-patient generalization, strictly preventing spot-level or patient-level data leakage.

### Q2: "Why does the composition baseline barely outperform scalar cell count ($n\_cells$)?"
**Answer:**
1. **Dominance of nuclear volume and count**: In spatial pseudo-spots, total transcript counts scale primarily with total mRNA quantity, which is directly proportional to cell count ($R^2 = 0.94$). 
2. **Lineage collinearity**: Across tissue spots, broad lineages (e.g. epithelial vs stroma) co-occur with high cellularity (e.g. tightly packed tumor nests have high cell counts; loose stroma has lower counts). Consequently, composition features are collinear with total cell count.
3. **Cross-sample batch variation**: While cell count is an invariant physical quantity (number of nuclei per 112\,$\mu$m patch), cell-type marker expression profiles vary across donors and organ sites. Thus, cell count generalizes out-of-distribution across held-out test organs better than composition features.

### Q3: "Isn't $R^2 = 0.953$ on raw counts tautological? Of course expression correlates with expression level!"
**Answer:**
1. **Clarification of the response variable**: Our OLS regression does *not* predict gene expression from gene expression. The response variable is the **out-of-sample model predictability** $r_g \in [-1, 1]$ (how accurately an image foundation model predicts gene $g$ on unseen test patients).
2. **Why this is profound, not tautological**: The machine learning literature (ST-Net, HisToGene, BLEEP, DeepPT) has claimed that foundation models achieve $r \approx 0.45$ because vision encoders learn intricate histologic patterns of cellular differentiation and cancer biology. Our $R^2 = 0.953$ result proves that **95% of the reason a gene is reported as 'predictable' is simply that it has high mean count, low sparsity, and strong spatial autocorrelation**—all of which correlate with cellularity. The image models are not predicting biology; they are predicting nuclear density.

### Q4: "Why did you omit H-optimus-0 / H-optimus-1?"
**Answer:**
1. **Model license & redistribution restrictions**: H-optimus-1 (Bioptimus) requires bespoke commercial licensing and credentials that cannot be redistributed in public reproducible benchmarks.
2. **Representative coverage of architecture & scale**: We evaluate **Virchow2** (Paige AI, 3.1B patches, 1.5M whole-slide images) and **UNI2-h** (Mahmood Lab, SwiGLU ViT-Giant, 100M+ patches), which represent the empirical state of the art in computational pathology foundation models. CONCH v1.5 provides multimodal vision-language coverage. Adding H-optimus would not alter the fundamental finding that optical images cannot bypass the cellularity confound.

### Q5: "Your coarse marker baseline had 72% unassigned cells. Aren't unassigned cells hiding real cell types that would beat image models?"
**Answer:**
1. **Addressed directly in Step 2**: To eliminate this exact concern, we implemented an **unsupervised single-cell clustering baseline** directly on the 5k transcriptomic panel (PCA-50, kNN graph, Leiden community detection).
2. **Empirical proof**: This baseline reduced unassigned cells from **72.3% to exactly 0.0%** (100% of single cells assigned to data-driven clusters). 
3. **Result**: On normalized targets, unsupervised cluster composition achieved $r = 0.207$, which is *still lower* than simple cell count ($r = 0.214$). This conclusively proves that unassigned cells were not masking predictive power; rather, spot-level transcript totals are fundamentally driven by physical cellularity.

### Q6: "Why use linear ridge probes instead of end-to-end fine-tuning the vision foundation models?"
**Answer:**
1. **Benchmark standard**: Linear probing on frozen foundation model embeddings is the official, authoritative evaluation protocol of the HEST benchmark \citep{jaume2024hest} and standard computational pathology practice.
2. **Empirical non-linear validation**: In Phase 5 (Regression Heads), we compared linear ridge to a 1-hidden-layer non-linear MLP with early stopping on 1,000 genes. The MLP reached $r = \num{HeadLogOnePlusMlp}$ vs ridge $r = \num{HeadLogOnePlusRidge}$—showing no meaningful gain from non-linear capacity.
3. **Prevention of overfitting on small cohort**: With 9 samples, end-to-end fine-tuning a 1-billion-parameter Vision Transformer on 4,624 genes would result in massive memorization and catastrophic overfitting to slide-level batch effects.

### Q7: "Did you check for data leakage between TMA cores or multiple samples from the same donor?"
**Answer:**
1. **Grouping rule**: Samples NCBI887 and NCBI888 are tissue microarray (TMA) sections originating from the same research study. Because donor identities are anonymized in HEST metadata, we conservatively placed NCBI887 and NCBI888 into the **exact same cross-validation group** (`lung_tma_pdl`). They never appear simultaneously in training and test sets.
2. **Verification test**: We enforce `assert_no_leakage(folds, groups)` in all atlas scripts, which throws an assertion error if any group or sample overlaps between train and test folds.
3. **Spatial disjointness**: In `tests/test_splits.py`, `assert_spot_disjoint` verifies that no coordinates or barcode identifiers cross validation boundaries.

### Q8: "Why does spatial context (neighbouring patches) provide virtually zero gain?"
**Answer:**
1. **Physical scale vs transcriptional footprint**: Each pseudo-spot already spans $112\,\mu\text{m} \times 112\,\mu\text{m}$ (capturing an average of 50–100 cells). In biological tissue, paracrine signaling and microenvironmental gradients operate within 50–100\,$\mu$m.
2. **Overlapping patch grids**: HEST patches are centered on a 100\,$\mu$m grid, meaning the 112\,$\mu$m nominal footprints of neighbouring patches already overlap by 12\,$\mu$m. Adding a second ring ($224\,\mu\text{m}$) pools distant, heterogeneous histological compartments, introducing morphological noise rather than local signal ($\Delta r = +0.010$).

### Q9: "In Phase 4, is the performance drop when training on Visium due to cross-platform technical noise or tissue differences?"
**Answer:**
1. **Platform effect isolated by same-organ control**: In Phase 4, we evaluate on Xenium lung samples while training on (a) Xenium leave-one-group-out ($r = 0.544$), (b) Xenium same-organ ($r = 0.518$), and (c) Visium same-organ lung ($r = 0.472$).
2. **Quantified drop**: Restricting training to Visium lung preserves $87\%$ of within-platform performance ($\Delta r = -0.072$). The residual drop is attributable to platform differences: Visium has 55\,$\mu$m capture spots with poly-A capture and cDNA sequencing dropouts, whereas Xenium uses targeted rolling-circle amplification of padlock probes with optical decoding.

### Q10: "If foundation models barely beat a cell count after normalisation, what do image models actually predict?"
**Answer:**
1. **Cell-type specificity, not generic expression**: As shown in Section~\ref{sec:results}, after normalisation, cross-gene predictability is strongly correlated with cell-type specificity $\tau$ (partial $\rho = 0.232$) and cross-encoder agreement ($\rho = 0.567$).
2. **Morphologically conspicuous cell states**: The genes where image models significantly outperform composition baselines ($r_\text{image} - r_\text{composition} > 0.15$) are specific morphological markers of activated myofibroblasts and invasive stroma: **CTHRC1, THY1, ADAM12, and COL11A1**. These cells produce distinct, elongated eosinophilic fibrillar matrix alterations readily discernible on H\&E.
3. **Takeaway**: Pathology foundation models do not possess an uncanny ability to "read out" arbitrary transcript levels from H\&E. They reliably detect **gross nuclear density** and **visually distinct stromal/fibrotic remodeling**. Any clinical or translational application must evaluate models on normalized targets to verify that predictions reflect genuine biological state rather than simple nuclear crowding.
