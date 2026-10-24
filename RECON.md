# RECON: HEST-1k / HEST-Benchmark (Phase 0)

Sources read on 2026-10-24: HEST GitHub README and source (commit `3ddb5ea`, 2026-04-16),
`src/hest/bench/{benchmark.py,trainer.py,st_dataset.py}`, `bench_config/bench_config.yaml`,
the tutorial list (`tutorials/4-Running-HEST-Benchmark.ipynb` drives `hest.bench.benchmark`),
HF metadata `MahmoodLab/hest/HEST_v1_3_0.csv`, and the HF file trees of `MahmoodLab/hest` and `MahmoodLab/hest-bench`.

Citation (from the HEST README): Jaume et al., *HEST-1k: A Dataset for Spatial Transcriptomics and Histology
Image Analysis*, NeurIPS 2024 (arXiv:2406.16192). Data CC BY-NC-SA 4.0.

## 1. HEST-Benchmark protocol (as implemented in code)

| Item | Value |
|---|---|
| Input | 224x224 px patches at 0.5 um/px (= 112x112 um) centred on spots, pre-extracted in `hest-bench/<TASK>/patches/*.h5` (`img`, `coords`, `barcode`) |
| Target | 50 genes per task (`var_50genes.json`, highly variable genes), raw counts -> `log1p` (no library-size normalisation; `normalize_adata` only applies `sc.pp.log1p`) |
| Features | frozen encoder embedding -> `StandardScaler` -> `PCA(256, random_state=seed)` fitted on the train split only |
| Head | `sklearn.linear_model.Ridge(solver='lsqr', alpha=100/(256*50), fit_intercept=False, max_iter=1000)` (alpha fixed, not tuned) |
| Metric | Pearson r per gene on the test split; mean over 50 genes per fold; task score = mean over folds; overall = mean over tasks |
| Splits | `splits/train_i.csv` / `test_i.csv`, whole samples per fold (patient-/sample-level) |
| Seed | 1 (default `BenchmarkConfig.seed`) |

Note: `fit_intercept=False` on PCA features with non-centred log1p targets. We reproduce this exactly in
`hestfm.heads.HESTRidge` and keep our own (intercept, CV-tuned) ridge as a separate head.

## 2. Tasks, folds and sizes

| Task | Organ / cancer | Samples | Folds | Test fold sample IDs | hest-bench size |
|---|---|---|---|---|---|
| IDC | breast IDC (Xenium) | 4 | 4 | TENX99 / TENX95 / NCBI785 / NCBI783 | 5.6 GB |
| PRAD | prostate (Visium) | 23 | 2 | MEND154-162 / MEND139-153 | 11.0 GB |
| PAAD | pancreas (Xenium) | 3 | 3 | TENX116 / TENX140 / TENX126 | 1.2 GB |
| SKCM | melanoma (Xenium) | 2 | 2 | TENX117 / TENX115 | 0.5 GB |
| COAD | colon (Xenium) | 4 | 2 | TENX147-149 / TENX111 | 2.5 GB |
| READ | rectum (Visium) | 4 | 2 | ZEN48-49 / ZEN36,40 | 1.6 GB |
| CCRCC | kidney ccRCC (Visium) | 24 | 6 | groups of 4 INT samples | 14.0 GB |
| LUNG | lung LUAD (Xenium) | 2 | 2 | TENX141 / TENX118 | 0.8 GB |
| LYMPH_IDC | axillary lymph node, IDC mets (Visium) | 4 | 4 | NCBI681-684 | 4.0 GB |
| HCC | liver HCC | 2 | 2 | NCBI643 / NCBI642 | 0.9 GB (in repo, not in the published table) |

Gene lists and split CSVs are mirrored locally in `data/meta/hest-bench/` (small files only).

## 3. Published results (README table dated 03.04.26, ridge + PCA-256, Pearson)

| Model | Avg | IDC | PRAD | PAAD | SKCM | COAD | READ | CCRCC | LUNG | LYMPH_IDC |
|---|---|---|---|---|---|---|---|---|---|---|
| H-Optimus-1 | 0.4229 | 0.6024 | 0.3781 | 0.4964 | 0.6589 | 0.3195 | 0.2421 | 0.2533 | 0.5779 | 0.2774 |
| GenBio-PathFM | 0.4197 | 0.5872 | 0.3913 | 0.4959 | 0.6715 | 0.3284 | 0.1785 | 0.2615 | 0.5787 | 0.2842 |
| H-Optimus-0 | 0.4150 | 0.5976 | 0.3848 | 0.4911 | 0.6454 | 0.3086 | 0.2216 | 0.2676 | 0.5590 | 0.2591 |
| **UNI2-h** | 0.4141 | 0.5898 | 0.3569 | 0.5001 | 0.6606 | 0.3015 | 0.2223 | 0.2640 | 0.5587 | 0.2727 |
| **Virchow2** | 0.4034 | 0.5971 | 0.3529 | 0.4779 | 0.6402 | 0.2581 | 0.2074 | 0.2719 | 0.5685 | 0.2568 |
| **CONCH v1.5** | 0.3792 | 0.5440 | 0.3808 | 0.4570 | 0.5517 | 0.2802 | 0.1600 | 0.2176 | 0.5513 | 0.2699 |
| **ResNet50 (tv_in1k)** | 0.3252 | 0.4739 | 0.3044 | 0.3880 | 0.4821 | 0.2500 | 0.0783 | 0.2252 | 0.4949 | 0.2305 |

(25 models in total in the README; bold = encoders used in this project. Copied numbers, not re-derived.)

Phase 1 reproduction targets (chosen for size and relevance to the Xenium atlas): **IDC, LUNG, COAD**
(all Xenium-derived, 8.9 GB total on Colab).

## 4. Xenium and Xenium 5k samples in HEST v1.3.0

HEST v1.3.0 has 79 samples labelled `Xenium` and 7 labelled `Xenium 5k`. Per sample HEST ships: `wsis/` (pyramidal
H&E), `patches/<id>.h5` (224 px tiles resized from 112 um at full res, centres on a 100 um grid, top-left `coords`
in full-res px), `st/<id>.h5ad` (HEST pseudo-spots: 100 um squares on a 100 um grid), `transcripts/<id>_transcripts.parquet`
(per-transcript `cell_id`, `feature_name`, `qv`, `is_gene`, and **H&E-registered `he_x`, `he_y`**), and `xenium_seg/`
cell + nucleus segmentation.

### Findings that matter for Phase 2
1. **TENX190 is not a 5k sample.** Its HEST label is `Xenium 5k`, but `subseries` = "Experiment 1, Xenium v1" and
   `nb_genes` = 541. It is the Xenium v1 run on the same block as TENX189 (Prime 5K, "Experiment 2"). Excluded from the
   5k atlas; it is used in Phase 4 as a same-tissue v1 vs 5k comparison.
2. **True Xenium 5k set (6 samples):** TENX189 (lung LUAD, FFPE), TENX160 (ovary HGSOC, fresh frozen, 104 GB transcripts),
   TENX187 (ovary HGSOC, FFPE), TENX188 (cervix, FFPE), NCBI887 and NCBI888 (healthy lung TMAs, FFPE, study
   "Fishing with Two Lines", PMC12803277).
3. **No patient IDs** for any 5k sample. NCBI887/888 are TMA slides of the same study (PDLTMA06-10 / 06-11) and may share
   donors; NCBI886 (Xenium v1) has the same H&E dimensions as NCBI887 (likely the same or a serial section).
4. Gene panels: 5001 genes (TENX160, TENX188), 5101 (TENX187: 5K + 100 custom), and `nb_genes`=10029 for TENX189/NCBI887/888
   which is 5001 genes + 5028 control/deprecated codewords (checked on TENX189: DeprecatedCodeword 3291,
   UnassignedCodeword 1067, NegControlCodeword 609, NegControlProbe 40, Intergenic 21).
5. Pixel size ~0.274 um/px (0.252 for NCBI887/888). A 112 um patch is 409 px (TENX189) at full res.

### Xenium 5k / Xenium sample table
(`patient` = HEST metadata; sizes in GB on HF.)

| id      | tech      | organ          | disease_state   | oncotree_code   | species      | patient    |   spots |   nb_genes | preservation_method   |   patches_GB |   transcripts_GB |
|:--------|:----------|:---------------|:----------------|:----------------|:-------------|:-----------|--------:|-----------:|:----------------------|-------------:|-----------------:|
| TENX202 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 4  |    5599 |        541 | FFPE                  |         0.66 |             3.58 |
| TENX201 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 4  |    6993 |        541 | FFPE                  |         0.85 |             5.2  |
| TENX200 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 4  |    6410 |        541 | FFPE                  |         0.72 |             3.48 |
| TENX199 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 3  |    3285 |        541 | FFPE                  |         0.35 |             1.74 |
| TENX198 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 3  |    8307 |        541 | FFPE                  |         1.12 |             8.28 |
| TENX197 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 3  |   10083 |        541 | FFPE                  |         1.41 |             2.69 |
| TENX196 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 2  |   10020 |        541 | FFPE                  |         1.41 |             4.38 |
| TENX195 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 2  |    6824 |        541 | FFPE                  |         0.97 |             3.89 |
| TENX194 | Xenium    | Breast         | Healthy         | -               | Homo sapiens | Patient 2  |    7650 |        541 | FFPE                  |         0.99 |             1.89 |
| TENX193 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 1  |    8987 |        541 | FFPE                  |         1.24 |             5.62 |
| TENX192 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 1  |    7972 |        541 | FFPE                  |         0.98 |             7.26 |
| TENX191 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | Patient 1  |    6111 |        541 | FFPE                  |         0.77 |             2.6  |
| TENX190 | Xenium 5k | Lung           | Cancer          | LUAD            | Homo sapiens | n/a        |    7281 |        541 | FFPE                  |         0.89 |             1.44 |
| TENX189 | Xenium 5k | Lung           | Cancer          | LUAD            | Homo sapiens | n/a        |    7189 |      10029 | FFPE                  |         0.9  |             7.19 |
| TENX160 | Xenium 5k | Ovary          | Cancer          | HGSOC           | Homo sapiens | n/a        |   20123 |       5001 | Fresh Frozen          |         2.74 |           103.86 |
| TENX187 | Xenium 5k | Ovary          | Cancer          | HGSOC           | Homo sapiens | n/a        |    8556 |       5101 | FFPE                  |         1.15 |             5.97 |
| TENX188 | Xenium 5k | Cervix         | Cancer          | TODO            | Homo sapiens | n/a        |   10924 |       5001 | FFPE                  |         1.56 |             4.8  |
| NCBI888 | Xenium 5k | Lung           | Healthy         | -               | Homo sapiens | n/a        |   22736 |      10029 | FFPE                  |         2.4  |            12.77 |
| NCBI887 | Xenium 5k | Lung           | Healthy         | -               | Homo sapiens | n/a        |   23044 |      10029 | FFPE                  |         2.42 |            11.37 |
| NCBI886 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | n/a        |   23411 |        541 | FFPE                  |         2.42 |            13.4  |
| NCBI885 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | n/a        |   23209 |        541 | FFPE                  |         2.4  |            13.58 |
| TENX159 | Xenium    | Brain          | Healthy         | -               | Mus musculus | n/a        |    3429 |      13780 | Fresh Frozen          |         0.39 |            15.26 |
| TENX158 | Xenium    | Skin           | Cancer          | SKCM            | Homo sapiens | n/a        |    2179 |      10017 | FFPE                  |         0.24 |             3.57 |
| TENX157 | Xenium    | Prostate       | Cancer          | PRAD            | Homo sapiens | n/a        |    4427 |      10006 | FFPE                  |         0.57 |             2.95 |
| TENX149 | Xenium    | Bowel          | Cancer          | COAD            | Homo sapiens | Patient 1  |    4009 |        541 | FFPE                  |         0.56 |             2.13 |
| TENX148 | Xenium    | Bowel          | Cancer          | COAD            | Homo sapiens | Patient 2  |    4379 |        541 | FFPE                  |         0.55 |             4.02 |
| TENX147 | Xenium    | Bowel          | Cancer          | COAD            | Homo sapiens | Patient 5  |    3997 |        541 | FFPE                  |         0.53 |             2.43 |
| TENX143 | Xenium    | Lymphoid       | Diseased        | -               | Homo sapiens | n/a        |    4889 |      11094 | FFPE                  |         0.65 |            13.96 |
| NCBI884 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | Patient 27 |    2504 |        541 | FFPE                  |         0.16 |             0.87 |
| NCBI883 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | Patient 28 |    1277 |        541 | FFPE                  |         0.08 |             0.23 |
| NCBI882 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 1  |    1435 |        541 | FFPE                  |         0.15 |             0.82 |
| NCBI881 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 2  |    1392 |        541 | FFPE                  |         0.11 |             0.63 |
| NCBI880 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 3  |    1168 |        541 | FFPE                  |         0.11 |             0.59 |
| NCBI879 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | Patient 4  |    1017 |        541 | FFPE                  |         0.07 |             0.26 |
| NCBI876 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | Patient 7  |    1062 |        541 | FFPE                  |         0.04 |             0.17 |
| NCBI875 | Xenium    | Lung           | Healthy         | -               | Homo sapiens | Patient 7  |     999 |        541 | FFPE                  |         0.08 |             0.32 |
| NCBI873 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 9  |    1484 |        541 | FFPE                  |         0.13 |             0.55 |
| NCBI870 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 12 |    1073 |        541 | FFPE                  |         0.09 |             0.27 |
| NCBI867 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 15 |    3499 |        541 | FFPE                  |         0.39 |             1.58 |
| NCBI866 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 16 |    1604 |        541 | FFPE                  |         0.17 |             0.95 |
| NCBI865 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 17 |    3062 |        541 | FFPE                  |         0.33 |             1.71 |
| NCBI864 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 18 |    2917 |        541 | FFPE                  |         0.3  |             1.04 |
| NCBI861 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 21 |    1121 |        541 | FFPE                  |         0.11 |             0.26 |
| NCBI860 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 22 |    1282 |        541 | FFPE                  |         0.1  |             0.24 |
| NCBI859 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 23 |    1116 |        541 | FFPE                  |         0.11 |             0.5  |
| NCBI858 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 24 |     837 |        541 | FFPE                  |         0.09 |             0.33 |
| NCBI857 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 25 |    1767 |        541 | FFPE                  |         0.18 |             0.83 |
| NCBI856 | Xenium    | Lung           | Diseased        | -               | Homo sapiens | Patient 26 |    1607 |        541 | FFPE                  |         0.15 |             0.68 |
| TENX142 | Xenium    | Ovary          | Cancer          | HGSOC           | Homo sapiens | n/a        |    3743 |        541 | FFPE                  |         0.43 |             1.16 |
| TENX141 | Xenium    | Lung           | Cancer          | LUAD            | Homo sapiens | patient 1  |    4390 |        541 | FFPE                  |         0.48 |             2.44 |
| TENX140 | Xenium    | Pancreas       | Cancer          | PAAD            | Homo sapiens | patient 3  |    4513 |        541 | FFPE                  |         0.54 |             1.26 |
| TENX139 | Xenium    | Bowel          | Cancer          | COADREAD        | Homo sapiens | n/a        |    4167 |        541 | FFPE                  |         0.52 |             3.02 |
| TENX138 | Xenium    | Brain          | Cancer          | GBM             | Homo sapiens | n/a        |   10641 |        541 | FFPE                  |         1.37 |            10.69 |
| TENX137 | Xenium    | Bone           | Healthy         | -               | Mus musculus | n/a        |    9200 |        541 | FFPE                  |         0.38 |             1.91 |
| TENX136 | Xenium    | Bone           | Healthy         | -               | Mus musculus | n/a        |   12535 |        541 | FFPE                  |         0.77 |             2.9  |
| TENX135 | Xenium    | Bone           | Healthy         | -               | Mus musculus | n/a        |    7590 |        541 | FFPE                  |         0.6  |             2.02 |
| TENX134 | Xenium    | Lymphoid       | Cancer          | ALL             | Homo sapiens | n/a        |    3811 |        541 | FFPE                  |         0.21 |             0.49 |
| TENX133 | Xenium    | Lymphoid       | Healthy         | -               | Homo sapiens | n/a        |    3811 |        541 | FFPE                  |         0.21 |             0.08 |
| TENX132 | Xenium    | Bone           | Healthy         | -               | Homo sapiens | n/a        |    9682 |        541 | FFPE                  |         0.83 |             0.11 |
| TENX127 | Xenium    | Bowel          | Healthy         | -               | Mus musculus | n/a        |    4453 |        541 | Fresh Frozen          |         0.4  |             7.66 |
| TENX126 | Xenium    | Pancreas       | Cancer          | PAAD            | Homo sapiens | patient1   |    2190 |        541 | FFPE                  |         0.3  |             0.49 |
| TENX125 | Xenium    | Lymphoid       | Diseased        | -               | Homo sapiens | patient 1  |   10005 |        541 | FFPE                  |         0.97 |             5.37 |
| TENX124 | Xenium    | Lymphoid       | Diseased        | -               | Homo sapiens | patient 2  |   12707 |        541 | FFPE                  |         0.89 |             4.17 |
| TENX123 | Xenium    | Skin           | Healthy         | -               | Homo sapiens | n/a        |   10810 |        541 | FFPE                  |         0.86 |             0.16 |
| TENX122 | Xenium    | Skin           | Healthy         | -               | Homo sapiens | n/a        |    5626 |        541 | FFPE                  |         0.51 |             0.12 |
| TENX121 | Xenium    | Liver          | Healthy         | -               | Homo sapiens | n/a        |    9200 |        541 | FFPE                  |         0.89 |             1.52 |
| TENX120 | Xenium    | Liver          | Cancer          | HCC             | Homo sapiens | n/a        |    4004 |        538 | FFPE                  |         0.4  |             1.65 |
| TENX119 | Xenium    | Heart          | Healthy         | -               | Homo sapiens | n/a        |    2263 |        541 | FFPE                  |         0.23 |             0.19 |
| TENX118 | Xenium    | Lung           | Cancer          | LUAD            | Homo sapiens | patient 2  |    3115 |        541 | FFPE                  |         0.28 |             0.73 |
| TENX117 | Xenium    | Skin           | Cancer          | SKCM            | Homo sapiens | patient 1  |    1830 |        541 | FFPE                  |         0.2  |             0.94 |
| TENX116 | Xenium    | Pancreas       | Cancer          | PAAD            | Homo sapiens | patient 2  |    3090 |        538 | FFPE                  |         0.32 |             1.69 |
| TENX115 | Xenium    | Skin           | Cancer          | SKCM            | Homo sapiens | patient 2  |    3886 |        541 | FFPE                  |         0.25 |             1.4  |
| TENX114 | Xenium    | Bowel          | Healthy         | -               | Homo sapiens | n/a        |    6006 |        541 | FFPE                  |         0.68 |             1.73 |
| TENX111 | Xenium    | Bowel          | Cancer          | COAD            | Homo sapiens | n/a        |    6643 |        541 | FFPE                  |         0.8  |             4.68 |
| TENX110 | Xenium    | Whole organism | Healthy         | -               | Mus musculus | n/a        |   26565 |        541 | FFPE                  |         2.43 |            22.05 |
| TENX106 | Xenium    | Kidney         | Healthy         | -               | Homo sapiens | n/a        |    2730 |        541 | FFPE                  |         0.3  |             0.61 |
| TENX105 | Xenium    | Kidney         | Cancer          | PRCC            | Homo sapiens | n/a        |    1738 |        541 | FFPE                  |         0.18 |             0.3  |
| TENX99  | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 1  |   25080 |        541 | FFPE                  |         3.18 |             6.46 |
| TENX98  | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 1  |   26070 |        541 | FFPE                  |         3.14 |             6.26 |
| TENX97  | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 2  |   11845 |        541 | FFPE                  |         1.2  |             5.66 |
| TENX96  | Xenium    | Breast         | Cancer          | ILC             | Homo sapiens | patient 3  |    8624 |        541 | FFPE                  |         0.96 |             2.76 |
| TENX95  | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 2  |   11845 |        541 | FFPE                  |         1.18 |             4.86 |
| TENX94  | Xenium    | Breast         | Cancer          | ILC             | Homo sapiens | patient 3  |    9152 |        541 | FFPE                  |         0.94 |             2.25 |
| NCBI785 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 4  |    4180 |        541 | FFPE                  |         0.63 |             2.39 |
| NCBI784 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 4  |    4180 |        541 | FFPE                  |         0.45 |             1.54 |
| NCBI783 | Xenium    | Breast         | Cancer          | IDC             | Homo sapiens | patient 5  |    3869 |        541 | FFPE                  |         0.54 |             0.94 |

## 5. Encoders and access (checked 2026-10-24, config files only)

| Encoder | HF repo | Arch / dim | Access | License |
|---|---|---|---|---|
| ResNet50 | timm/resnet50.tv_in1k | truncated after layer3, avg-pool, 1024-d (HEST `resnet50`) | public | BSD-3 |
| UNI2-h | MahmoodLab/UNI2-h | ViT-g/14 (timm `vit_giant_patch14_224`), 1536-d | granted | CC BY-NC-ND 4.0 |
| Virchow2 | paige-ai/Virchow2 | ViT-H/14 + 4 reg tokens; CLS + mean patch token = 2560-d | granted | CC BY-NC-ND 4.0 |
| CONCH v1.5 | MahmoodLab/TITAN (`conch_v1_5_pytorch_model.bin`, `model.return_conch()`) | ViT-L/16, 768-d | granted | CC BY-NC-ND 4.0 |
| H-optimus-1 | bioptimus/H-optimus-1 | ViT-g/14, 1536-d | **pending (403)** | CC BY-NC-ND 4.0 |
