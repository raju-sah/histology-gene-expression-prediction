# Related work: cited sources and novelty note

Every entry below is in `paper/refs.bib`. Titles, venues and first authors were checked on 2026-11-01 against
Crossref (DOIs) or the arXiv API (arXiv IDs). That check only shows the papers exist and match the metadata;
**the author must still read each paper** and confirm the sentence that cites it is fair.

| Key | Paper | Venue | Identifier | Cited for |
|---|---|---|---|---|
| `he2020integrating` | He et al., Integrating spatial gene expression and breast tumour morphology via deep learning (ST-Net) | Nat. Biomed. Eng. 2020 | doi:10.1038/s41551-020-0578-x | early H&E-to-expression model |
| `pang2021leveraging` | Pang, Su, Li, Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors (HisToGene) | bioRxiv 2021 | doi:10.1101/2021.11.28.470212 | transformer with spatial context |
| `xie2023bleep` | Xie et al., Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning (BLEEP) | NeurIPS 2023 | arXiv:2306.01859 | contrastive image-expression model |
| `hoang2024deeppt` | Hoang et al., A deep-learning framework to predict cancer treatment response from histopathology images through imputed transcriptomics (DeepPT) | Nat. Cancer 2024 | doi:10.1038/s43018-024-00793-2 | downstream use of imputed expression |
| `jaume2024hest` | Jaume et al., HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis | NeurIPS 2024 (Datasets & Benchmarks) | arXiv:2406.16192 | data and benchmark protocol |
| `chen2024uni` | Chen et al., Towards a general-purpose foundation model for computational pathology (UNI) | Nat. Med. 2024 | doi:10.1038/s41591-024-02857-3 | UNI |
| `chen2024uni2` | Mahmood Lab, UNI2-h model card | Hugging Face | huggingface.co/MahmoodLab/UNI2-h | UNI2-h (no paper found) |
| `vorontsov2024virchow` | Vorontsov et al., A foundation model for clinical-grade computational pathology and rare cancers detection (Virchow) | Nat. Med. 2024 | doi:10.1038/s41591-024-03141-0 | Virchow (v1) |
| `zimmermann2024virchow2` | Zimmermann et al., Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology | arXiv 2024 | arXiv:2408.00738 | Virchow2, the model used |
| `lu2024conch` | Lu et al., A visual-language foundation model for computational pathology (CONCH) | Nat. Med. 2024 | doi:10.1038/s41591-024-02856-4 | CONCH |
| `ding2025titan` | Ding et al., A multimodal whole-slide foundation model for pathology (TITAN) | Nat. Med. 2025 | doi:10.1038/s41591-025-03982-3 | TITAN; source of CONCH v1.5 |
| `hafemeister2019normalization` | Hafemeister & Satija, Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression | Genome Biol. 2019 | doi:10.1186/s13059-019-1874-1 | count normalisation |
| `kleshchevnikov2022cell2location` | Kleshchevnikov et al., Cell2location maps fine-grained cell types in spatial transcriptomics | Nat. Biotechnol. 2022 | doi:10.1038/s41587-021-01139-4 | cell abundance vs. counts in spots |

Removed in the 2026-11-01 review: a UNI2 entry pointing to arXiv:2407.03613 (an unrelated mathematics paper), an
ST-Net entry that carried the cell2location DOI, and a wrong cell2location title.

## Novelty note (honest)

- **Known:** total counts per spot depend on how many cells it holds, and normalisation and deconvolution methods
  exist for exactly this reason (Hafemeister 2019; Kleshchevnikov 2022).
- **What this work adds:** a transcriptome-wide (~4.6k genes) comparison of frozen pathology foundation models against
  a per-spot cell-count baseline and composition baselines, on both raw and library-size-normalised targets, with
  group-level splits and paired bootstrap intervals.
- **Not established:** that no prior paper has done this. We did not run a systematic literature search. Before
  submission, search for recent work on cell-density or cellularity baselines and normalised targets in
  histology-to-expression benchmarks, and soften or update the paper's framing if such work exists.
- **Do not claim** that prior methods "misinterpreted" their results; we did not re-evaluate them.
