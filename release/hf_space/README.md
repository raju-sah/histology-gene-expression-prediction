---
title: HEST Xenium 5k Atlas Explorer
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: cc-by-nc-sa-4.0
tags:
  - spatial-transcriptomics
  - histopathology
  - pathology-foundation-models
  - xenium-5k
---

# HEST Xenium 5k Gene Expression Predictability Atlas Explorer

An interactive Static web application for benchmarking histology foundation models (Virchow2, UNI2-h, CONCH v1.5) predicting spatial gene expression across 5,001 genes on 6 Xenium 5k tissue sections from HEST-1k.

### Key Finding
**The Cellularity Confound:** Histopathology image foundation models achieve seemingly high correlation ($r \approx 0.51$) under raw `log1p` targets, but this is predominantly confounded by spot cellularity (cell density). After library-size normalisation (`lognorm`), foundation models ($r \approx 0.167$) barely outperform a scalar cell count baseline ($r \approx 0.158$).

- Dataset: [raju-ai/hest-xenium5k-predictability-atlas](https://huggingface.co/datasets/raju-ai/hest-xenium5k-predictability-atlas)
- Code: [TODO: GitHub repository link](https://github.com/TODO)
