# Transactions on Machine Learning Research (TMLR) Submission Notes

## 1. Scope & Fit for TMLR
- **Target Journal**: *Transactions on Machine Learning Research* (TMLR).
- **Relevant Area**: Applied Machine Learning, Computational Pathology, Spatial Biology, Benchmark & Critical Evaluation.
- **Why this paper fits TMLR criteria**:
  - TMLR explicitly evaluates submissions based on **technical correctness and claims supported by evidence**, rather than subjective "perceived novelty" or "hype".
  - Negative results, critical evaluations, and rigorous re-evaluations that uncover fundamental confounders in widely cited machine learning models are explicitly within scope and encouraged by TMLR's editorial policy.
  - This paper systematically evaluates claims from multiple high-profile machine learning frameworks (ST-Net, HisToGene, BLEEP, DeepPT, HEST), proving that apparent cross-gene predictability is driven by spot cellularity rather than learned morphological transcriptional mapping.

## 2. Review Format Compliance (Single-Blind)
- TMLR uses **single-blind review**: reviewers know the author identities, but reviews are public.
- The manuscript (`paper/main.tex`) includes full technical derivations, explicit methodology, complete hyperparameter specifications, and exact dataset identifiers.
- No confidential credentials, API keys, or proprietary weights are referenced or included.

## 3. Reproducibility & Open Science Compliance
1. **Deterministic Protocol**:
   - As documented in Section 4.7, our linear probe evaluation on exact PCA features contains zero stochastic elements; variance across random seeds is $< 10^{-12}$.
2. **Code & Environment**:
   - The entire codebase is pinned in `requirements.txt` and python virtual environment.
   - Every single number and table in the paper is deterministically generated from raw experimental CSVs via `scripts/make_paper_numbers.py`. No manual editing or hardcoded values exist.
3. **Data Availability**:
   - All spatial transcriptomics data are available through HEST-1k (CC BY-NC-SA 4.0).
   - All derived per-gene predictability metrics and summary tables are packaged for public release on Hugging Face.
