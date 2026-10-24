# Predicting spatial gene expression from H&E with pathology foundation models

## Working rules
- Data: HEST-1k (HuggingFace MahmoodLab/hest, gated, CC BY-NC-SA 4.0) and HEST-Benchmark. Cite them. Non-commercial research use only.
- Model licenses: UNI2-h is CC BY-NC-ND 4.0. Using it to extract embeddings is fine; never publish or upload fine-tuned weights of any gated model. Record each model's license in LICENSES.md.
- Local disk is ~10 GiB usable and is shared with other projects. NEVER download whole-slide images or raw HEST samples locally. Only small metadata files and final embeddings/results live locally.
- Heavy work (image patching, embedding extraction) happens in Google Colab notebooks that I run:
  - Write them to notebooks/colab_*.ipynb. Each notebook must: mount Google Drive; read the HuggingFace token from Colab Secrets (`from google.colab import userdata; userdata.get('HF_TOKEN')`) and never contain a token; download one HEST sample at a time, process it, save outputs and delete the raw files before the next sample; be resumable (skip samples whose outputs already exist); save embeddings as float16 (.h5 or .npy) plus expression matrices and metadata to a clearly named Drive folder; print progress and an end-of-run summary.
  - Use mixed precision and a sensible batch size for A100/L4.
  - When a notebook is ready, STOP and tell me: which notebook, which GPU type (A100 or L4 preferred), the estimated runtime and Colab compute units, the Drive output folder, and which files I should download into data/embeddings/ afterwards.
- Do NOT use the Kaggle GPU quota (reserved for another project).
- Never ask for, print, or store tokens or passwords.
- Splits are patient-level (and cross-sample or cross-platform where stated). Never put patches of one patient/slide in both train and test. Enforce this with a unit test.
- Primary metric: per-gene Pearson correlation between predicted and measured expression on held-out patients, averaged over the gene set, as in HEST-Benchmark. Also report per-gene distributions and 95% bootstrap CIs over held-out samples.
- Hyperparameters are chosen on validation folds only. Log every experiment in experiments.csv (date, commit, encoder, task, split, metric).
- Never invent citations. Related-work references stay as TODO placeholders for me to fill in.
- Stop conditions (STOP and tell me): tests fail after 3 fix attempts; disk < 5 GiB; anything needs sudo; a model's gated access fails; a result looks implausibly good (investigate leakage first); or a decision isn't covered here.

## Phase 0: Recon
- Read the HEST paper/README, the HEST GitHub tutorials (including 4-Running-HEST-Benchmark.ipynb) and the published benchmark results. Write RECON.md covering: benchmark tasks, gene sets, splits, top encoders and their scores, and the list of Xenium and Xenium 5k samples (IDs, organ, cancer type, patient, size).
- Verify access to UNI2-h, Virchow2 and CONCH v1.5 (via TITAN) by downloading only their config/metadata files, not full weights. Record the results in PROGRESS.md.
- Set up the local Python environment (venv + requirements.txt), project structure (src/, notebooks/, tests/, configs/, data/ gitignored, reports/), and README.

## Phase 1: Reproduce the benchmark (Colab)
- Write a Colab notebook that reproduces HEST-Benchmark for an ImageNet ResNet50 baseline and UNI2-h on 2–3 benchmark tasks and saves the embeddings to Drive.
- Locally: regression heads (ridge + small MLP) and evaluation code with unit tests, runnable on the downloaded embeddings.
- STOP for me to run the notebook. After I download the outputs, evaluate them and compare with the published numbers. If they differ substantially, investigate before continuing.

## Phase 2: Xenium 5k predictability atlas (main contribution)
Research question: on the new Xenium 5k samples, which genes are predictable from H&E morphology, and what explains predictability?
- Colab notebook: extract frozen embeddings with UNI2-h, Virchow2 and CONCH v1.5 (plus H-optimus-1 if accessible) on the Xenium 5k samples; resumable. STOP for me to run it.
- Pseudo-spot aggregation: aggregate cell-level Xenium counts into patch-aligned pseudo-spots. Document patch size and magnification.
- Train ridge/MLP heads per encoder with patient-level splits. Compute per-gene Pearson for all ~5k genes.
- Explain predictability: correlate per-gene performance with mean expression, variance, sparsity, cell-type specificity (from marker annotation) and cross-encoder agreement. Build a gene-level results table and figures.

## Phase 3: Spatial context ablation
- Compare a single patch against neighborhood aggregation of frozen embeddings (mean of k-ring neighbors; small attention pooling over neighbors). Report which genes gain from context.

## Phase 4: Generalization
- Cross-sample and, where the data allows, cross-platform tests (train on Visium samples of the same organ, test on Xenium pseudo-spots). Report the performance drop.

## Phase 5: Robustness + write-up (STOP, final gate)
- Multiple seeds, CIs, sensitivity to patch size.
- Paper draft in paper/ (LaTeX, workshop-style two-column), with every number generated from results files by script. Related work as TODO placeholders.
- STOP for my review.
