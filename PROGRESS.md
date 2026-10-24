# PROGRESS (project memory)

## Phase 0: Recon (done 2026-10-24, tag `phase-0`)
**Done**
- RECON.md: benchmark protocol (read from HEST source), 10 tasks with folds, published results, Xenium/Xenium 5k sample table, encoder access.
- Access verified by downloading only `config.json` files: UNI2-h OK, Virchow2 OK, TITAN (CONCH v1.5) OK, HEST + hest-bench OK.
  H-optimus-1: 403 "requires approval" (expected, pending).
- Env: `.venv` (uv, Python 3.12, CPU torch), 1.4 GB. Project layout, README, LICENSES.md, configs/project.yaml, experiments.csv.
- Local data: `data/meta/` (HEST_v1_3_0.csv, xenium_samples.csv, hest-bench splits + gene lists): < 1 MB.

**Key numbers**
- Published UNI2-h / ResNet50 Pearson: IDC 0.5898 / 0.4739, LUNG 0.5587 / 0.4949, COAD 0.3015 / 0.2500.
- Xenium 5k: 6 true samples (TENX189, TENX160, TENX187, TENX188, NCBI887, NCBI888); ~5001 genes each.

**Decisions**
- Phase 1 tasks: IDC, LUNG, COAD (Xenium-based, 8.9 GB on Colab, relevant to Phase 2).
- TENX190 excluded from the 5k atlas (actually Xenium v1, 541 genes); reused in Phase 4 as v1-vs-5k on the same block.
- No patient IDs exist for 5k samples. Splits are leave-one-group-out with groups in configs/project.yaml; NCBI887+888 (same TMA study) form one group -> 5 groups.
  This is the conservative reading of "patient-level"; flagged for user review.
- Pseudo-spots: our own patch-aligned aggregation (cells whose centroid lies inside the 112 um patch footprint), computed in Colab
  by streaming the transcripts parquet (`he_x`, `he_y`, `cell_id`) row-group by row-group; HEST's own 100 um st h5ad kept as a cross-check.
- User answers (2026-10-24): build later phases ahead while Colab gates wait; Colab Pro (100 CU/month), target L4; Drive root `MyDrive/hest_fm_project`.

**Open issues**
- Patient identity of 5k samples is unknown (see decisions).
- TENX160 transcripts parquet is 104 GB: stream remotely, never download whole.

## Phase 1: Reproduce the benchmark (code done 2026-10-25; **STOP: waiting for the Colab run**)
**Done**
- `hestfm.heads`: `HESTRidge` (exact HEST probe, verified equal to the sklearn reference in tests), `RidgeCVHead`
  (intercept, alpha by inner validation, global or per-gene), `MLPHead` (early stopping on inner validation, refit).
- `hestfm.splits`: HEST folds, LOGO, `assert_no_leakage` (sample + group) and row-level `assert_spot_disjoint`, called inside every CV runner.
- `hestfm.metrics`: per-gene Pearson, bootstrap CI over held-out samples, gene bootstrap.
- `notebooks/colab_phase1_benchmark.ipynb` (built by `scripts/build_notebooks.py`): ResNet50 (HEST truncated, 1024-d) + UNI2-h on IDC/LUNG/COAD, in-notebook sanity ridge.
- Local smoke test executes the notebook's cells on fake bench files with dummy encoders (download/delete, resume, sanity cell, local loader).
- `scripts/run_phase1_eval.py`: hest_ridge / ridge_cv / mlp, compares with published numbers, flags |diff| > 0.03 and "too good" > +0.05.

**Next (when outputs are in `data/embeddings/phase1/`)**: `scripts/run_local_pipeline.sh` (or just `run_phase1_eval.py`), compare, tag `phase-1`.

## Phases 2-5: built ahead (code + tests on synthetic data only; NO real results yet)
- Phase 2 notebook `notebooks/colab_phase2_xenium5k.ipynb`: UNI2-h, Virchow2, CONCH v1.5, H-optimus-1 (auto-skipped while access is pending),
  `uni2_h_c56` (central 56 um), streamed pseudo-spots (windows 0.5/1/2x), coarse cell types, HEST st cross-check; Section B embeds 13 Visium
  samples (lung/ovary/cervix) for Phase 4. Smoke-tested end-to-end on fake HEST files incl. resume and skipped encoder.
- Pre-flight on real data (one TENX189 row group, streamed, nothing stored): 100% of cells fall inside patch footprints in the x/y frame
  (48% if swapped) -> `he_x/he_y` and patch `coords` share a frame. Transcript row groups are spatially local (~1k cells each).
- Phase 2 analysis: `hestfm.atlas` (block-wise per-gene-alpha ridge, sparse targets, cross-encoder agreement, Moran's I),
  `run_phase2_atlas.py` (+ composition-oracle and cell-count baselines, permutation control), `analyze_predictability.py`, `plot_phase2.py`.
- Phase 3 `run_phase3_context.py` (k-ring means, attention pooling); Phase 4 `run_phase4_generalization.py`; Phase 5 `run_phase5_robustness.py`.
- Paper: `paper/main.tex` + `scripts/make_paper_numbers.py`; a test checks every `\num{}` macro in the paper resolves from pipeline outputs.

**Decisions (made autonomously, flagged for review)**
- Pseudo-spot = cells whose centroid (mean transcript position) lies in the 112 um patch footprint; footprints overlap (100 um grid).
- Atlas primary target `log1p(counts)` (HEST-compatible); secondary `lognorm` (library-size normalised) to separate cellularity from composition.
  Phase 4 uses `lognorm` by default (Visium vs Xenium depth).
- Atlas head: ridge with per-gene alpha on PCA-256, alpha chosen on an inner validation group from the training samples.
- "Composition oracle" baseline (measured cell-type counts per spot -> expression) as a reference for how much is explained by composition.
- Phase 5 patch-size sensitivity uses expression windows (56/112/224 um) and an image centre crop (56 um FOV), not re-patching WSIs.
- Visium samples assumed to be different donors from the Xenium samples (different studies).
- Plot script drafted by an orchestra worker (glm-5.3-flash), reviewed; patched to ignore derived `r_*` summary columns.

**Open issues**
- Patient identity unknown for all 5k samples (groups are a proxy). Only 5 test groups -> wide CIs.
- `transformers==4.46.3` pin for TITAN in Colab; if `return_conch` fails, conch_v15 is skipped and recorded in run_summary.json.
- Colab compute-unit estimates are approximate (L4 rate assumed ~5 CU/h).
