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
