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

## Phase 1: Reproduce the benchmark (done 2026-10-25, tag `phase-1`)
**Done**
- `hestfm.heads`: `HESTRidge` (exact HEST probe, verified equal to the sklearn reference in tests), `RidgeCVHead`
  (intercept, alpha by inner validation, global or per-gene), `MLPHead` (early stopping on inner validation, refit).
- `hestfm.splits`: HEST folds, LOGO, `assert_no_leakage` (sample + group) and row-level `assert_spot_disjoint`, called inside every CV runner.
- `hestfm.metrics`: per-gene Pearson, bootstrap CI over held-out samples, gene bootstrap.
- `notebooks/colab_phase1_benchmark.ipynb` (built by `scripts/build_notebooks.py`): ResNet50 (HEST truncated, 1024-d) + UNI2-h on IDC/LUNG/COAD, in-notebook sanity ridge.
- Local smoke test executes the notebook's cells on fake bench files with dummy encoders (download/delete, resume, sanity cell, local loader).
- `scripts/run_phase1_eval.py`: hest_ridge / ridge_cv / mlp, compares with published numbers, flags |diff| > 0.03 and "too good" > +0.05.

**Run**: Colab CLI (`colab` v0.7.4, account with Colab Pro), session `hestfm-p1` on L4, ~25 min, ~1 CU (L4 ~1.5 CU/h).
Drive/Secrets are not usable from the CLI: outputs went to VM disk and were downloaded (`colab download`, 296 MB, 30 h5 files);
the user uploaded their HF token file to the VM themselves (`HF_TOKEN_PATH=/content/.hf_token`). VM stopped afterwards.
58,393 patches embedded at ~79 patch/s (ResNet50 + UNI2-h together); every patch matched to expression.

**Key numbers** (`reports/phase1/benchmark_table.md`, exact HEST protocol `hest_ridge`; published in brackets)
- IDC: ResNet50 0.4739 (0.4739), UNI2-h 0.5899 (0.5898)
- LUNG: ResNet50 0.4953 (0.4949), UNI2-h 0.5593 (0.5587)
- COAD: ResNet50 0.2503 (0.2500), UNI2-h 0.3026 (0.3015)   -> max |diff| 0.0011: reproduction confirmed.
- CV-tuned ridge (`ridge_cv`) is better: IDC 0.4769/0.5975, LUNG 0.5380/0.5922, COAD 0.3086/0.3437. MLP ~= HEST ridge.

**Leakage investigation** (triggered by the "too good" flag on COAD ridge_cv, +0.059 over published):
the HEST probe with only alpha changed from 0.0078 to 1e4 gives 0.3099 (ResNet50) / 0.3366 (UNI2-h), i.e. the gain is
regularisation (HEST's fixed alpha = 100/(256*50) is effectively unregularised; our validation picks 1e4-3e4).
Permuted labels: r = -0.013 / -0.007. Folds sample-disjoint; alpha chosen on training rows only. Conclusion: no leakage.

**Decision**: use CV-tuned ridge (per-gene alpha in the atlas) as the main head; report the HEST probe for comparability.

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

## Phase 2: Xenium 5k predictability atlas (done 2026-10-26, tag `phase-2`)
**Data run** (Colab CLI, L4): first attempt lost its VM after a dropped connection (3 samples kept only on VM disk were lost,
~2.5 CU). Fixes: transcripts downloaded to VM disk instead of remote row-group reads (~3 MB/s -> ~250 MB/s; 24 -> ~1 min per
sample); `scripts/colab_driver.py` runs one sample per exec and downloads it immediately, retrying dropped/hung execs.
All 6 Xenium 5k samples downloaded (2.3 GB). HEST cross-check per sample: total-count r 0.994-0.998, median per-gene r
0.86-0.96; orientation 99.8-99.96 % of cells inside patch footprints. H-optimus-1 still gated (skipped).
Visium (Phase 4): only 5/13 downloaded (TENX62, TENX72, MISC13-15, all lung) before the Colab balance reached **0 CU**
(an unrelated A100 session `ntl14c`/`ntlolmo` on the account burned ~5 CU/h). Ovary/cervix Visium missing.

**Local analysis under memory limits** (machine has ~1-3 GB free): gene-chunked atlas (250 genes per pass), float16
embeddings, exact streaming PCA (chunked covariance), fold fits cached on disk (`data/tmp_fitcache/shared`) and reused
across chunks/targets/phases; 3.8 GB RSS watchdog. Peak RSS 2.7-2.9 GB. Equivalence to in-memory runs is unit-tested.

**Key numbers** (5001 genes, LOGO over 5 groups, per-gene-alpha ridge on PCA-256; 95% CI over held-out samples)
| target | Virchow2 | UNI2-h | CONCH v1.5 | composition oracle | cell count only | permuted |
|---|---|---|---|---|---|---|
| log1p | 0.510 [0.42,0.58] | 0.474 | 0.463 | 0.497 | 0.425 | 0.003 |
| lognorm | 0.167 [0.14,0.19] | 0.148 | 0.123 | 0.137 | 0.158 | 0.000 |

**Findings**
- Raw-count predictability (r~0.5) is dominated by cellularity: cell count alone gives 0.425. After library-size
  normalisation, image models reach only 0.12-0.17 mean / 0.10-0.15 median per-gene r, i.e. no better than cell count.
  Not leakage: permutation control ~0, folds group-disjoint (checked in code at sample, group and row level).
- log1p: r tracks expression level (rho 0.88) and Moran's I (rho 0.93; partial 0.60 given expression); image vs
  composition-oracle per-gene r rho 0.98. Genes the image predicts above the oracle: CTHRC1, THY1, SULF1, ADAM12 (activated
  fibroblast / CAF markers), suggesting morphology captures cell states the coarse typing cannot.
- lognorm: cell-type specificity (tau; partial rho 0.38, largest unique OLS delta R^2 0.06) and cross-encoder agreement
  (partial rho 0.68) explain which genes are predictable; OLS R^2 0.65.
- Encoder ranking consistent across targets: Virchow2 > UNI2-h > CONCH v1.5.

**Open issues / limitations**
- Coarse marker typing leaves 60-70 % of cells "unassigned" (sparse marker counts on the 5k panel); weakens tau and the
  composition oracle (which scores below cell count under lognorm). Fix needs cell-level matrices -> another Colab pass.
- Only 5 test groups; CIs are wide.

## Phase 3: Spatial context ablation (done 2026-10-31, tag `phase-3`)
**Done**
- `scripts/run_phase3_context.py` run on both `log1p` and `lognorm` targets for Virchow2 (single, k1 ring mean, k2 ring mean, attn_k1 centre-query attention pooling).
- Peak RSS capped at 3.8 GB via `scripts/run_with_watchdog.sh`; lognorm run completed with `--attention-genes 500` (peak RSS 3.80 GB).
- All per-gene fold predictions, gene tables, gain summaries, and feature correlations saved to `reports/phase3/log1p/` and `reports/phase3/lognorm/`.

**Key numbers** (Virchow2, 5001 genes [attention on 1000 log1p / 500 lognorm random subset], LOGO over 5 groups)
| target | single | k1 ring mean | k2 ring mean | attn_k1 |
|---|---|---|---|---|
| log1p | 0.510 (sd 0.110) | 0.514 (sd 0.108) | 0.513 (sd 0.108) | 0.485 (sd 0.111) |
| lognorm | 0.167 (sd 0.042) | 0.178 (sd 0.042) | 0.184 (sd 0.040) | 0.171 (sd 0.035) |

Gains (variant - single):
- `log1p`: k1 mean gain +0.0038 (median +0.0032); k2 mean gain +0.0031 (median +0.0028); attn_k1 mean gain -0.0269 (median -0.0229).
- `lognorm`: k1 mean gain +0.0102 (median +0.0087; 101 genes gain > 0.05 all folds, 160 majority); k2 mean gain +0.0162 (median +0.0135; 371 genes gain > 0.05 all folds, 550 majority); attn_k1 mean gain -0.0034 (median -0.0061).

**Findings**
- Spatial context adds very little under `log1p` (+0.003-0.004), where cell count and local cellularity already dominate the single-patch signal.
- Under `lognorm`, spatial context provides a modest, consistent gain (+0.010 for k1, +0.016 for k2). Context gains correlate positively with spatial autocorrelation (Moran's I rho 0.42 for k1, 0.57 for k2) and expression level (mean_log1p rho 0.43 for k1, 0.58 for k2), and correlate negatively with marker cell-type specificity (tau rho -0.20 for k1, -0.26 for k2).
- Attention pooling over the 3x3 neighbourhood does not improve over simple ring averaging or single patches (-0.027 on log1p, -0.003 on lognorm).

**Open issues / limitations**
- Attention head evaluated on a fixed random subset (1000 genes for log1p, 500 for lognorm) due to CPU/RAM budget.
- Only Virchow2 evaluated across all context variants; consistent with its top ranking in Phase 2.

## Phase 4: Generalization (done 2026-10-31, tag `phase-4`)
**Done**
- `scripts/run_phase4_generalization.py` run on Virchow2 and UNI2-h across 4,634 shared genes under `lognorm`.
- Evaluated three settings: `xenium_logo` (train on other Xenium groups, all organs), `xenium_same_organ` (train on other Xenium samples of the same organ), and `visium_same_organ` (cross-platform: train on 5 HEST lung Visium samples, test on Xenium pseudo-spots).
- Note on data availability: **Only lung has a cross-platform test** (Visium samples TENX62, TENX72, MISC13-15 are all lung; ovary/cervix Visium were never downloaded due to Colab CU exhaustion).

**Key numbers** (4,634 shared genes, `lognorm` target)
Summary across settings:
| setting | encoder | mean per-sample r | mean drop vs Xenium LOGO | n test samples |
|---|---|---|---|---|
| visium_same_organ | virchow2 | 0.1320 | -0.0208 | 3 (lung only) |
| visium_same_organ | uni2_h | 0.1166 | -0.0259 | 3 (lung only) |
| xenium_logo | virchow2 | 0.1642 | 0.0000 | 6 (all organs) |
| xenium_logo | uni2_h | 0.1478 | 0.0000 | 6 (all organs) |
| xenium_same_organ | virchow2 | 0.2050 | +0.0339 | 5 (lung & ovary) |
| xenium_same_organ | uni2_h | 0.1726 | +0.0186 | 5 (lung & ovary) |

Per-sample performance and drop vs Xenium LOGO:
- `visium_same_organ` (lung only, test on Xenium):
  - Virchow2: NCBI887 r = 0.1259 (drop -0.0112); NCBI888 r = 0.1224 (drop -0.0152); TENX189 r = 0.1477 (drop -0.0361).
  - UNI2-h: NCBI887 r = 0.1017 (drop -0.0339); NCBI888 r = 0.1027 (drop -0.0341); TENX189 r = 0.1455 (drop -0.0097).
- `xenium_same_organ` (within-platform, same-organ):
  - Virchow2: NCBI887 r = 0.1801 (+0.0430); NCBI888 r = 0.1861 (+0.0484); TENX189 r = 0.2524 (+0.0686); TENX160 r = 0.2325 (-0.0002); TENX187 r = 0.1741 (+0.0098).
  - UNI2-h: NCBI887 r = 0.1538 (+0.0182); NCBI888 r = 0.1599 (+0.0231); TENX189 r = 0.1857 (+0.0306); TENX160 r = 0.1880 (-0.0202); TENX187 r = 0.1758 (+0.0413).
- `xenium_logo` baseline:
  - Virchow2: NCBI887 0.1371, NCBI888 0.1376, TENX189 0.1838, TENX160 0.2327, TENX187 0.1643, TENX188 0.1295.
  - UNI2-h: NCBI887 0.1356, NCBI888 0.1367, TENX189 0.1551, TENX160 0.2082, TENX187 0.1344, TENX188 0.1167.

**Findings**
- Cross-platform transfer (Visium -> Xenium) suffers only a modest drop (-0.021 for Virchow2, -0.026 for UNI2-h) despite distinct platforms, spot sizes/resolutions, and laboratory protocols, demonstrating meaningful cross-technology generalisation for morphology-expression mapping.
- Organ-specific training (`xenium_same_organ`) significantly improves upon pan-organ training (`xenium_logo`) by +0.034 (Virchow2) and +0.019 (UNI2-h), indicating that tissue-specific morphology-expression relationships are partially blurred when pooled across diverse organs.
- Virchow2 consistently outperforms UNI2-h across all generalization regimes (cross-platform, pan-organ, same-organ).

**Open issues / limitations**
- Only lung has Visium data locally available; ovary and cervix cross-platform transfer could not be assessed.
- Cervix has only one Xenium sample (TENX188), so no within-platform same-organ training fold exists for cervix.

## Phase 5: Robustness + paper draft (done 2026-10-31, tag `phase-5`)
**Done**
- `scripts/run_phase5_robustness.py` run on Virchow2, UNI2-h, and CONCH v1.5 under the 3.8 GB watchdog (peak RSS 3.54 GB).
- Evaluated 4 axes: random seeds (0, 1, 2 on 1000-gene subset), pseudo-spot expression window (56, 112, 224 um), image field of view (112 um vs central 56 um), and regression heads (ridge vs 1-hidden-layer MLP).
- `scripts/make_paper_numbers.py` generated 109 LaTeX macros and 3 tables in `paper/generated/` without manual values.
- `paper/main.tex` revised to lead with the lognorm finding and cellularity confound, linked to generated figures and tables, and compiled to `paper/main.pdf` via Tectonic with zero `\pending` placeholders.
- Full test suite verified (54/54 passed).

**Key numbers**
- **Seeds** (1000 genes, LOGO): seed standard deviation is negligible (< 1e-12) for all encoders (Virchow2 mean 0.5123, UNI2-h mean 0.4763, CONCH v1.5 mean 0.4649).
- **Expression window** (Virchow2, 5001 genes):
  - 56 um: 0.390 [0.311, 0.444]
  - 112 um (default): 0.510 [0.422, 0.576]
  - 224 um: 0.565 [0.468, 0.640]
  - Aggregating transcripts over larger physical footprints progressively increases predictability by mitigating Poisson sampling sparsity.
- **Field of view** (UNI2-h, 5001 genes):
  - 112 um (full patch): 0.474 [0.390, 0.545]
  - 56 um (central crop): 0.463 [0.381, 0.532]
  - Restricting morphological context to the central quarter-area causes only a minimal drop (-0.011), confirming that local cellular features carry the primary signal.
- **Regression heads** (UNI2-h, 1000 genes, LOGO):
  - Ridge CV: 0.477 [0.387, 0.548]
  - MLP (hidden=512, 3 seeds): 0.484 [0.398, 0.554] (seeds: 0.483, 0.490, 0.478)
  - Non-linear head offers a modest gain (+0.007) over regularised linear projection on PCA features.

**Findings & Manuscript Synthesis**
- The full manuscript `paper/main.pdf` is assembled, leading with the critical negative control: while foundation models achieve $r \approx 0.51$ on raw counts, this predictability is largely explained by spot cellularity ($r=0.425$ from cell count alone). Once library-size normalisation is applied, image models drop to $r \approx 0.12$--$0.17$, matching cell-count baselines ($r=0.158$).
- Residual variation is explained by cell-type specificity and spatial autocorrelation, while spatial context (ring averaging) provides small gains (+0.010 to +0.016) only after cellularity is normalised out.
- Cross-platform generalisation from Visium to Xenium pseudo-spots shows modest transfer loss (-0.021 for Virchow2), and same-organ training substantially outperforms pan-organ training.

**Final gate status**
- Ready for user review. All roadmap phases (0 through 5) complete.
