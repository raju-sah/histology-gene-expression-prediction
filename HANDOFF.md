# Handoff prompt (paste into the next agent)

You are continuing an autonomous research project in `/home/raju/AI-ML Projects/histology-gene-expression-prediction`
(git repo). Before anything else read, in this order: `ROADMAP.md` (rules + phases), `PROGRESS.md` (state, numbers,
decisions), `RECON.md` (data/protocol facts). Follow ROADMAP's working rules exactly (no tokens printed/stored, patient/
group-level splits, never invent citations, STOP conditions incl. disk < 5 GiB). Commit after each step with a
descriptive message ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` only if you are that model;
otherwise use your own attribution. Update PROGRESS.md after each phase and `git tag phase-N`.

## Where things stand (2026-10-31)
- Done + tagged: phase-0, phase-1 (HEST-Benchmark reproduced, max |diff| 0.0011), phase-2 (Xenium 5k atlas, log1p and
  lognorm; see PROGRESS.md for numbers and findings).
- Phase 3: `log1p` done and committed (`reports/phase3/log1p/`): Virchow2 single 0.510, +k1 ring mean 0.514,
  +k2 0.513, attention (1000-gene subset) 0.485 -> context adds ~nothing. **`lognorm` run NOT done** (`reports/phase3/lognorm/` empty).
- Phase 4, Phase 5, paper numbers: code ready and unit-tested, NOT run on real data.
- Colab balance is 0 CU: no more Colab work. Everything runs locally.

## Hard constraints of this machine
- ~1-3 GB free RAM (shared with other apps). Run every analysis through the RSS watchdog:
  `scripts/run_with_watchdog.sh <log> 3800000 .venv/bin/python -u scripts/<script>.py ...` (run it in the background with
  nohup; poll the log; never run two analyses at once).
- Disk was 2.0 GB free at handoff (below the 5 GiB STOP rule). `data/tmp_fitcache/shared` (3.2 GB) is a regenerable cache
  of PCA/SVD fold fits that makes reruns fast. Ask the user before deleting it or before continuing with < 5 GiB free.
- Analyses are gene-chunked (`--chunk 250`, float16 embeddings, streaming PCA, disk fit cache). Do not "simplify" this
  back to in-memory runs: they need 6-7 GB RAM.
- `pkill -f <pattern>` matches its own shell command line and kills it; kill by PID instead.

## Remaining steps
1. Phase 3 lognorm:
   `nohup scripts/run_with_watchdog.sh data/phase3_lognorm.log 3800000 .venv/bin/python -u scripts/run_phase3_context.py --encoders virchow2 --attention-encoder virchow2 --y-kind lognorm --out reports/phase3/lognorm --phase2-table reports/phase2/lognorm/gene_table.csv &`
   (peak RSS of the log1p run was 3.73 GB, close to the cap; if the watchdog kills it, add `--attention-genes 500`).
   Then summarise gains in PROGRESS.md, commit, tag `phase-3`.
2. Phase 4 (lung only: Visium TENX62, TENX72, MISC13-15 are local; ovary/cervix Visium were never downloaded):
   `scripts/run_phase4_generalization.py --encoders virchow2 uni2_h` via the watchdog. Report per-sample r and the drop
   vs Xenium LOGO; state clearly that only lung has a cross-platform test. Commit, tag `phase-4`.
3. Phase 5: `scripts/run_phase5_robustness.py --encoders virchow2 uni2_h conch_v15` via the watchdog (seeds on a
   1000-gene subset, expression window 56/112/224 um, FOV 56 vs 112 um, ridge vs MLP). If too slow, use `--skip` per part
   and run parts separately.
4. Paper: `scripts/make_paper_numbers.py` then `cd paper && tectonic -X compile main.tex`. Every number must come from
   the generated macros; check `\pending` placeholders. The paper must lead with the lognorm finding (image models
   ~ cell-count baseline after library-size normalisation) and the log1p cellularity confound; keep related work as TODOs.
5. Update PROGRESS.md, commit, tag `phase-5`, then STOP for the user's review (final gate).

## Known limitations to report (do not hide)
- Coarse marker cell typing leaves 60-70 % of cells unassigned (weakens tau and the composition oracle).
- 5 test groups only (no patient IDs; NCBI887/888 grouped), wide CIs. H-optimus-1 never accessible.
- Phase 3 attention and Phase 5 seeds use fixed random 1000-gene subsets (CPU/RAM budget).

Run `.venv/bin/python -m pytest -q` (52+ tests) after any code change.
