#!/usr/bin/env bash
# Run every local analysis step after the Colab outputs are in data/embeddings/. Stops on the first error.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
step() { echo; echo "=== $*"; }
if [ -d data/embeddings/phase1 ]; then step "Phase 1 eval"; $PY scripts/run_phase1_eval.py; fi
if [ -d data/embeddings/phase2/xenium5k ]; then
  step "Phase 2 atlas";    $PY scripts/run_phase2_atlas.py
  step "Phase 2 explain";  $PY scripts/analyze_predictability.py
  step "Phase 2 figures";  $PY scripts/plot_phase2.py --table reports/phase2/log1p/gene_table.csv --out reports/phase2/figures
  step "Phase 3 context";  $PY scripts/run_phase3_context.py
  if [ -d data/embeddings/phase4/visium ]; then step "Phase 4"; $PY scripts/run_phase4_generalization.py; fi
  step "Phase 5 robustness"; $PY scripts/run_phase5_robustness.py
fi
step "Paper numbers"; $PY scripts/make_paper_numbers.py
