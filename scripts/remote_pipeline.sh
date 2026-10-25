#!/usr/bin/env bash
# Phase 2-5 analyses on a Colab VM (large RAM), launched detached by scripts/colab_analysis.py.
# Usage (on the VM): bash scripts/remote_pipeline.sh /content/hest_fm_project/embeddings /content/reports
# Each step writes a marker so a re-launch skips finished steps. Experiments are logged locally afterwards.
set -uo pipefail
EMB=$1
REP=$2
cd "$(dirname "$0")/.."
mkdir -p "$REP"
PY=python
step() {   # step <name> <command...>
  local name=$1; shift
  if [ -f "$REP/.done_$name" ]; then echo "=== $name: already done"; return 0; fi
  echo "=== $name: start $(date +%T)"
  if "$@"; then touch "$REP/.done_$name"; echo "=== $name: ok $(date +%T)"; else echo "=== $name: FAILED $(date +%T)"; fi
}
step atlas    $PY scripts/run_phase2_atlas.py --root "$EMB" --out "$REP/phase2" --no-log
step explain  $PY scripts/analyze_predictability.py --table "$REP/phase2/log1p/gene_table.csv"
step explain_norm $PY scripts/analyze_predictability.py --table "$REP/phase2/lognorm/gene_table.csv"
step figures  $PY scripts/plot_phase2.py --table "$REP/phase2/log1p/gene_table.csv" --out "$REP/phase2/figures"
step context  $PY scripts/run_phase3_context.py --root "$EMB" --out "$REP/phase3" --no-log \
                  --phase2-table "$REP/phase2/log1p/gene_table.csv"
step general  $PY scripts/run_phase4_generalization.py --root "$EMB" --out "$REP/phase4" --no-log
step robust   $PY scripts/run_phase5_robustness.py --root "$EMB" --out "$REP/phase5" --no-log
echo "=== PIPELINE FINISHED $(date +%T)"
