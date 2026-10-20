#!/usr/bin/env bash
# Rebuild every number, figure and the model file from the two TSV files.
#   ./run_all.sh          reuses notebook 2's cached out-of-centre predictions if data and code are unchanged
#   ./run_all.sh --fresh  refits the leave-one-centre-out models (a few minutes, same results)
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--fresh" ]; then export RECOMPUTE_LOCO=1; fi
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
run_nb() { .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$1"; }

$PY tests/test_cytogenetics.py
$PY tests/test_scores.py
run_nb notebooks/01_data_audit.ipynb         # figures F1-F4, outputs/tables, outputs/audit.json
run_nb notebooks/02_model_comparison.ipynb   # outputs/validation (incl. selection.json), figures F5-F9
$PY scripts/03_train_final.py > outputs/train_log.txt
$PY scripts/predict.py --clinical df_clinical_train.tsv --mutations mut_clinical_train.tsv \
    --out outputs/predictions_training_file.csv --cards outputs/cards \
    --card-ids P132699 P132718 P105706 P132710 P132729 P132716
run_nb notebooks/03_final_model_and_tool.ipynb
echo "done"
