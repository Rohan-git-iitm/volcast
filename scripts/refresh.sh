#!/usr/bin/env bash
# Daily data refresh: move the window, download it, then delete old data and rebuild through features.
set -euo pipefail
cd "$(dirname "$0")/.."
exec 9> .pipeline.lock
flock 9
source .venv/bin/activate
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
trap 'python -m src.monitoring.pipeline_status refresh failed' ERR

python -m src.data.roll_window set
dvc repro download
python -m src.data.roll_window cleanup
dvc repro features
git add configs/config.yaml dvc.lock artifacts/thresholds.json metrics/bars.json
if git diff --cached --quiet; then
  echo "Nothing to commit"
else
  git commit -m "Daily data refresh $(date -u +%F)"
fi
python -m src.monitoring.pipeline_status refresh ok
