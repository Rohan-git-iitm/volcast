#!/usr/bin/env bash
# Retrain on the current window; rebuild the serving containers only if a new champion was promoted.
set -euo pipefail
cd "$(dirname "$0")/.."
exec 9> .pipeline.lock
flock 9
source .venv/bin/activate
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

echo "=== Retrain started $(date -u '+%F %T') UTC ==="
dvc repro

if python -c "import json, sys; sys.exit(0 if json.load(open('metrics/evaluate.json'))['promoted'] else 1)"; then
  echo "New champion promoted: rebuilding serving containers"
  docker compose build
  docker compose up -d
else
  echo "Challenger did not beat the champion: serving unchanged"
fi

git add dvc.lock metrics/ artifacts/
if ! git diff --cached --quiet; then
  git commit -m "Retrain $(date -u +%F)"
fi
echo "=== Retrain finished $(date -u '+%F %T') UTC ==="
