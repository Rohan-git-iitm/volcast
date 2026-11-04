#!/usr/bin/env bash
# Usage: scripts/simulate_drift.sh 3   (volatility x3)   |   scripts/simulate_drift.sh 1   (stop)
curl -s -X POST "localhost:8000/admin/simulate-drift?factor=${1:-3}"
echo
