#!/usr/bin/env bash
# Train + evaluate every synthetic config. Usage: bash scripts/run_all.sh [extra --set overrides]
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in experiments/configs/{copy,induction,assoc_recall}_*.yaml; do
  echo "=== $cfg ==="
  python experiments/train.py --config "$cfg" --set "$@"
  python experiments/eval.py  --config "$cfg"
done
python scripts/make_report.py
