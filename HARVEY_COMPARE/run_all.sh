#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Run bash setup.sh first, or set PYTHON to an existing compatible Python.' >&2
  exit 1
fi
# Each process exits before the next starts, releasing its MPS allocation.
for seed in ${SEEDS:-1}; do
  for variant in harvey harvey_cata harvey_cata_upgp; do
    "$PYTHON" -u experiment.py --variant "$variant" --seed "$seed" "$@"
  done
done
"$PYTHON" compare.py
