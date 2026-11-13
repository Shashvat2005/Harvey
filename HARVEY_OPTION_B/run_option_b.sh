#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${PYTHON:-}" ]]; then
  RUN_PYTHON="$PYTHON"
elif [[ -x ../HARVEY_COMPARE/.venv/bin/python ]]; then
  RUN_PYTHON=../HARVEY_COMPARE/.venv/bin/python
else
  RUN_PYTHON=.venv/bin/python
fi
exec "$RUN_PYTHON" -u experiment.py "$@"
