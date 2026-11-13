#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${PYTHON:-}" ]]; then
  RUN_PYTHON="$PYTHON"
elif [[ -x ../HARVEY_OPTION_B/.venv/bin/python ]]; then
  RUN_PYTHON=../HARVEY_OPTION_B/.venv/bin/python
elif [[ -x ../HARVEY_COMPARE/.venv/bin/python ]]; then
  RUN_PYTHON=../HARVEY_COMPARE/.venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  RUN_PYTHON=.venv/bin/python
else
  echo 'No environment found. Run bash setup.sh in this folder, then retry.' >&2
  exit 1
fi
exec "$RUN_PYTHON" -u training.py --variant harvey_option_b_v2 "$@"
