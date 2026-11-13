#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-.venv/bin/python}" -u experiment.py --variant "harvey_cata" "$@"
