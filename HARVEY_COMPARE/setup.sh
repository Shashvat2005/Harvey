#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
"${BOOTSTRAP_PYTHON:-python3}" -c 'import platform,sys; assert platform.machine()=="arm64", "Use native Apple Silicon Python"; assert sys.version_info >= (3,10), "Use Python 3.10 or newer"'
"${BOOTSTRAP_PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; print("PyTorch:",torch.__version__,"MPS available:",torch.backends.mps.is_available()); assert torch.backends.mps.is_available()'
