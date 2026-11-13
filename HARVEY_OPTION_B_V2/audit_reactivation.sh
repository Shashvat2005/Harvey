#!/bin/bash
# Optional POST-SELECTION audit. Does not change any selected model or alpha.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash audit_reactivation.sh runs/<name>/harvey_option_b_v2/seed1' >&2
  exit 1
fi
if [[ -n "${PYTHON:-}" ]]; then
  RUN_PYTHON="$PYTHON"
elif [[ -x ../HARVEY_OPTION_B/.venv/bin/python ]]; then
  RUN_PYTHON=../HARVEY_OPTION_B/.venv/bin/python
elif [[ -x ../HARVEY_COMPARE/.venv/bin/python ]]; then
  RUN_PYTHON=../HARVEY_COMPARE/.venv/bin/python
else
  RUN_PYTHON=.venv/bin/python
fi
"$RUN_PYTHON" - "$1" <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
import torch
from training import make_data,new_model,loader,metrics,seed_all
from refinement import option_direction,write_json
folder=Path(sys.argv[1]).resolve()
if not (folder/'result.json').is_file():raise SystemExit('A completed result.json is required before the stress audit.')
meta=json.loads((folder/'metadata.json').read_text())
args=SimpleNamespace(**{key:value for key,value in meta.items() if key not in ('resolved_device','source_hashes','torch','torchvision','numpy','python')})
args.direction_samples=128;args.direction_lr=.0001;args.direction_epochs=1
seed_all(args.seed);torch.set_num_threads(args.threads);device=torch.device(meta['resolved_device'])
_,data,clean,bad,_,labels=make_data(args)
g=np.load(folder/'refinement_groups.npz')
report={'selection_already_frozen':True,'weights_not_overwritten':True,
        'description':'One 1:1 epoch of low-LR re-exposure to estimated suspicious and likely-clean fitting images; no poison truth selects samples.',
        'base':None,'selected':None}
if not len(g['suspect_fit']) or not len(g['clean_fit']):
    report['skipped']='Insufficient filtered fitting samples'
else:
    model=new_model(args.seed,device)
    for name,filename in [('base','base_model.pt'),('selected','final_model.pt')]:
        model.load_state_dict(torch.load(folder/filename,map_location='cpu',weights_only=True))
        option_direction(model,data,g['clean_fit'],g['suspect_fit'],labels,args,loader,device,mixed=True)
        report[name]=metrics(model,clean,bad,args,device)
write_json(folder/'reactivation_v2.json',report);print(json.dumps(report,indent=2))
PY
