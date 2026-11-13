#!/usr/bin/env python3
"""Descriptive comparison only; never used by training or candidate selection."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('result',type=Path);p.add_argument('--baseline-result',type=Path)
p.add_argument('--baseline-accuracy',type=float,default=93.34);p.add_argument('--baseline-asr',type=float,default=.56)
a=p.parse_args();r=json.loads(a.result.read_text())
if r.get('synthetic'):raise SystemExit('Synthetic metrics are not research results')
if a.baseline_result:
    b=json.loads(a.baseline_result.read_text());label=str(a.baseline_result)
else:b={'clean_accuracy':a.baseline_accuracy,'asr':a.baseline_asr};label='Published reference or user-entered numeric reference; not a matched replication'
print('Reference:',label)
for name,row in [('Reference',b),('V2 base control',r['base_control']),('After utility stage',r['before_refinement']),('V2 selected',r)]:
    print(f"{name}: ACC={row['clean_accuracy']:.2f}% ASR={row['asr']:.4f}%")
print('Selected edit:',r['selected_edit']);print('Reference choices:',r['reference_counts'])
print(f"Descriptive change vs reference: ACC {r['clean_accuracy']-b['clean_accuracy']:+.2f} pp; ASR {r['asr']-b['asr']:+.4f} pp")
print('These numbers do not establish statistical significance or matched data/hardware/protocol. Report fallback and utility-stage gains separately.')
