#!/usr/bin/env python3
"""Descriptive four-method report; Option B changes the fitting pool and validation."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--first-three-run', default='release-full')
p.add_argument('--fourth-run', default='option-b-full')
a = p.parse_args()
for value in [a.first_three_run, a.fourth_run]:
    if Path(value).name != value or value in ('.', '..'):
        p.error('Run names must be single folder names.')
rows = []
locations = [(ROOT.parent / 'HARVEY_COMPARE/runs' / a.first_three_run, variant)
             for variant in ('harvey', 'harvey_cata', 'harvey_cata_upgp')]
locations.append((ROOT / 'runs' / a.fourth_run, 'harvey_option_b'))
for group, variant in locations:
    files = sorted((group / variant).glob('seed*/result.json'))
    if not files:
        print(f'Missing completed results: {group / variant}')
    for file in files:
        result = json.loads(file.read_text())
        if result.get('synthetic'):
            continue
        metadata = json.loads((file.parent / 'metadata.json').read_text())
        row = dict(method=variant, seed=result['seed'], clean_accuracy=result['clean_accuracy'],
                   asr=result['asr'], target_accuracy=result.get('target_accuracy', ''),
                   hours=result['elapsed_seconds'] / 3600, protocol=result['protocol'],
                   batch_size=metadata['batch_size'], poison_hash=result['poison_hash'],
                   alpha=result.get('selected_alpha', ''),
                   accuracy_delta_vs_own_base_pp=result.get('accuracy_delta_vs_base_pp', ''),
                   asr_delta_vs_own_base_pp=result.get('asr_delta_vs_base_pp', ''))
        rows.append(row)
        if variant == 'harvey_option_b':
            control = result['base_control']
            rows.append(row | dict(method='option_b_base_control', clean_accuracy=control['clean_accuracy'],
                                   asr=control['asr'], target_accuracy=control['target_accuracy'],
                                   hours='', alpha='', accuracy_delta_vs_own_base_pp='', asr_delta_vs_own_base_pp=''))
if rows:
    output = ROOT / 'reports' / f'{a.first_three_run}--{a.fourth_run}'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Four experiment comparison', '',
             '| Method | Seed | Clean ACC | ASR | Target ACC | Alpha |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        target = f"{r['target_accuracy']:.2f}" if r['target_accuracy'] != '' else 'not recorded'
        lines.append(f"| {r['method']} | {r['seed']} | {r['clean_accuracy']:.2f} | {r['asr']:.2f} | {target} | {r['alpha']} |")
    lines.extend(['', 'This is a descriptive comparison. Option B changes the training pool and reserves validation samples.',
                  'Its raw base control isolates the subtraction effect on the same trained weights; changes vs HARVEY also include pool and holdout changes.',
                  'Do not compare different seeds, protocols, batch sizes or poison hashes as matched experiments.',
                  'Target-class test ACC was not recorded by the original three runners; missing values are not zero.',
                  'Synthetic results are excluded. No statistical significance or achieved improvement is asserted.'])
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Saved report in {output}')
else:
    print('No completed real-data results available yet.')
