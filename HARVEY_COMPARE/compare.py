#!/usr/bin/env python3
"""Compare only completed runs with matching settings, data and source hashes."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
VARIANTS = ['harvey', 'harvey_cata', 'harvey_cata_upgp']


def compare(group):
    runs = {}
    for file in sorted(group.glob('*/seed*/result.json')):
        row = json.loads(file.read_text())
        if row['synthetic']:
            continue
        metadata = json.loads((file.parent / 'metadata.json').read_text())
        signature = {k: v for k, v in metadata.items() if k not in ('variant', 'seed', 'resume')}
        runs[(row['variant'], row['seed'])] = (row, signature)
    seeds = sorted({s for _, s in runs})
    rows = []
    for seed in seeds:
        base = runs.get(('harvey', seed))
        for variant in VARIANTS:
            item = runs.get((variant, seed))
            if item is None:
                print(f'{group.name}: missing {variant}, seed {seed}')
                continue
            row, signature = item
            matched = base is not None and base[1] == signature and base[0]['poison_hash'] == row['poison_hash']
            output = {k: row[k] for k in ('variant', 'seed', 'clean_accuracy', 'asr', 'elapsed_seconds')}
            output['matched_to_baseline'] = matched
            output['accuracy_delta_pp'] = row['clean_accuracy'] - base[0]['clean_accuracy'] if matched else ''
            output['asr_delta_pp'] = row['asr'] - base[0]['asr'] if matched else ''
            rows.append(output)
    if not rows:
        return
    with (group / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# Comparison: {group.name}', '',
             'Final checkpoint only. ACC and ASR are percentages; deltas are percentage points.', '',
             '| Variant | Seed | ACC | ASR | Δ ACC | Δ ASR | Hours | Matched |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        delta = lambda key: f'{r[key]:+.2f}' if r[key] != '' else '—'
        lines.append(f"| {r['variant']} | {r['seed']} | {r['clean_accuracy']:.2f} | {r['asr']:.2f} | {delta('accuracy_delta_pp')} | {delta('asr_delta_pp')} | {r['elapsed_seconds']/3600:.2f} | {r['matched_to_baseline']} |")
    lines.extend(['', 'Published HARVEY reference: ACC 93.34%, ASR 0.56% (ResNet18/CIFAR-10/BadNets).',
                  'A larger ACC alone does not establish a better backdoor defense; inspect ASR as well.',
                  'One seed is exploratory. Use at least three paired seeds before drawing an improvement conclusion.'])
    for variant in VARIANTS:
        complete = [r for r in rows if r['variant'] == variant and r['matched_to_baseline']]
        if len(complete) >= 2:
            acc = [r['clean_accuracy'] for r in complete]
            asr = [r['asr'] for r in complete]
            lines.append(f'{variant}: n={len(acc)}, ACC {statistics.mean(acc):.2f} ± {statistics.stdev(acc):.2f}, ASR {statistics.mean(asr):.2f} ± {statistics.stdev(asr):.2f} (sample SD).')
    (group / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.', '..')):
        parser.error('Invalid run name')
    for group in ([ROOT / 'runs' / args.run_name] if args.run_name else sorted((ROOT / 'runs').glob('*'))):
        if group.is_dir():
            compare(group)
