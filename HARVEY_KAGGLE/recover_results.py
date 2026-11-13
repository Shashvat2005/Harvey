"""Read/export completed results without training or prior notebook variables."""
import json
from pathlib import Path
import re
import zipfile


def recover_results(search_roots=None, output_root=None):
    roots = [Path(p) for p in search_roots] if search_roots is not None else [Path('/kaggle/working'), Path('/kaggle/input')]
    output = Path(output_root) if output_root is not None else Path('/kaggle/working/harvey-results')
    discovered = {}
    for root in roots:
        if not root.is_dir():
            continue
        for file in sorted(root.rglob('result.json')):
            if 'harvey_cata_upgp' not in file.parts:
                continue
            try:
                result = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if result.get('variant') == 'harvey_cata_upgp':
                discovered[str(file.resolve())] = (file, result)
    if not discovered:
        print('No completed HARVEY+CATA+UPGP result.json found in the current files.')
        print('This does not prove the training failed. The earlier session may have ended or the run may be in a saved notebook version.')
        print('Open the completed notebook VERSION and inspect its Output files. Attach those outputs here if needed.')
        print('If no files were preserved, inspect the original executed notebook output for its printed FINAL CIFAR-10 RESULT.')
        checkpoints = [p for root in roots if root.is_dir() for p in root.rglob('checkpoint.pt') if 'harvey_cata_upgp' in p.parts]
        for path in checkpoints:
            print('Checkpoint found (not proof of completion):', path)
        print('This recovery cell does not start or resume training.')
        return []
    output.mkdir(parents=True, exist_ok=True)
    exports = []
    for index, (file, result) in enumerate(discovered.values(), start=1):
        folder = file.parent
        label = 'SYNTHETIC CHECK ONLY' if result.get('synthetic') else 'COMPLETED RESULT'
        print(f'\n{label}: {file}')
        print(json.dumps(result, indent=2))
        before = result.get('before_upgp')
        if before:
            for key in ('clean_accuracy', 'asr'):
                if key in before and key in result:
                    print(f'UPGP change in {key}: {result[key] - before[key]:+.2f} percentage points')
        stem = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{folder.parent.parent.name}-{folder.name}')
        export = output / f'{index:02d}-{stem}-results.zip'
        with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ('metadata.json', 'result.json', 'metrics.jsonl', 'training.log',
                         'split_quality.json', 'split.npz', 'poison_ids.npy',
                         'before_upgp.pt', 'final_model.pt'):
                path = folder / name
                if path.is_file():
                    archive.write(path, arcname=f'{folder.name}/{name}')
            environment = next((parent / 'notebook_environment.json' for parent in folder.parents
                                if (parent / 'notebook_environment.json').is_file()), None)
            if environment is not None:
                archive.write(environment, arcname='notebook_environment.json')
        print('Export created:', export)
        exports.append(export)
    print('\nDownload ZIPs from the Kaggle file browser: /kaggle/working/harvey-results/')
    print('If viewing a saved notebook version, use its Output files instead of inline links.')
    print('ZIPs exclude checkpoint.pt; preserve the original complete run folder separately for resume.')
    return exports


if __name__ == '__main__':
    recover_results()
