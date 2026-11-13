# Standalone recovery/export: no RUN_DIR, CODE_DIR or training variables needed.
import importlib.util
import json
from pathlib import Path

completed=[]
for root in (Path('/kaggle/working'),Path('/kaggle/input')):
    if root.is_dir():
        for file in root.rglob('result.json'):
            if any(part in ('harvey_cata_upgp_v2','harvey_v2_control') for part in file.parts):completed.append(file)
if not completed:
    print('No completed results in current/attached files. Check the saved notebook version Output tab.')
    print('Do not retrain just to download. If the VM files were discarded without a saved version or download, this cell cannot restore them.')
for file in sorted(set(completed)):
    result=json.loads(file.read_text())
    print('\nSYNTHETIC CHECK' if result.get('synthetic') else '\nCOMPLETED RESULT')
    print(file);print(json.dumps(result,indent=2))
    source_root=next((parent for parent in file.parents if (parent/'artifacts.py').is_file() and (parent/'training.py').is_file()),None)
    if source_root is None:
        print('Full source root not found alongside this run; attach the full saved output/backup to verify a self-contained export.')
        continue
    spec=importlib.util.spec_from_file_location('harvey_export_artifacts',source_root/'artifacts.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    try:
        status=exporter.create_backup(file.parent,source_root,Path('/kaggle/working/harvey-v2-full-backups'),require_complete=True)
        print('VERIFIED FULL BACKUP:',json.dumps(status,indent=2))
    except Exception as error:
        print('BACKUP ERROR:',error)
        print('Original run files are unchanged; preserve them via Quick Save with output files enabled.')
backup_root=Path('/kaggle/working/harvey-v2-full-backups')
if backup_root.is_dir():
    for archive in sorted(backup_root.glob('*-FULL.zip')):print('DOWNLOAD FROM FILE BROWSER:',archive)
print('Remote persistence: Save Version > Save & Run All for a new run; Quick Save with output files for an interactive run.')
