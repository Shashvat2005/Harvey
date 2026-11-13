"""Atomic training artifacts and self-contained verified run backups.

These archives live on the current disk. Remote Kaggle persistence still needs
Save & Run All or Quick Save with output files; this module cannot force it.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import torch


def atomic_torch_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    try:
        with temp.open('wb') as stream:
            torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _eligible(path):
    return path.is_file() and not path.is_symlink() and not path.name.endswith('.tmp')


def verify_backup(path):
    """Check every archived byte against the backup's own SHA-256 manifest."""
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('BACKUP_MANIFEST.json'))
        expected=set(manifest['files'])|{'BACKUP_MANIFEST.json','RESTORE_INSTRUCTIONS.txt'}
        if set(archive.namelist())!=expected:raise ValueError('Archive member list differs from manifest')
        for name,info in manifest['files'].items():
            checksum=hashlib.sha256();size=0
            with archive.open(name) as stream:
                while True:
                    chunk=stream.read(1024*1024)
                    if not chunk:break
                    checksum.update(chunk);size+=len(chunk)
            if size!=info['bytes'] or checksum.hexdigest()!=info['sha256']:
                raise ValueError(f'Archive checksum failed: {name}')
    return manifest


def create_backup(run_dir,code_root,backup_root,require_complete=False):
    """One run + actual source; excludes datasets, caches and temporary writes.

    Rolling replacement is atomic: failure leaves the previous verified ZIP.
    Current files are captured, not every historical optimizer state. Atomic
    model saves make active-run snapshots safe; logs are a captured prefix.
    """
    run_dir,code_root,backup_root=map(lambda p:Path(p).resolve(),(run_dir,code_root,backup_root))
    if not run_dir.is_dir():raise FileNotFoundError(run_dir)
    if run_dir==code_root or not run_dir.is_relative_to(code_root):
        raise ValueError('Run folder must be inside the source root')
    if backup_root==run_dir or backup_root.is_relative_to(run_dir):
        raise ValueError('Backups must be outside the run folder')
    if (run_dir/'metadata.json').is_file():
        metadata=json.loads((run_dir/'metadata.json').read_text())
        for name,expected in metadata.get('source_hashes',{}).items():
            source=code_root/name
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Saved run source differs: {name}. Back up with its original code, not an updated training source.')
    completed=(run_dir/'result.json').is_file()
    if require_complete and not completed:raise ValueError('Completed result.json missing')
    if completed:
        required=['metadata.json','checkpoint.pt','base_model.pt','pre_refinement.pt','final_model.pt','selection.json','result.json']
        missing=[name for name in required if not (run_dir/name).is_file()]
        if missing:raise ValueError(f'Completed run missing required artifacts: {missing}')
    items={}
    for path in code_root.rglob('*'):
        relative=path.relative_to(code_root)
        if any(part in ('runs','data','.venv','venv','__pycache__','backups') for part in relative.parts):continue
        if _eligible(path) and (path.suffix in ('.py','.png','.json','.md','.txt','.sh') or path.name=='requirements.txt'):
            items['code/'+relative.as_posix()]=path
    for path in run_dir.rglob('*'):
        if _eligible(path):items['code/'+path.relative_to(code_root).as_posix()]=path
    backup_root.mkdir(parents=True,exist_ok=True)
    estimated=sum(path.stat().st_size for path in items.values())
    if shutil.disk_usage(backup_root).free<estimated*1.1+1024*1024:
        raise OSError('Insufficient free disk for another complete backup; original run files were not removed')
    stem=f'{run_dir.parent.parent.name}-{run_dir.parent.name}-{run_dir.name}'
    output=backup_root/f'{stem}-FULL.zip';temp=output.with_name(output.name+'.tmp')
    manifest={'format':1,'created_utc':datetime.now(timezone.utc).isoformat(),'complete':completed,
              'run_relative_path':'code/'+run_dir.relative_to(code_root).as_posix(),
              'checkpoint_present':(run_dir/'checkpoint.pt').is_file(),'files':{},
              'scope':'All current run artifacts plus source/config. Dataset caches and obsolete historical epoch states are excluded.'}
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for name,path in sorted(items.items()):
                checksum=hashlib.sha256();size=0
                with path.open('rb') as source,archive.open(name,'w',force_zip64=True) as target:
                    while True:
                        chunk=source.read(1024*1024)
                        if not chunk:break
                        checksum.update(chunk);size+=len(chunk);target.write(chunk)
                manifest['files'][name]={'bytes':size,'sha256':checksum.hexdigest()}
            archive.writestr('BACKUP_MANIFEST.json',json.dumps(manifest,indent=2))
            archive.writestr('RESTORE_INSTRUCTIONS.txt',
                'Extract this ZIP and attach its contents to Kaggle. Set RESTORE_RUN_DIR to '+manifest['run_relative_path']+'.\n'
                'Use the exact code under code/ and the same run name/settings/library/device versions.\n'
                'Checkpoint files contain Python objects: restore only your own trusted backups.\n'
                'This is a local archive, not confirmation that Kaggle retained a saved version.\n')
        verify_backup(temp)
        temp.replace(output)
        summary={key:manifest[key] for key in ('created_utc','complete','checkpoint_present','run_relative_path','scope')}
        summary.update(archive=str(output),archive_bytes=output.stat().st_size,file_count=len(manifest['files']),verified=True)
        status=backup_root/f'{stem}-backup-status.json';temporary=status.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(status)
        return summary
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
