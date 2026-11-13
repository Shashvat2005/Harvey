import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock
import zipfile
import torch
from artifacts import atomic_torch_save,create_backup,verify_backup


class ArtifactTests(TestCase):
    def fixture(self,root,complete=True):
        code=root/'code';code.mkdir();(code/'training.py').write_text('# test source\n')
        run=code/'runs/test/harvey_cata_upgp_v2/seed1';run.mkdir(parents=True)
        (run/'metadata.json').write_text(json.dumps({'source_hashes':{'training.py':hashlib.sha256((code/'training.py').read_bytes()).hexdigest()}}))
        atomic_torch_save({'model':torch.ones(2)},run/'checkpoint.pt')
        (run/'metrics.jsonl').write_text('{"event":"test"}\n')
        (run/'incomplete.pt.tmp').write_bytes(b'partial')
        (code/'data').mkdir();(code/'data/skip.bin').write_bytes(b'cache')
        if complete:
            for name in ('base_model.pt','pre_refinement.pt','final_model.pt'):
                atomic_torch_save({'weight':torch.ones(2)},run/name)
            (run/'selection.json').write_text('{"selected":"base"}')
            (run/'result.json').write_text('{"clean_accuracy":90,"asr":1}')
            (run/'candidates').mkdir();atomic_torch_save({'weight':torch.ones(2)},run/'candidates/001-upgp.pt')
        return code,run

    def test_full_backup_roundtrip_and_manifest(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);code,run=self.fixture(root)
            status=create_backup(run,code,root/'backups',True)
            self.assertTrue(status['verified']);self.assertTrue(status['complete'])
            manifest=verify_backup(status['archive'])
            with zipfile.ZipFile(status['archive']) as archive:
                names=archive.namelist()
                self.assertIn(manifest['run_relative_path']+'/checkpoint.pt',names)
                self.assertIn(manifest['run_relative_path']+'/candidates/001-upgp.pt',names)
                self.assertFalse(any(name.endswith('.tmp') or '/data/' in name for name in names))
                archive.extractall(root/'restore')
            restored=torch.load(root/'restore'/manifest['run_relative_path']/'checkpoint.pt',weights_only=True)
            torch.testing.assert_close(restored['model'],torch.ones(2))

    def test_atomic_model_failure_keeps_old_checkpoint(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';atomic_torch_save({'x':torch.ones(1)},path)
            before=path.read_bytes()
            with mock.patch('artifacts.torch.save',side_effect=RuntimeError('simulated write failure')):
                with self.assertRaises(RuntimeError):atomic_torch_save({'x':torch.zeros(1)},path)
            self.assertEqual(path.read_bytes(),before);self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_failed_new_backup_preserves_verified_old_zip(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);code,run=self.fixture(root,False)
            status=create_backup(run,code,root/'backups');before=Path(status['archive']).read_bytes()
            self.assertFalse(status['complete'])
            (run/'result.json').write_text('{}')
            with self.assertRaises(ValueError):create_backup(run,code,root/'backups',True)
            self.assertEqual(Path(status['archive']).read_bytes(),before)

    def test_tamper_and_source_mismatch_detected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);code,run=self.fixture(root)
            status=create_backup(run,code,root/'backups',True)
            corrupt=root/'corrupt.zip'
            with zipfile.ZipFile(status['archive']) as source,zipfile.ZipFile(corrupt,'w') as dest:
                for name in source.namelist():dest.writestr(name,b'changed' if name.endswith('result.json') else source.read(name))
            with self.assertRaises(ValueError):verify_backup(corrupt)
            (code/'training.py').write_text('# changed source\n')
            with self.assertRaises(ValueError):create_backup(run,code,root/'backups',True)


if __name__=='__main__':main()
