#!/usr/bin/env python3
"""Build the standalone V2 Kaggle notebook, without modifying earlier artifacts."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files=['training.py','refinement.py','cata.py','upgp.py','vendor/models/target_models.py',
       'vendor/cifar_badnets.png','tests/test_v2.py','artifacts.py','tests/test_artifacts.py']
manifest={};memory=io.BytesIO()
with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in files:
        content=(ROOT/name).read_bytes();manifest[name]=hashlib.sha256(content).hexdigest()
        z.writestr(zipfile.ZipInfo(name),content,compress_type=zipfile.ZIP_DEFLATED)
payload=base64.b64encode(memory.getvalue()).decode();digest=hashlib.sha256(memory.getvalue()).hexdigest()
cells=[]
def add(kind,source):
    cell={'cell_type':kind,'id':f'v2-{len(cells):02d}','metadata':{},'source':source.strip()+'\n'}
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)
add('markdown','''# HARVEY plus guarded CATA and conservative UPGP V2

This is a NEW exploratory version, addressing the earlier run's excessive clean rejection and destructive UPGP update. It targets higher clean accuracy and lower ASR; neither outcome is guaranteed. Original runs are preserved.

Select a Kaggle GPU (T4 or P100) and enable Internet for CIFAR-10, then use **Save Version > Save & Run All**. One CUDA GPU is used. All required source code is embedded; no GitHub clone is needed. The full classifier schedule remains 100 epochs. CATA now uses two adaptation epochs, smaller correction scales and a per-cycle ordinary-HARVEY fallback. UPGP uses short, low-LR trials with validation guards. This is a revised hybrid, not an exact reproduction of the standalone CATA/UPGP papers.

Five percent of training IDs are reserved before fitting, stratified by observed labels. True poison IDs are used only in AFTER-selection audit reports. Clean validation and suspicious response are estimated proxies, not guaranteed ground truth. No final test accuracy/ASR selects the edits. A fallback or all-GA reference history must be reported honestly; merely running this notebook does not prove CATA or UPGP helped.''')
add('code','''PROFILE = "full"                 # "smoke" uses tiny RANDOM SYNTHETIC images
PROTOCOL = "release"             # release: 14 reference epochs; paper-text: 10
METHOD = "harvey_cata_upgp_v2"    # optional matched control: "harvey_v2_control"
SEED = 1
BATCH_SIZE = 128
CPU_THREADS = 2
CATA_SCALES = [0.1, 0.3]
POLISH_LRS = [0.001, 0.005]
UPGP_LRS = [0.00001, 0.0001]
RECOVER_CLEAN = True             # False: ablate label-agreement fitting-data recovery
UPGP_EPOCHS = 2                  # per LR arm; both restart from the utility anchor
BACKUP_INTERVAL_MINUTES = 15     # also back up on phase changes and on exit
RUN_NAME = f"v2-{PROFILE}-{PROTOCOL}-b{BATCH_SIZE}-recover{int(RECOVER_CLEAN)}-saved"
WORK_DIR = "/kaggle/working/harvey-cata-upgp-v2-saved"
REQUIRE_CUDA = True
CIFAR10_ROOT = None              # optional attached cache containing cifar-10-batches-py/
RESTORE_RUN_DIR = None           # previous output's COMPLETE .../<METHOD>/seed1 folder
assert PROFILE in {"full", "smoke"}
assert METHOD in {"harvey_cata_upgp_v2", "harvey_v2_control"}
assert PROTOCOL in {"release", "paper-text"}
assert BATCH_SIZE >= 2 and BATCH_SIZE % 2 == 0
''')
add('markdown','''## Environment and embedded source

Keep Kaggle's installed PyTorch/torchvision pair. Resume checks enforce source and library versions. Full CUDA training has not been verified on the development Mac. Local synthetic tests verify execution only, not improved accuracy. For offline data, set CIFAR10_ROOT before running.''')
add('code','''import os,sys,json,subprocess,signal,shutil,zipfile,base64,hashlib,io
from pathlib import Path
import torch,torchvision,numpy as np
print("Python",sys.version.split()[0],"torch",torch.__version__,"torchvision",torchvision.__version__)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("Enable a CUDA GPU in Kaggle settings and restart the session.")
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
if DEVICE.startswith("cuda"):
    print("Training uses only GPU 0:",torch.cuda.get_device_name(0))
if CIFAR10_ROOT:
    if not (Path(CIFAR10_ROOT)/"cifar-10-batches-py/data_batch_1").is_file():
        raise FileNotFoundError("CIFAR10_ROOT must contain cifar-10-batches-py/data_batch_1")
    os.environ["HARVEY_CIFAR10_ROOT"]=str(CIFAR10_ROOT)
else:
    os.environ.pop("HARVEY_CIFAR10_ROOT",None)
CODE_DIR=Path(WORK_DIR);CODE_DIR.mkdir(parents=True,exist_ok=True)
''')
archive_source='SOURCE_ARCHIVE = (\n'+'\n'.join('    '+repr(payload[i:i+120]) for i in range(0,len(payload),120))+'\n)\n'
add('code',archive_source+f'ARCHIVE_SHA256={digest!r}\nFILE_HASHES={manifest!r}\n'+'''
raw=base64.b64decode(SOURCE_ARCHIVE)
assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA256
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert set(archive.namelist())==set(FILE_HASHES)
    for name in archive.namelist():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe archive path")
        content=archive.read(name);assert hashlib.sha256(content).hexdigest()==FILE_HASHES[name]
        target=CODE_DIR/relative
        if target.exists() and target.read_bytes()!=content:raise RuntimeError("Existing source differs; use a new WORK_DIR")
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
(CODE_DIR/"source_manifest.json").write_text(json.dumps(FILE_HASHES,indent=2)+"\\n")
print("Verified",len(FILE_HASHES),"embedded files")
subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=CODE_DIR,check=True)
RUN_DIR=CODE_DIR/"runs"/RUN_NAME/METHOD/f"seed{SEED}"
if RESTORE_RUN_DIR:
    previous=Path(RESTORE_RUN_DIR)
    for required in ("metadata.json","checkpoint.pt"):
        if not (previous/required).is_file():raise FileNotFoundError(previous/required)
    if not (RUN_DIR/"metadata.json").exists():
        if RUN_DIR.exists() and any(RUN_DIR.iterdir()):raise RuntimeError("Restore destination is nonempty without metadata")
        RUN_DIR.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(previous,RUN_DIR,dirs_exist_ok=True)
RUN_DIR.mkdir(parents=True,exist_ok=True)
settings={"profile":PROFILE,"protocol":PROTOCOL,"method":METHOD,"seed":SEED,"batch":BATCH_SIZE,
          "cata_scales":CATA_SCALES,"polish_lrs":POLISH_LRS,"upgp_lrs":UPGP_LRS,
          "upgp_epochs_per_arm":UPGP_EPOCHS,"recover_clean":RECOVER_CLEAN}
settings_file=RUN_DIR/"notebook_settings.json"
if settings_file.exists() and json.loads(settings_file.read_text())!=settings:
    raise RuntimeError("Saved notebook settings differ; use a new RUN_NAME.")
settings_file.write_text(json.dumps(settings,indent=2)+"\\n")
environment={"python":sys.version,"torch":torch.__version__,"torchvision":torchvision.__version__,
             "numpy":np.__version__,"device":DEVICE,"cuda_runtime":torch.version.cuda,
             "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
if not (RUN_DIR/"notebook_environment.json").exists():
    (RUN_DIR/"notebook_environment.json").write_text(json.dumps(environment,indent=2)+"\\n")
BACKUP_DIR=CODE_DIR.parent/"harvey-v2-full-backups"
print("Results/checkpoints:",RUN_DIR)
print("Verified full backups will be written to:",BACKUP_DIR)
print("Kaggle persistence: Save Version > Save & Run All; for interactive runs Quick Save with output files enabled.")
''')
add('markdown','''## Train the revised method

Full profile: 20 warmup epochs, up to 20 reference cycles, 100 final classifier epochs, then bounded validation-selected refinement. Do not start another process in the same run directory.

Reference cycles compare CATA proposals with ordinary HARVEY GA on fixed reference probes and estimated retention; fallback cycles are explicitly counted. After final classifier training, two-view semantic agreement filters the suspected pool. Rejected fitting images may re-enter utility fitting only with >=0.95 original/weak-view agreement on their observed label; validation images never re-enter fitting. This is fallible label-agreement recovery, separately switchable with RECOVER_CLEAN. Utility fine-tuning and UPGP are separate ablations. Both keep BatchNorm running buffers fixed. UPGP projects its forget gradient, retains the clean gradient separately, uses a parameter-change cap, and runs at most two epochs per low-LR arm. No eligible samples or no accepted validation gain means the prior model is retained.

Only after selection is written are clean test ACC, target-class ACC and triggered test ASR evaluated for the base, utility-stage and selected weights. These are distinct checkpoints in this revised pipeline; they are not automatically equivalent to your previous standalone runs.''')
add('code','''import time
sys.path.insert(0,str(CODE_DIR))
from artifacts import create_backup
command=[sys.executable,"-u",str(CODE_DIR/"training.py"),"--variant",METHOD,
         "--profile",PROFILE,"--protocol",PROTOCOL,"--seed",str(SEED),
         "--batch-size",str(BATCH_SIZE),"--threads",str(CPU_THREADS),"--device",DEVICE,
         "--run-name",RUN_NAME,"--resume","--cata-scales",*map(str,CATA_SCALES),
         "--polish-lrs",*map(str,POLISH_LRS),"--upgp-lrs",*map(str,UPGP_LRS),
         "--upgp-epochs",str(UPGP_EPOCHS)]
if not RECOVER_CLEAN:command.append("--no-recover-clean")
print("Starting:"," ".join(command))
process=None;last_backup=0.;last_phase=None
try:
    with (RUN_DIR/"training.log").open("a",buffering=1) as log_file:
        process=subprocess.Popen(command,cwd=CODE_DIR,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,start_new_session=True)
        for line in process.stdout:
            print(line,end="",flush=True);log_file.write(line)
            try:entry=json.loads(line)
            except (ValueError,TypeError):entry={}
            if entry.get("event")=="checkpoint_saved":
                phase=entry.get("phase")
                if phase!=last_phase or time.monotonic()-last_backup>=BACKUP_INTERVAL_MINUTES*60:
                    try:
                        backup_status=create_backup(RUN_DIR,CODE_DIR,BACKUP_DIR)
                        print("VERIFIED ROLLING BACKUP:",backup_status["archive"],flush=True)
                        last_backup=time.monotonic();last_phase=phase
                    except Exception as backup_error:
                        print("BACKUP WARNING (run files remain on disk):",backup_error,flush=True)
        status=process.wait()
    if status:raise RuntimeError(f"Training exited with {status}; inspect training.log. The last complete checkpoint can be resumed.")
finally:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    # Runs on success, interruption and Python-level errors, without relying on
    # the user remembering the final cell. A VM-wide termination can bypass it.
    if (RUN_DIR/"metadata.json").exists():
        try:
            backup_status=create_backup(RUN_DIR,CODE_DIR,BACKUP_DIR,
                                        require_complete=(RUN_DIR/"result.json").exists())
            print("VERIFIED FULL BACKUP:",json.dumps(backup_status,indent=2),flush=True)
        except Exception as backup_error:
            print("FULL BACKUP FAILED:",backup_error,flush=True)
            print("Save the complete working output files now; the independent export cell can retry.",flush=True)
print("IMPORTANT: retain the output through Kaggle Save Version, then download the FULL.zip from Output files.")
''')
add('markdown','''## Verify and download the full experiment

The training cell already creates and verifies a FULL.zip, including checkpoints and models, on phase changes, every 15 minutes when a new checkpoint is emitted, and on exit. This cell independently prints saved results and retries the complete export. It does not start training or require RUN_DIR.

Download the FULL.zip and backup-status JSON from `/kaggle/working/harvey-v2-full-backups/` through the file browser or a saved notebook version's **Output** tab. The archive contains source, settings, environment, current resume state, reference end state, first/last references, base/utility/final models, every evaluated refinement candidate model, split IDs/scores, candidate tables and logs. Its manifest verifies every included file with SHA-256. Dataset caches are re-downloadable and excluded; obsolete historical optimizer checkpoints are overwritten rather than accumulating past Kaggle's output quota.

**Persistence is a separate step:** for a new run use **Save Version > Save & Run All**. For an already completed interactive run use **Quick Save > Advanced settings > save output files**, then inspect that version's Output and download the FULL.zip. A draft save or local ZIP is not confirmation of remote retention. Sudden VM termination before a saved version/download can still lose local files; periodic backups can also be downloaded during a long run.
''')
add('code', (ROOT/'export_cell.py').read_text())
add('markdown','''## What counts as improvement

Report the selected test metrics alongside the base control, utility-stage control, selected edit kind and reference GA/CATA cycle counts. An unchanged model is a fallback, not a successful unlearning result. A clean-polish gain is not automatically a CATA/UPGP gain. Validation safeguards can fail on true test data; report regressions.

Compare against your own baseline with matching seeds, architecture, attack and access to validation data, then confirm across multiple paired seeds. This V2 reserves training validation images and changes the reference/postprocessing policy, so comparisons to the paper or the old full-data baseline must state those differences. True poison masks and test metrics are audit-only. The generic weak-view filter does not use the known trigger mask, but assumes a single target and semantically stable weak augmentations. BadNets results do not establish effectiveness against other attacks.

The prior Kaggle run's missing weights cannot be reconstructed; this notebook starts a new V2 experiment unless a compatible V2 checkpoint is attached.''')
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                            'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},
        nbformat=4,nbformat_minor=5)
path=ROOT/'HARVEY_CATA_UPGP_V2_Kaggle.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n')
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built',path,len(cells),'cells',path.stat().st_size,'bytes')
