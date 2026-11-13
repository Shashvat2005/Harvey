#!/usr/bin/env python3
"""Build a self-contained Kaggle notebook from the reviewed CUDA source snapshot."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
buffer = io.BytesIO()
manifest = {}
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / 'source').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(ROOT / 'source').as_posix()
        content = path.read_bytes()
        manifest[relative] = hashlib.sha256(content).hexdigest()
        archive.writestr(zipfile.ZipInfo(relative), content, compress_type=zipfile.ZIP_DEFLATED)
payload = base64.b64encode(buffer.getvalue()).decode()
checksum = hashlib.sha256(buffer.getvalue()).hexdigest()
cells = []


def add(kind, source):
    cell = dict(cell_type=kind, id=f'cell-{len(cells):02d}', metadata={}, source=source.strip() + '\n')
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''# HARVEY plus CATA plus UPGP on Kaggle

Self-contained CIFAR-10 / ResNet18 / BadNets 10% poisoning experiment. This notebook embeds the training code and trigger; it does not need GitHub access. Select a **GPU accelerator** in Kaggle settings and enable **Internet** to download CIFAR-10, or attach an existing CIFAR-10 cache.

The full profile runs 20 warmup epochs, up to 20 reference cycles at 14 epochs per cycle, four CATA adaptation epochs per cycle, 100 final classifier epochs, then five UPGP postprocessing epochs. It uses one GPU only, even if Kaggle exposes two. There is no AMP, reduced dataset, shortened full schedule, or automatic batch-size change.

CATA replaces reference-model benign-set unlearning. UPGP's gradient-projection component postprocesses the final model using HARVEY-estimated sets; its separate perturbation detector is not used. This is an exploratory hybrid, not a reproduction of all three papers or a guaranteed accuracy gain. If HARVEY finds no suspected poison, UPGP is explicitly skipped.

GPU execution has not been tested on this Mac. CPU execution and notebook checks are recorded beside the notebook. Synthetic smoke metrics are execution checks only.''')
add('code', '''# Edit this cell before Run All.
PROFILE = "full"                # "full" or "smoke" (tiny synthetic data)
PROTOCOL = "release"            # "release" (14 ref epochs), "paper-text" (10)
SEED = 1
BATCH_SIZE = 128                # use 64 in a NEW run if GPU memory is insufficient
CPU_THREADS = 2
RUN_NAME = f"kaggle-{PROFILE}-{PROTOCOL}-b{BATCH_SIZE}"

# Optional: attach a previous notebook output and point to its COMPLETE seed folder.
# Example: /kaggle/input/<previous-output>/harvey-cata-upgp/runs/
#          kaggle-full-release-b128/harvey_cata_upgp/seed1
RESTORE_RUN_DIR = None

# Optional offline CIFAR-10 cache ROOT, containing cifar-10-batches-py/.
CIFAR10_ROOT = None

# True by default: a smoke check still needs a GPU in Kaggle.
# False is only for local CPU validation; a full CPU run would be very slow.
REQUIRE_CUDA = True

assert PROFILE in {"full", "smoke"}
assert PROTOCOL in {"release", "paper-text"}
assert isinstance(SEED, int) and SEED >= 0
assert isinstance(BATCH_SIZE, int) and BATCH_SIZE >= 2
assert isinstance(CPU_THREADS, int) and CPU_THREADS >= 1
assert RUN_NAME and "/" not in RUN_NAME and "\\\\" not in RUN_NAME and RUN_NAME not in {".", ".."}
''')
add('markdown', '''## Check the environment

Keep Kaggle's preinstalled PyTorch/torchvision pair. This cell does not replace it. CUDA absence stops execution before training. If the import fails, restart with a compatible Kaggle image rather than mixing arbitrary torch and torchvision versions.''')
add('code', '''import os
import sys
import platform
import json
from pathlib import Path
import torch
import torchvision
import numpy as np
from PIL import Image

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU. Select a GPU accelerator in Kaggle settings, then restart and Run All.")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    print("Only GPU 0 will be used.")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if CIFAR10_ROOT is not None:
    cache = Path(CIFAR10_ROOT)
    if not (cache / "cifar-10-batches-py" / "data_batch_1").is_file():
        raise FileNotFoundError("CIFAR10_ROOT must contain cifar-10-batches-py/data_batch_1")
    os.environ["HARVEY_CIFAR10_ROOT"] = str(cache)
else:
    os.environ.pop("HARVEY_CIFAR10_ROOT", None)
''')
add('markdown', '''## Extract the embedded code

Code goes to `/kaggle/working/harvey-cata-upgp`. Every embedded file is verified by SHA-256. Running this cell again keeps existing data and run outputs. A source mismatch stops execution so a running experiment is not silently changed.''')
add('code', f'''import base64
import hashlib
import io
import zipfile

CODE_DIR = Path("/kaggle/working/harvey-cata-upgp")
CODE_DIR.mkdir(parents=True, exist_ok=True)
PAYLOAD_SHA256 = {checksum!r}
FILE_HASHES = {manifest!r}
SOURCE_ARCHIVE = (
'''+'\n'.join('    '+repr(payload[i:i+120]) for i in range(0,len(payload),120))+'''
)
raw = base64.b64decode(SOURCE_ARCHIVE)
assert hashlib.sha256(raw).hexdigest() == PAYLOAD_SHA256, "Embedded archive checksum failed"
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert set(archive.namelist()) == set(FILE_HASHES)
    for name in archive.namelist():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe embedded path")
        content = archive.read(name)
        assert hashlib.sha256(content).hexdigest() == FILE_HASHES[name]
        target = CODE_DIR / relative
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"Existing code differs: {target}. Use a new clean working directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
print(f"Verified {len(FILE_HASHES)} source files in {CODE_DIR}")
''')
add('markdown', '''## Validate methods and optionally restore a run

The CPU correctness tests check CATA aggregation, buffers and UPGP projection. Restore only your own trusted checkpoints: they include Python objects. Set `RESTORE_RUN_DIR` to the previous complete `seed1` folder, including both `metadata.json` and `checkpoint.pt`. The runner rejects changes to seed, profile, schedule, device, code or PyTorch version.

A checkpoint is written after each warmup/reference/final epoch and each UPGP epoch. Interrupted CATA unlearning replays from the last pre-request epoch checkpoint. Completed runs are skipped. Saving local checkpoints does **not** guarantee survival after a Kaggle session is discarded: retain the working outputs as a saved notebook version or downloadable output and attach them to the next session.''')
add('code', '''import shutil
import subprocess

subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
               cwd=CODE_DIR, check=True)
RUN_DIR = CODE_DIR / "runs" / RUN_NAME / "harvey_cata_upgp" / f"seed{SEED}"
if RESTORE_RUN_DIR is not None:
    source_run = Path(RESTORE_RUN_DIR)
    for required in ("metadata.json", "checkpoint.pt"):
        if not (source_run / required).is_file():
            raise FileNotFoundError(f"Missing restore file: {source_run / required}")
    if not (RUN_DIR / "metadata.json").exists():
        if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
            raise RuntimeError("Destination is nonempty without metadata. Use a different RUN_NAME.")
        RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_run, RUN_DIR, dirs_exist_ok=True)
        print("Restored previous run into", RUN_DIR)
    else:
        print("Existing writable run found; keeping it instead of overwriting with the restore copy.")
RUN_DIR.mkdir(parents=True, exist_ok=True)
environment = {"python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "torchvision": torchvision.__version__,
               "cuda_runtime": torch.version.cuda, "device": DEVICE,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "profile": PROFILE, "protocol": PROTOCOL, "seed": SEED}
(CODE_DIR / "notebook_environment.json").write_text(json.dumps(environment, indent=2) + "\\n")
print("Run directory:", RUN_DIR)
print("Synthetic smoke data" if PROFILE == "smoke" else "Full CIFAR-10 training protocol")
''')
add('markdown', '''## Train HARVEY plus CATA plus UPGP

Run this cell once and let it finish. Output streams to the notebook and `training.log`. If you interrupt the cell, the child process is stopped; rerun to continue from its last complete checkpoint. Do not start a second copy against the same run folder.

The release profile follows the local runner's settings, including its documented differences from the paper. Low reference-model clean accuracy is intentional; final clean accuracy is measured during Stage 4. UPGP can increase or decrease accuracy/ASR; use the saved before-UPGP metrics to isolate that postprocessing step. Changing hardware introduces another source of variation when comparing against Mac runs.''')
add('code', '''import signal

command = [sys.executable, "-u", str(CODE_DIR / "experiment.py"),
           "--variant", "harvey_cata_upgp", "--profile", PROFILE,
           "--protocol", PROTOCOL, "--seed", str(SEED),
           "--batch-size", str(BATCH_SIZE), "--threads", str(CPU_THREADS),
           "--device", DEVICE, "--run-name", RUN_NAME, "--resume"]
print("Starting:", " ".join(command))
process = None
try:
    with (RUN_DIR / "training.log").open("a", buffering=1) as log_stream:
        process = subprocess.Popen(command, cwd=CODE_DIR, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training exited with status {return_code}. Inspect training.log; rerun to resume.")
except BaseException:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    raise
''')
add('markdown', '''## Results and downloadable artifacts

`result.json` records final clean ACC, ASR, before-UPGP metrics, seed and elapsed time. `split_quality.json` uses true poison IDs only for reporting, never for training or splitting. `metrics.jsonl` includes all phase logs and explicitly records a skipped UPGP step if the estimated poisoned set is empty.

This cell can be run independently after a kernel restart. It searches existing working files and attached outputs without starting training. Download the generated ZIPs from the Kaggle file browser rather than inline FileLink URLs, which may return 404.

The compact export below includes the final and pre-UPGP models, metrics and split IDs. It excludes `checkpoint.pt` to avoid duplicating the large resume artifact; **preserve the full working output directory separately for resumption**, including its checkpoint, metadata, and embedded source. Neither the compact export nor a saved cell output alone is sufficient to resume.''')
add('code', (ROOT / 'recover_results.py').read_text())
add('markdown', '''## Protocol and interpretation notes

- Final training is supervised CE as in the released CIFAR config, not MixMatch. SGD uses LR 0.1, momentum 0.9, effective weight decay 1e-4 and milestones 60/80/90 for 100 full-profile epochs.
- CATA keeps the notebook/local hybrid's fixed anchor, top-30% magnitude mask, sign aggregation, scale 0.7, AdamW LR 1e-5, weight decay 0.1 and four adaptation epochs per reference cycle. These are starting values, not tuned performance claims.
- UPGP uses CPU layer-wise SVD, 97% gradient energy and relaxation alpha 10, then five SGD epochs at LR 0.001 without momentum or decay. BatchNorm buffers are frozen during its postprocessing. Full epoch and candidate settings are in `metadata.json`.
- The final checkpoint is reported; no best-test-epoch selection occurs. Compare clean ACC and ASR together, and use multiple paired seeds. A single Mac-vs-Kaggle accuracy difference is not evidence that a method improved.
- Full runs can outlast a session. Session limits and GPU availability vary; this notebook does not assume a guaranteed runtime or persist files outside Kaggle's saved outputs.

References: [official HARVEY project](https://intellisec.de/research/harvey/), the supplied CATA and UPGP papers and existing project implementation, [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).''')
notebook = dict(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
    'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python',
               'sourceType': 'notebook', 'isGpuEnabled': True}}, nbformat=4, nbformat_minor=5)
output = ROOT / 'HARVEY_CATA_UPGP_Kaggle.ipynb'
output.write_text(json.dumps(notebook, indent=1) + '\n')
(ROOT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Created {output.name}: {len(cells)} cells, {output.stat().st_size:,} bytes, {len(manifest)} embedded files')
