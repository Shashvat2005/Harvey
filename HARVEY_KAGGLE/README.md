# HARVEY plus CATA plus UPGP Kaggle notebook

Upload **HARVEY_CATA_UPGP_Kaggle.ipynb** to Kaggle. Select a GPU accelerator and enable Internet in notebook settings, then Run All. The notebook embeds its six required source/test/trigger files; no GitHub clone, local Mac folder, or separately uploaded script is required.

The first code cell defaults to `PROFILE = "full"`, seed 1, batch 128 and the release protocol (14 reference epochs per cycle). It matches the local HARVEY+CATA+UPGP learning settings. For an execution check first, use `PROFILE = "smoke"`, then change back to full and Run All. Smoke uses random synthetic images and must not be interpreted as research accuracy. One GPU is used even if two are available. The notebook retains Kaggle's installed torch/torchvision pair and does not install the Mac-pinned versions.

If GPU memory is insufficient, choose batch 64 in a new run. `RUN_NAME` includes profile, protocol and batch by default; do not change settings or source code and try to continue the same checkpoint. Full training can outlast a Kaggle session; there is no completion-time guarantee.

## Outputs

Default full-run output directory:

```text
/kaggle/working/harvey-cata-upgp/runs/kaggle-full-release-b128/harvey_cata_upgp/seed1/
```

It contains final metrics in `result.json`, detailed curves in `metrics.jsonl`, terminal output in `training.log`, split quality/IDs, models before and after UPGP, metadata and a resumable `checkpoint.pt`. True poison IDs are used only for reporting split quality.

The last cell creates a downloadable compact ZIP containing results, split information and model weights. It intentionally excludes the large checkpoint. To resume in another Kaggle session, preserve the **full working output folder**, including checkpoint and metadata, through Kaggle's saved outputs or a download/upload. Attach those previous outputs and set `RESTORE_RUN_DIR` to their full `seed1` directory. Code and environment checks reject incompatible resumes. Use only your own trusted checkpoints.

The notebook has CPU validation support (`REQUIRE_CUDA=False`) for testing outside Kaggle; leave it True for your GPU training. `CIFAR10_ROOT` optionally points to an attached cache containing `cifar-10-batches-py/` for offline use. Otherwise CIFAR-10 downloads automatically with torchvision.

## Method and limits

This is the same hybrid as the local third run: CATA replaces reference-model benign-set unlearning; the UPGP gradient-projection component postprocesses the final classifier using HARVEY's estimated benign and poisoned sets. It does not run UPGP's perturbation detector or reproduce the complete UPGP method. If the estimated poisoned set is empty, UPGP skips explicitly. Compare clean ACC and ASR together; before-UPGP metrics isolate that step on the same trained classifier. Mac versus Kaggle differences can reflect hardware and library differences, so they are not a controlled method comparison.

`source/` contains the CUDA-adapted source for review. `build_notebook.py` regenerates the self-contained notebook and `source_manifest.json`. The builder verifies file paths and embeds SHA-256 hashes. Existing Mac runners and previous notebooks were not changed. This folder has not been pushed as part of notebook creation.

## Validation

On the Mac, all code cells were syntax checked and executed in order with a temporary working path, CPU, and synthetic smoke profile. Extraction/checksums, all six method tests, subprocess logging, full ResNet18/CATA training orchestration, checkpoint and final artifact saving, compact export, and completed-run resume passed. Synthetic data correctly produced no suspected poisons and skipped UPGP; its projection/loss path was exercised separately by the method tests. No Kaggle session, CUDA kernel execution, full CIFAR-10 experiment, or measured accuracy improvement has been verified. Details are in `local_validation.json`.

## Recovery after undefined variables or broken download links

The export cell was revised to work independently of earlier notebook variables. It searches `/kaggle/working` and attached `/kaggle/input` outputs for completed HARVEY+CATA+UPGP results, prints them, and writes ZIPs into `/kaggle/working/harvey-results`. Download those files through Kaggle's file browser or a saved notebook version's Output tab; absolute filesystem FileLink URLs are not used. This cell never starts training. If the completed files are absent from both locations, inspect the earlier saved notebook version/output; a restarted kernel alone does not prove either loss or persistence of its filesystem.

The recovery helper was checked with a temporary completed-result fixture and with absent files. Both checks passed, including ZIP contents. No research metrics were created by these checks. Only the notebook export cell and its explanation changed; embedded training sources and existing checkpoints were not modified.
