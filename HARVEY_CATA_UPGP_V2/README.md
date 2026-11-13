# HARVEY plus CATA plus UPGP V2 on Kaggle

Upload **HARVEY_CATA_UPGP_V2_Kaggle.ipynb**, select a CUDA GPU and enable Internet, then use **Save Version > Save & Run All**. This runs the notebook as a saved version so a successful run retains its output files. It embeds all nine source/test/trigger files and uses one GPU. The original notebook and its recovered results are unchanged.

This version targets the observed failures: the earlier CATA split rejected 15,105 clean images, and broad UPGP unlearning reduced final ACC from 86.04% to 51.08%. The changes below are exploratory responses to those failures, not promises of beating HARVEY.

## What changed

- **A fixed training validation reserve.** Reserve 5% of training IDs stratified by observed labels before warmup. They are excluded from reference fitting, reference unlearning, final fitting and refinement fitting. Poison truth is used only for later audits. The reference guard, final validation groups and candidate selection use estimated labels/groups, not a trusted clean external dataset.
- **Guarded CATA proposals.** Ordinary HARVEY GA is the per-cycle control. CATA uses two adaptation epochs instead of four, AdamW LR 1e-5 with weight decay zero, the original 30% mask, and scales 0.1 and 0.3. All task vectors still use a fixed anchor. Proposals must maintain estimated benign retention versus GA, preserve fixed suspicious-probe low RCE within 0.1, and not worsen clean-minus-suspicious RCE separation. Probes are selected from the warmup reserve without true poison labels. If insufficient probes or no proposal qualifies, use GA and count that cycle honestly. A CATA-selected correction is removed before the next poison-learning phase; a GA-selected update is retained, as in ordinary HARVEY. History contains all proposed CATA requests, including those not accepted, and the first-cycle chosen model is used for meta splitting.
- **Semantic filtering after classifier training.** Only target-labelled, first/last-reference low-RCE rejected images are considered. A frozen base must give consistent, confident non-target semantics in two weak views (flip and reflected translation). Fit and validation groups are filtered separately. It does not inspect the known trigger mask, repaired originals or poison truth. Weak-view agreement is a fallible proxy and assumes a single target and semantic invariance.
- **A separate utility stage.** Class-balanced likely-clean CE plus frozen-base KL, fixed BN buffers, LR 0.001/0.005, at most two epochs per arm. Its selection is reported separately from CATA/UPGP. By default, rejected FITTING images whose original and both weak views agree with their observed label at >=0.95 may re-enter this utility stage. They are label-consistent candidates, not verified clean; residual triggers are possible. This never includes held-out validation images. Set `RECOVER_CLEAN=False` to ablate this extra data-reuse refinement. Its count is reported, so improvements must not be attributed solely to CATA or UPGP. It may be skipped if no adequate suspicious validation group exists.
- **Conservative UPGP.** Compute layer row-space bases on balanced likely-clean fitting examples, 99% squared singular-value energy, relaxation alpha 10, CPU SVD. Apply projection only to the forgetting gradient, not the clean retention gradient. The forget objective combines a small negative-target-probability term and CE toward the teacher's non-target semantic pseudo label. Remove residual first-order conflict with the current clean gradient; add clean CE/KL separately. Use no momentum/decay, LR 1e-5/1e-4 and at most two epochs per independent arm, at most 1,024 suspicious samples per epoch. Clip gradient norms and bound global parameter displacement to 0.2% of the utility anchor norm. BN buffers remain fixed throughout.
- **Controls and validation fallback.** An update cannot reduce overall validation ACC or increase suspicious target-prediction rate relative to the raw base. Target-class validation loss is limited to 1 percentage point; mean suspicious target probability may increase by at most 0.001. Rank permissible candidates by accuracy, then response/confidence. At least 32 filtered suspicious fitting/validation samples and 20 target validation examples are required. If nothing improves, retain the previous control. Validation proxies do not guarantee improvement on true test data.

These modify the previous hybrid substantially and must be reported as **V2 with guarded CATA and semantic-label UPGP**. It replaces UPGP's independent perturbation detector with HARVEY/teacher estimates; it is not the complete UPGP paper. If all reference cycles use GA, do not attribute the result to CATA. If `selected_edit.kind` is `base` or `clean_polish`, do not claim a successful UPGP edit. The teacher pseudo-label objective and separate clean-gradient update are new hybrid design choices, not claims about the original UPGP algorithm.

The final classifier remains ResNet18, original CIFAR transforms, BadNets 10%, target 0, 100 supervised epochs, SGD LR 0.1/momentum 0.9/decay 1e-4 with milestones 60/80/90. Release reference training uses 14 epochs; paper-text uses 10. The full profile never silently shortens training, changes batch size or enables AMP. There is no prior Kaggle weight reuse: the old text log cannot reconstruct weights.

## Notebook settings

Defaults: full profile, release protocol, seed 1, batch 128, one GPU. Use smoke first only for execution checks; smoke uses random images and its accuracy is not a research result. The first code cell exposes the method, seed, batch and LR grids. For a fresh matched control choose `METHOD = "harvey_v2_control"`; use the same data, seed and protocol. The control runs ordinary HARVEY with the V2 validation reservation and skips all refinement.

Full training may exceed a session. Save the FULL working output, not only visible cell outputs. Resume requires the compatible `metadata.json` and `checkpoint.pt`; attach the earlier output folder and set `RESTORE_RUN_DIR` to its complete method/seed folder. Interrupted refinement restarts that bounded stage from the saved base. Do not edit source/environment/settings and then reuse the same run name.

## Results and downloads

```text
/kaggle/working/harvey-cata-upgp-v2-saved/runs/
  v2-full-release-b128-recover1-saved/harvey_cata_upgp_v2/seed1/
```

`result.json` reports selected test metrics, base/utility controls, selected edit, GA/CATA cycle counts and elapsed time. `selection.json` stores the validation decisions. Models, reference scores, group IDs and the checkpoint are separate files. All true test metrics are evaluated after selection, not used to pick an epoch.

The training cell now automatically creates a **verified full ZIP**, not only a small report export. It backs up after the first/phase-changing checkpoints, every 15 minutes when a new checkpoint event arrives, and in a `finally` block on success or Python-level interruption/error. Every archived file is checked against a SHA-256 manifest before the previous ZIP is replaced. A sudden VM-wide termination may bypass the final backup.

Full ZIPs and their `backup-status.json` files are under:

```text
/kaggle/working/harvey-v2-full-backups/
```

They include all CURRENT run files and actual source: metadata, notebook settings/environment, poison/split/group IDs, reference scores, complete logs, candidate tables, latest resume checkpoint, warmup/reference/classifier end states, first/last reference models, base/utility/final models and every evaluated refinement candidate model. Temporary writes, downloaded CIFAR caches and obsolete historical epoch checkpoints are excluded to avoid exhausting the output quota. The stage-end states retain optimizer/scheduler/RNG/CATA information appropriate to those stages; the latest checkpoint resumes the same pipeline, with bounded refinement replay when interrupted there. This does not mean every historical minibatch/epoch optimizer state is retained.

**Kaggle persistence still requires a saved version or a download.** Use Save Version > Save & Run All before a new run. If you already used ordinary interactive Run All, use Quick Save > Advanced settings > save output files before ending the session. Then open the saved version's Output tab and download the FULL.zip. Check its companion status JSON for `verified: true` and `complete: true`. Saving the draft code alone does not guarantee output retention. A local archive cannot survive a discarded VM unless retained through Kaggle or downloaded.

The final export cell is independent of notebook variables. It finds completed results, prints JSON and retries the full export from the co-located source. It uses file-browser paths rather than unreliable filesystem FileLink URLs. To restore, extract the ZIP and point RESTORE_RUN_DIR at its `code/runs/<run-name>/<method>/seedN` folder; keep the exact source/settings/library/device versions. Only restore your own trusted checkpoints.

This saving revision intentionally changes the Kaggle artifact-writing code and source hashes. Use the updated notebook in its new default WORK_DIR/RUN_NAME for a NEW run. Do not overwrite an active older run or expect an older V2 checkpoint to pass the new source-hash check. The Mac V2 code/results were not changed.

Compare against your own baseline and report all changes in validation access, training pool, hardware, policy and seeds. Because earlier outcomes informed this development, V2 is exploratory until settings are frozen and verified on further paired seeds. A paper number or one favorable seed does not establish reliable improvement. See `VALIDATION.md` for execution evidence and limitations.


## Why UPGP is short

The old experiment trained the final classifier for 100 epochs and then performed five additional UPGP epochs. Five was an experimental postprocessing default, not the full training budget. In that run, the first UPGP epoch already reduced clean ACC from 86.04% to 59.76%; longer unlearning did not fix the underlying sample/objective problem.

The new notebook exposes `UPGP_EPOCHS = 2` per low-LR arm. Both arms restart from the same utility-stage anchor, so this is not a single chained four-epoch update. Validation may retain the anchor or select epoch 1 or 2. Change the knob only in a newly named experiment; do not pick its value from test-set accuracy.
