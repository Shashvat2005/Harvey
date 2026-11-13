# Fourth experiment HARVEY with Option B weight subtraction

This implements Extension 4 / Option B from the supplied `HARVEY_Extensions_Stepwise_Examples.docx`. It is a separate exploratory experiment, **HARVEY + Option B**, without CATA or UPGP. All original files and the first three runners stay unchanged. The training plan and review are in `TRAINING_PLAN.md`.

## Run

From the repository root, after setting up the existing `HARVEY_COMPARE` environment:

```bash
cd HARVEY_OPTION_B
# Uses ../HARVEY_COMPARE/.venv/bin/python if available.
caffeinate -i bash run_option_b.sh --run-name option-b-full --seed 1 --resume
```

If you have no existing environment, run `bash setup.sh` inside this folder first. You can also select an interpreter with `PYTHON=/absolute/path/to/python`. Python/PyTorch requirements match the first three experiments.

Synthetic smoke check before the full run:

```bash
bash run_option_b.sh --profile smoke --run-name option-b-smoke --resume
```

Three paired seeds, executed one at a time:

```bash
for seed in 1 2 3; do
  caffeinate -i bash run_option_b.sh --run-name option-b-three-seeds --seed "$seed" --resume
done
```

Use the same seeds, batch size, architecture, attack, and release/paper-text protocol as the first three runs. Run one training process at a time on the 16 GB Mac. The original `run_all.sh` still runs exactly the original three; this fourth run is launched separately. No full training has been run or accuracy improvement established.

## Default configuration

- CIFAR-10 / original CIFAR ResNet18 / BadNets / target 0 / 10% poison / seed 1 / MPS when available / batch 128 / workers 0.
- Original HARVEY detection with GA, 20 warmup epochs, up to 20 isolation cycles at 14 reference epochs, first-cycle meta splitting. `--protocol paper-text` uses 10 reference epochs. Same explicit model seed, poison IDs, transforms, threshold and optimizer settings as the first runner.
- High-confidence suspicious samples must be in HARVEY's final rejected pool, have observed label 0, and have RCE <= 0.5 under **both** first and last reference models. This is agreement, not independent proof of poisoning. Threshold is fixed in advance.
- Reserve 5% of HARVEY's likely-clean pool per represented class, with at least one fitting sample retained. Reserve 20% of high-confidence suspicious samples. Holdouts are excluded from base fitting, adaptation, BN recalibration and re-exposure. Reference detection necessarily saw the full original training pool before these holdouts were defined.
- Train a fresh base model on all training images except the entire high-confidence suspicious group and the likely-clean validation group. This deliberately retains uncertain images. Base schedule: 100 epochs, SGD LR 0.1, momentum 0.9, decay 1e-4, milestones 60/80/90.
- Copy the trained base; fine-tune on up to 512 suspicious fitting samples and 512 likely-clean fitting samples in exactly equal batches. One epoch, SGD LR 1e-4, no momentum or weight decay. BN buffers stay frozen. Delta = fine-tuned parameters minus base parameters.
- Candidates: base minus alpha × delta, alpha in {0, 0.1, 0.25, 0.5, 0.75, 1}. Always start from the same base. Never subtract BN buffers. Recalibrate each candidate using the same up-to-4096 likely-clean fitting images.
- Require at least 32 suspicious fitting and 32 suspicious validation examples, plus at least 10 target-class validation examples. Otherwise keep the raw base. These are minimum execution guards, not evidence that the groups are statistically large enough.
- A nonzero candidate must reduce suspicious validation target-prediction rate by at least 1 percentage point relative to **both** raw and alpha-zero recalibrated controls, lose no more than 1 point overall and 3 points on target-class validation accuracy relative to either control. Rank feasible candidates by lowest response, then lowest mean target confidence, then smallest alpha. If none qualifies, keep the **raw unmodified base**; do not force a subtraction.
- Freeze the decision before clean/triggered test evaluation. Save base and selected model ACC, ASR, and genuine clean target-class test accuracy.
- Isolated post-selection stress audit: one balanced epoch of re-exposure to up to 128 estimated suspicious and 128 likely-clean fitting examples, LR 1e-4, from each of base and selected weights. Report post-exposure test metrics. This uses estimated suspicious samples, not verified poison truth. It is a weaker proxy than the document's proposed controlled verified-poison reactivation experiment. Stress weights never replace `final_model.pt`.

Target-class direction projection is optional in the document and is **not enabled** here: no independently verified clean target-class dataset has been supplied. Validation guards are not a substitute for that direction estimate. Feature voting, repair, CATA and UPGP are also excluded from this first Option B test.

The likely-clean validation labels and suspicious response are imperfect proxies. Clean-looking target-class examples may be mislabeled poisons; harmless target-class images may appear suspicious. A successful validation edit may still fail genuine test ASR or clean accuracy. Report those failures. A raw-base fallback is a valid experimental outcome, not evidence that Option B worked.

## Outputs and comparison

```text
HARVEY_OPTION_B/runs/option-b-full/harvey_option_b/seed1/
```

- `metadata.json`, `metrics.jsonl`, `checkpoint.pt`: settings, source hashes, progress and recovery state.
- `first_reference.pt`, `last_reference.pt`, `split.npz`: reference models, meta split, first/last RCE arrays.
- `option_b_groups.npz`: exact fitting, validation and excluded IDs, fixed before base training.
- `split_quality.json`, `poison_ids.npy`: detection audit only; poison truth never selects groups, strengths or updates.
- `base_model.pt`, `task_vector.pt` (when eligible), `final_model.pt`: base, aligned task vector and selected/fallback weights.
- `selection.json`: every candidate's validation measurements, chosen alpha and reason. No test-set selection.
- `result.json`: final/base ACC, ASR, target accuracy, paired changes relative to the same base and total runtime.
- `reactivation.json`: isolated post-selection stress audit.

Resume restores the last complete warmup/reference/base epoch and RNG state. If interrupted inside adaptation/search/stress, it restarts that entire postprocessing stage from the saved base. This makes its stage recovery simple and consistent; do not edit code/settings mid-run. Completed results are skipped with `--resume`.

Combine with the original results after all runs finish:

```bash
# Reuse the same Python environment as training.
../HARVEY_COMPARE/.venv/bin/python compare_four.py \
  --first-three-run release-full --fourth-run option-b-full
```

Or use `.venv/bin/python` if you set up this folder separately. Reports go to `HARVEY_OPTION_B/reports/release-full--option-b-full/`. The four-method table is descriptive: Option B changes the training pool and adds validation holdouts. Its raw base is the paired control for subtraction. Cross-method differences include those data-policy changes. Multiple paired seeds are necessary before interpreting small changes; no significance test is claimed.

`VALIDATION.md` records execution checks and remaining limitations. The fourth experiment is ready to run, not a demonstrated reproduction or improvement.
