# HARVEY experiments on Apple Silicon

Everything created for this work is inside `HARVEY_COMPARE`. Existing source, MPS code, notebooks, results and papers are unchanged. The runner reads an existing CIFAR-10 cache if available; downloads and every new output go here.

Start with HARVEY alone. The published ResNet18 / CIFAR-10 / BadNets 10% poisoning reference is **93.34% clean ACC and 0.56% ASR**, Table 2. This is a reproduction attempt, not a promise of 93%. Low reference-model accuracy during Stage 2 is expected: that model deliberately learns the backdoor. Judge the Stage-4 final model.

## Setup and baseline first

From the repository root:

```bash
cd HARVEY_COMPARE
bash setup.sh
.venv/bin/python -m unittest discover -s tests -v
# A few seconds/minutes on tiny SYNTHETIC data: checks plumbing only.
bash run_all.sh --profile smoke --run-name smoke
# Real CIFAR-10 speed measurement before committing to the long experiment.
.venv/bin/python benchmark.py --variant harvey
# Reproduction attempt: real CIFAR-10, 20 warmup + up to 20 reference cycles + 100 final epochs.
caffeinate -i bash run_harvey.sh --run-name release-full
```

Keep the Mac plugged in with the lid open. `caffeinate -i` prevents idle sleep; it does not prevent lid-close sleep. Do not assume that all three full runs fit in one night. The benchmark estimates only final-training compute; reference learning, CATA's four adaptation epochs per cycle, evaluation and checkpoints add time. Each variant runs in its own process, sequentially. Default batch size 128 and workers 0 reduce multiprocessing and memory complications on a 16 GB unified-memory machine. If memory pressure is high, use batch 64 for **all** variants in a new run group; batch size changes the protocol and results.

After checking the baseline's `runs/release-full/harvey/seed1/result.json`, run the extensions:

```bash
caffeinate -i bash run_harvey_cata.sh --run-name release-full
caffeinate -i bash run_harvey_cata_upgp.sh --run-name release-full
.venv/bin/python compare.py --run-name release-full
```

Or run/continue the full sequence automatically, skipping completed runs:

```bash
caffeinate -i bash run_all.sh --run-name release-full --resume
# Stronger paired comparison (expensive):
SEEDS="1 2 3" caffeinate -i bash run_all.sh --run-name three-seeds --resume
```

`--resume` also starts a missing run. An interrupted run resumes at its last complete training epoch/reference cycle. An interruption inside CATA or GA unlearning replays that unlearning request from the pre-request checkpoint, with saved random states. UPGP resumes at an epoch boundary. Settings, code hashes, PyTorch and device must match. Do not run two processes into the same variant/seed directory. Do not edit the runner during a resumable experiment.

Each variant also accepts `--device cpu`, `--batch-size 64`, `--seed 2`, and `--protocol paper-text`. Use the same arguments for all three. Shell scripts use `.venv/bin/python`; an existing compatible environment can be selected with `PYTHON=/absolute/path/to/python`.

## What is reproduced, and known discrepancies

Default `--protocol release` follows the supplied `HARVEY_source/conf/ResNet18_cifar10_SL_badnets_rate010.yml` and training behavior:

- Original CIFAR ResNet18 (vendored unchanged), 10% uniformly selected BadNets poison, target class 0, original RGB trigger pixels, all-to-one labels. Target-class training samples remain eligible for poisoning. ASR excludes original target-class test samples (9,000 examples); clean ACC uses all 10,000 test images.
- CIFAR normalization, crop/padding 4, rotation 10 degrees, horizontal flip for final training; no augmentation for reference learning/splitting, matching the source.
- 20 full-data warmup epochs, Adam LR 0.001; fresh reference model for the first isolation cycle. Up to 20 isolation cycles, 14 poison-learning epochs per cycle, LGA gamma 0.01. Correctly-predicted estimated benign examples receive one epoch of negative CE scaled by 0.01, with the same Adam optimizer as reference learning, for HARVEY alone.
- Reverse CE with one-hot clamp 1e-4; fixed threshold 4.6052 (the released query uses this, **not Otsu**, despite helper code). Initial low/high-loss half split, poisoned-subset retention max(0.5 × estimated poison count, 5% of training data), 1% per-class retention minimum, stop when estimated benign count stops changing, first-cycle meta splitting.
- Fresh final classifier, supervised CE for 100 epochs, SGD LR 0.1, momentum 0.9, effective weight decay **1e-4** (hard-coded in source, despite YAML saying 5e-4); milestones 60/80/90. No MixMatch because the released CIFAR config selects `SSL.sl_train: True`. Evaluate the last epoch, never choose a checkpoint from test accuracy.

The paper text says **10** poison-learning epochs, while released config says **14**. `--protocol paper-text` changes only that value to 10 and is separately labelled; it cannot resolve all paper/code differences. The paper text's epsilon statement and displayed threshold also differ; the implemented threshold/clamp follow released code. Do not mix protocols in a comparison.

This is a clean reimplementation of the supported CIFAR/BadNets path, not a byte-for-byte port of the CUDA orchestration. It uses explicit seeded poisoning, lazy image views, independent deterministic model initialization, CPU/MPS device transfers, finite-loss checks, correct split-quality metrics, and complete resumable state. The original uses generated model seeds and persists poison IDs in a shared trigger directory; those can make repeatability and cross-run comparisons unclear. Here model initialization uses the requested seed directly, and poison IDs use a dedicated NumPy generator, shared across methods. Stochastic trajectories and old CUDA results will therefore differ. Modern PyTorch/MPS also differs from the original torch 1.12/CUDA stack. MPS reproducibility is not guaranteed bitwise across OS/hardware/library versions.

If the full baseline falls short of the paper, examine final ACC **and ASR**, split contamination, data/trigger checks, and learning curves before tuning extensions. Try the paper-text protocol as a separately named run. Do not shorten training, substitute architectures, select the best test epoch, or adjust only an extension's final schedule and then call the resulting comparison a paper reproduction.

## CATA integration

`cata.py` is copied from the notebooks' embedded source. It implements fixed-anchor forget-set adaptation, negated task vectors, global top-30% magnitude masking (including threshold ties), magnitude-weighted sign election, and the disjoint mean. Defaults are the notebook's AdamW LR 1e-5, weight decay 0.1, four epochs, scale 0.7. These values came from the supplied CATA paper's VLM setting and have not been tuned for this classifier.

CATA replaces only the reference model's estimated-benign-set unlearning. Its anchor is the first poison-trained reference model before unlearning. It applies the merged correction to each current poison-trained base and restores that base before the next poison-learning phase. This follows the notebooks' **hybrid** formula, `reference_t = poison_base_t + 0.7 * aggregate_t`, rather than standalone CATA's fixed-anchor application formula. Historical statistics occupy constant memory in the number of cycles. Estimated benign requests may overlap. No actual poison IDs inform training or splitting.

## UPGP integration

The third variant runs the same HARVEY+CATA training, then adds a final-model UPGP **gradient-projection unlearning component**. HARVEY meta splitting supplies estimated benign and poisoned sets, so UPGP's perturbation detector is intentionally replaced. This is **not a reproduction of the full UPGP paper** or its trusted-clean-data setting.

On up to 512 estimated benign training images, collect layer gradients over up to four batches (all samples when batch size is 128). Stack gradient matrices reshaped to `(out_channels, remaining_dimensions)`, use CPU SVD, retain 97% squared singular-value energy, and use Eq.18 relaxation alpha=10. Apply Eq.19 without allocating a dense identity/projector. Bias and BatchNorm vectors use the identity projection. Basis computation and postprocessing freeze BN buffers in eval mode. Minimize clean CE plus `-log(1-p_observed_label)` on estimated poisoned examples, then project the combined gradient per layer. SGD has no momentum or weight decay so those do not introduce unprojected updates. Defaults are five epochs, LR 0.001; these hybrid hyperparameters are exploratory. A relaxed projector does **not** guarantee exact orthogonality. No projection-rank cap silently discards required energy.

SVD runs on CPU because it is more portable than depending on MPS SVD support. If no suspected poisons remain, UPGP is explicitly skipped and reported. If estimated benign samples contain poison or estimated poisons are clean, UPGP may hurt performance. A gain is a hypothesis, not an implemented guarantee.

## Outputs and interpretation

`runs/<run-name>/<variant>/seed<N>/` contains metadata and source hashes, poison IDs, JSONL learning curves, atomic epoch checkpoints with optimizer/scheduler/RNG/CATA state, final split IDs, split-quality metrics, pre-UPGP and final model weights, and final metrics. Checkpoints are overwritten rather than accumulating every epoch. Expect hundreds of MB per active checkpoint and roughly GB-scale retained data across variants. The terminal log prints every completed epoch; redirect it with `> training.log 2>&1` if desired.

`compare.py` writes `comparison.csv` and `comparison.md` beside the variants. It excludes synthetic smoke results, reports missing runs, refuses to compute paired deltas on unmatched settings or poison IDs, and reports final ACC/ASR, runtime, and changes relative to same-seed HARVEY. With multiple matched seeds it reports mean and sample SD. Use at least three paired seeds before interpreting a small gain; these reports do not assert statistical significance. Inspect `before_upgp` to isolate the effect of UPGP on the same trained CATA classifier.

## Audit and references

`audit/` contains extracted paper text and the notebooks' embedded source for inspection; original inputs are untouched. All three notebook files have empty saved output lists, so Kaggle accuracies cannot be recovered from their outputs here. Existing `HARVEY_MPS/results` is preserved and is not mixed into the new matched experiments.

- HARVEY paper: `../Harvey.pdf`, Table 2; [official project](https://intellisec.de/research/harvey/), [source](https://github.com/intellisec/harvey).
- CATA: `../CATA.pdf`, Section 4 Eqs.6-16 / Algorithm 1; notebook adaptation described above.
- UPGP: `../UPGP.pdf`, Section 3.3 Eqs.13 and 17-19 / Algorithm 2; projection component adapted above.
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

Test results and real-hardware smoke/benchmark observations are recorded in `VALIDATION.md`. Smoke metrics are synthetic execution checks and must never be reported as research accuracy.
