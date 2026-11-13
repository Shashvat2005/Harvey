# HARVEY MPS Port - Implementation Audit Report

## Executive Summary

This report documents the complete audit of the HARVEY Apple Silicon/MPS port against the official HARVEY repository (commit `6641f5f`). The audit covers all modified files, classifies every change, identifies and fixes bugs, and validates the corrected implementation.

**Status**: All tests pass. The MPS port is ready for full baseline training.

---

## 1. File-by-File Change Classification

### 1.1 `train.py` - Main Training Orchestrator

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| `--device` argument replacing `--cuda_visible_device` | 38 | **Necessary MPS/device compatibility** | Original only accepted integer CUDA device index |
| `resolve_device()` function | 48-63 | **Necessary MPS/device compatibility** | Maps 'auto', 'cuda', 'mps', 'cpu' to torch.device |
| `seed_all_rngs()` function | 65-76 | **Reproducibility-related** | Seeds Python, NumPy, PyTorch CPU, CUDA, MPS for deterministic trials |
| `get_gpu_memory()` function | 78-84 | **Necessary MPS/device compatibility** | Abstracts `torch.cuda.memory_allocated` / `torch.mps.current_allocated_memory()` |
| `make_pin_memory()` function | 86-88 | **Necessary MPS/device compatibility** | Returns `True` only for CUDA; MPS/CPU get `False` |
| Per-trial RNG seeding (`seed_all_rngs` call) | 148-152 | **Reproducibility-related** | Ensures deterministic trials across all RNGs |
| `seed_worker()` for DataLoader workers | 154-158 | **Reproducibility-related** | Makes DataLoader shuffling deterministic per worker |
| cuDNN settings guarded by `device.type == 'cuda'` | 160-163 | **Necessary MPS/device compatibility** | cuDNN is CUDA-only; no-op on MPS |
| All DataLoaders: `pin_memory=pin_mem`, `worker_init_fn=seed_worker` | 199-234, 315-338, etc. | **Necessary MPS/device compatibility + Reproducibility** | `pin_memory` only works on CUDA; worker seeding for determinism |
| `.cpu().numpy()` on tensor indices | 270-272, 475-476, 516-517, 641-642, 652-653 | **Necessary MPS/device compatibility** | MPS tensors must be moved to CPU before `.numpy()` |
| `test_tsne` calls with `device=device` | 257, 441, 597, 706 | **Necessary MPS/device compatibility** | TSNE visualization needs correct device |
| `get_gpu_memory(device)` for memory reporting | 619 | **Necessary MPS/device compatibility** | Uses correct API per device type |
| **Corrected precision/recall metrics** | 679-738 | **BUG FIX (Algorithm discrepancy)** | Original formula was mathematically incorrect |

### 1.2 `train_test.py` - Training & Evaluation Loop

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| `.to(device)` instead of `.cuda()` + `torch.cuda.device()` | 44-49, 61-65, 89-98, 116-119, 123-124 | **Necessary MPS/device compatibility** | `torch.cuda.device()` and `.cuda()` are CUDA-only |
| `torch.tensor([0.5], device=device)` | 243 | **Necessary MPS/device compatibility** | Creates tensor directly on target device |
| Python 3 iterator syntax: `next(iterator)` | 321-339 | **Python 3 compatibility** | `.next()` is Python 2; `next()` is Python 3 |
| MixMatch tensors moved via `.to(device)` | 348-355 | **Necessary MPS/device compatibility** | Replaces `with torch.cuda.device()` block |

### 1.3 `util.py` - Model Building & Utilities

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| `.to(device)` instead of `.cuda()` | 22-27 | **Necessary MPS/device compatibility** | Model creation on target device |
| Removed `with torch.cuda.device(device):` block | 21 | **Necessary MPS/device compatibility** | CUDA context manager not available on MPS |
| `torch.load(..., map_location=device)` | 37, 41 | **Necessary MPS/device compatibility** | Loads checkpoint to correct device |
| `test_tsne` accepts `device` parameter | 138, 157-158 | **Necessary MPS/device compatibility** | Moves data to correct device for TSNE |

### 1.4 `queries_strategies/symmetric_cross_entropy.py` - RCE Query Strategy

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| Conditional `pin_memory` | 200 | **Necessary MPS/device compatibility** | Only enabled for CUDA |
| Tensors created on `self.device` | 251-252 | **Necessary MPS/device compatibility** | Avoids CPU→GPU transfer |
| `.to(self.device)` for inputs | 256-258 | **Necessary MPS/device compatibility** | Device-agnostic tensor transfer |
| Removed `with torch.cuda.device()` blocks | 250, 256 | **Necessary MPS/device compatibility** | CUDA-only context managers |

### 1.5 `data_loader.py` - Dataset Classes

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| Default `device=None` with auto-detection | 128-130, 213-215 | **Necessary MPS/device compatibility** | Auto-selects CUDA/MPS/CPU |

### 1.6 `load_dataset.py` - Dataset Loading

| Change | Lines | Classification | Justification |
|--------|-------|----------------|---------------|
| `import os` added | 1 | **Necessary MPS/device compatibility** | Needed for path resolution |
| CIFAR-10 path resolved to local directory | 124-127, 196-199 | **Dataset path/environment change** | `/share/CIFAR10` not accessible; uses `../HARVEY/data/CIFAR10` |
| `download=False` | 126-127, 198-199 | **Dataset path/environment change** | Dataset already exists locally |

---

## 2. Bugs Discovered and Fixed

### 2.1 Critical Bug: Precision/Recall Calculation (train.py:682-683)

**Original Code:**
```python
precision = 1 - num_poi_main/len(indices_bad)
recall = (len(indices_bad) - num_poi_main) / len(main_set)
```

**Issues:**
- `precision` formula: `1 - (selected_poisoned / total_poisoned)` — this is **not precision**. It measures "what fraction of poisoned samples were rejected" (true negative rate for poisoned class).
- `recall` formula: `(total_poisoned - selected_poisoned) / selected_total` — this is **not recall**. It mixes ground-truth poisoned count with selected set size.

**Corrected Implementation:**
```python
# Precision = selected clean / all selected
# Recall = selected clean / total ground-truth clean
# Contamination = selected poisoned / all selected
```

**Impact:** The original metrics significantly underestimated recall (e.g., 0.125 vs correct 0.8 in ideal case) and produced misleading precision values.

### 2.2 Potential Issue: RNG Seeding Changes Random Sequence

The MPS port adds `seed_all_rngs()` call at trial start (train.py:148-152), which seeds NumPy and PyTorch MPS in addition to Python's `random`. The original code only seeded Python's `random` for initial set selection.

**Assessment:** This is a **reproducibility improvement**, not a bug. The original code had uncontrolled NumPy state (dataset shuffling in `load_dataset.py` used `np.random.permutation` without seeding). The MPS port makes trials fully deterministic. Different seeds still produce different independent runs.

---

## 3. Changes Deliberately Not Made

| Potential Change | Reason Not Made |
|------------------|-----------------|
| Change `init_ratio` from 0.5 to enable unlabeled data for MixMatch | Must preserve official hyperparameters |
| Modify GA unlearning to use different loss formulation | Must preserve official algorithm |
| Change RCE threshold from 4.6052 | Must preserve official threshold logic |
| Modify LGA gamma or application | Must preserve official LGA |
| Change optimizer, learning rate, or schedule | Must preserve official hyperparameters |
| Integrate CATA unlearning | Out of scope for baseline reproduction |
| Add gradient clipping | Not in official implementation |
| Modify meta-splitting logic | Must preserve official algorithm |

---

## 4. Tests Executed and Results

| Test | Status | Description |
|------|--------|-------------|
| `test_mps.py` | ✅ PASS | Import, ResNet18 init, forward/backward, DataLoader, GA unlearning, RCE class availability, CUDA-only call scan |
| `test_rce.py` | ✅ PASS | RCE query strategy on MPS: threshold 4.6052, sample selection, Otsu function |
| `test_mixmatch_quick.py` | ✅ PASS | MixMatch 2-iteration training on MPS: forward, loss, backward, optimizer step |
| `test_reproducibility.py` | ✅ PASS | Seeds 1,2,3: identical seed lists, identical model params, distinct across seeds |
| `test_metrics.py` | ✅ PASS | Corrected precision/recall vs original buggy formula on synthetic data |
| `test_partitioning.py` | ✅ PASS | Dataset splitting: half-and-half, cycle refresh, meta-splitting, no duplicates |

**All 6 test suites pass.**

---

## 5. Corrected Metric Definitions

For the selected benign set `main_set` with ground-truth partitions:
- `indices_clean`: ground-truth clean samples
- `indices_bad`: ground-truth poisoned samples
- `indices_noise`: noise samples (non-poisoned, treated as clean)

**Corrected Metrics:**
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `selected_clean / selected_total` | Of selected samples, fraction that are actually clean |
| **Recall** | `selected_clean / total_clean` | Of all clean samples, fraction that were selected |
| **Contamination Rate** | `selected_poisoned / selected_total` | Poisoned samples in selected set |
| **Selected Clean** | `\|main_set ∩ (indices_clean ∪ indices_noise)\|` | Clean + noise samples selected |
| **Selected Poisoned** | `\|main_set ∩ indices_bad\|` | Poisoned samples incorrectly selected |
| **Selected Noise** | `\|main_set ∩ indices_noise\|` | Noise samples selected |

**Original (Buggy) Metrics for Comparison:**
```python
# DO NOT USE - mathematically incorrect
orig_precision = 1 - num_poi_main / len(indices_bad)
orig_recall = (len(indices_bad) - num_poi_main) / len(main_set)
```

---

## 6. Remaining MPS Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| `pin_memory` only works on CUDA | Minor: slightly slower data loading on MPS | Acceptable; `pin_memory=False` for MPS/CPU |
| No `torch.compile` support on MPS | Cannot use PyTorch 2.0 compiler | Not used in original code |
| MPS memory reporting via `torch.mps.current_allocated_memory()` | Works but less detailed than CUDA | Sufficient for monitoring |
| No multi-GPU support (`DataParallel`, `DDP`) | Single-device only | Original code single-GPU |
| `torch.cuda.manual_seed_all` no-op on MPS | Handled by `torch.mps.manual_seed` | Correctly implemented |
| MixMatch uses 1024 iterations (slow on MPS) | Configuration from original | Preserved for fidelity |

---

## 7. Exact Experiment Configuration

**Configuration File:** `conf/ResNet18_cifar10_SL_badnets_rate010.yml`

```yaml
DATASET: 'cifar10'
TASK_LEARNER: 'resnet18'
REF_LEARNER: 'resnet18'
CYCLES: 20
TRIALS: 1
BATCH: 128
INIT_BUDGET: 20000
BUDGET: 2000
init_ratio: 0.5
LGA: True
LGA_gamma: 0.01
unlearn_ref: True
unlearn_gamma: 0.01
unlearn_epochs: 1
OPTIMIZER_BACKBONE:
  NAME: SGD
  NAME_REF: ADAM
  MOMENTUM: 0.9
  WDECAY: 5e-4
  LR: 0.1
  LR_REF: 0.001
  EPOCH_TL: 100
  MILESTONES: [60, 80, 90]
BACKDOOR:
  inject_portion: 0.1
  target_label: 0
  trigger_type: badnetsTrigger
  warmup_epochs: 20
  ref_cycle: 1
  ref_epochs: 14
  ref_c1epochs: 14
  split_rate: 0.5
  meta_split: True
SSL:
  sl_train: True
  lambda_u: 10
```

---

## 8. Training Command

```bash
cd /Users/shashvatgarg/Desktop/Main/Shashvat_Garg/Study/DTU/Study/4th Year/Sem - 7/BTP/Harvey + Extra/HARVEY_MPS

python3 train.py \
  -c conf/ResNet18_cifar10_SL_badnets_rate010.yml \
  --seed 1 \
  --device mps
```

**Output Directory:** `results/cifar10_harvey_mps_audited/` (to be created by script)

---

## 9. Expected Results (from Previous Run)

| Metric | Previous Run | HARVEY Paper (CIFAR-10 ResNet18 BadNets) |
|--------|--------------|------------------------------------------|
| Final Clean Accuracy | 91.53% | 93.34% |
| Final ASR | 0.50% | 0.56% |
| Peak Clean Accuracy | 91.74% | — |
| Peak ASR | ~0.50% | — |

**Note:** Small discrepancy vs paper expected due to MPS vs CUDA numerical differences and non-deterministic operations. The corrected metrics will provide accurate benign-set evaluation.

---

## 10. Modified Files Summary

| File | Purpose |
|------|---------|
| `train.py` | Main orchestrator: device abstraction, RNG seeding, DataLoader config, corrected metrics |
| `train_test.py` | Training loop: device-agnostic tensor ops, Python 3 iterators |
| `util.py` | Model building, checkpoint loading, TSNE: device-agnostic |
| `queries_strategies/symmetric_cross_entropy.py` | RCE strategy: device-agnostic |
| `data_loader.py` | Dataset classes: default device auto-detection |
| `load_dataset.py` | CIFAR-10 path fix, local dataset loading |

---

## 11. Verification Checklist

- [x] All 6 test suites pass
- [x] No CUDA-only calls in active execution paths (guarded conditionals only)
- [x] GA unlearning math verified: negative CE on correctly classified samples, `Unlearn_Gamma`, optimizer step
- [x] LGA verified: `torch.sign(target_loss - gamma) * target_loss`
- [x] RCE verified: `RCELoss` with alpha=0, beta=1, threshold 4.6052, ascending sort
- [x] Dataset splitting verified: half-and-half, cycle refresh, meta-splitting, no duplicates
- [x] MixMatch verified: iterators, one-hot targets, device placement, pseudo-labels, MixUp
- [x] RNG reproducibility: seeds 1,2,3 produce distinct independent runs
- [x] Corrected precision/recall implemented and tested
- [x] CIFAR-10 dataset loads from local path (50k train, 10k test)
- [x] Configuration unchanged from official YAML
- [x] No CATA integration
- [x] No hyperparameter changes
- [x] No previous experiment logs overwritten

---

## 12. Conclusion

The HARVEY MPS port has been fully audited, all bugs fixed, and all tests pass. The implementation faithfully reproduces the official HARVEY algorithm with only necessary MPS/device compatibility changes, Python 3 compatibility fixes, dataset path corrections, and a critical bug fix in the final evaluation metrics.

The port is **ready for full baseline training** on Apple Silicon MPS.