# HARVEY MPS Output Safety Report

## Executive Summary

This report documents the implementation of comprehensive output safety mechanisms for the HARVEY MPS port. All training outputs are now directed to unique, timestamped run directories with collision detection, overwrite protection, and full experiment traceability.

**Status**: All tests pass. The output safety system is ready for production use.

---

## 1. Files Modified for Output Safety

| File | Purpose |
|------|---------|
| `train.py` | Unique run directory creation, config copying, metadata logging, run_dir propagation |
| `train_test.py` | Datamap logging redirected to run directory |
| `util.py` | TSNE outputs redirected to run directory, query_strategy accepts run_dir |
| `queries_strategies/symmetric_cross_entropy.py` | RCE histogram outputs redirected to run directory |
| `train.py` (existing changes) | Model checkpoints, state pickles, training logs to run directory |

---

## 2. Complete Output Pipeline Audit

### 2.1 Output Locations Identified

| Component | Original Path | New Path (under run_dir) |
|-----------|---------------|--------------------------|
| Training log | `results_path/ex_setting.txt` | `run_dir/ex_setting.txt` |
| Model checkpoint | `results_path/ex_setting_latest_target_model.pt` | `run_dir/ex_setting_latest_target_model.pt` |
| State pickle | `results_path/ex_setting_latest_state.pkl` | `run_dir/ex_setting_latest_state.pkl` |
| Config copy | — | `run_dir/config_used.yml` |
| Run metadata | — | `run_dir/run_metadata.json` |
| Datamap CSV | `results_folder/METHOD/DATASET_full.csv` | `run_dir/datamap/DATASET_full.csv` |
| TSNE activations | `outputs/trigger/DATASET/ref_stg_tSNE/` | `run_dir/tsne/` |
| TSNE plots (PDF/PNG) | `outputs/trigger/DATASET/ref_stg_tSNE/` | `run_dir/tsne/` |
| RCE histograms (NPY/PNG) | `outputs/trigger/DATASET/ref_stg_alpha_beta/trainer/` | `run_dir/rce_histograms/` |
| RCE probs/thresholds | `outputs/trigger/DATASET/...` | `run_dir/rce_histograms/` |
| State pickle (final) | `results_path/ex_setting_latest_state.pkl` | `run_dir/ex_setting_latest_state.pkl` |
| Final model checkpoint | `results_path/ex_setting_latest_target_model.pt` | `run_dir/ex_setting_latest_target_model.pt` |

### 2.2 Overwrite Risks Identified and Mitigated

| Risk | Original Behavior | Mitigation |
|------|-------------------|------------|
| Same config re-run overwrites logs | `open(..., 'w')` on fixed path | Unique run directory per execution |
| Checkpoint overwrite | `torch.save()` to fixed filename | Unique run directory per execution |
| TSNE/RCE plots overwrite | `plt.savefig()` to fixed path | Subdirectory under unique run_dir |
| Datamap CSV overwrite | `df.to_csv()` to fixed path | Unique run_dir subdirectory |
| Config not preserved | No config saved | `config_used.yml` copied to run_dir |
| No experiment metadata | No tracking | `run_metadata.json` with full traceability |

---

## 3. Unique Run Directory Generation

### 3.1 Algorithm

```python
def create_unique_run_dir(base_results_folder, method_name, ref_method, seed):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_seed{seed}"
    run_dir = os.path.join(base_results_folder, f"{method_name}-{ref_method}", run_name)
    
    # Collision handling
    counter = 0
    original_run_dir = run_dir
    while os.path.exists(run_dir):
        counter += 1
        run_dir = f"{original_run_dir}_{counter}"
    
    os.makedirs(run_dir, exist_ok=False)
    return run_dir
```

### 3.2 Naming Convention

```
results/
  cifar10/
    RCE-RCE/
      run_20261109_180000_seed1/      # First run, seed 1
      run_20261109_180000_seed1_1/    # Collision: second run same timestamp/seed
      run_20261110_090000_seed1/      # Next day, seed 1
      run_20261110_150000_seed2/      # Different seed
```

### 3.3 Collision Handling

- Timestamp resolution: seconds (YYYYMMDD_HHMMSS)
- If directory exists, append `_1`, `_2`, etc.
- Uses `os.makedirs(..., exist_ok=False)` for atomic creation
- Never reuses an existing directory

---

## 4. Overwrite Protection Mechanisms

### 4.1 Directory-Level Protection
- Each run gets a **unique** directory
- `os.makedirs(..., exist_ok=False)` fails if directory exists
- Collision detection with numeric suffixes

### 4.2 File-Level Awareness
- All output functions use `run_dir` as base
- No hardcoded output paths outside `run_dir`
- Existing files in other run directories never touched

### 4.3 Checkpoint Safety
- Model checkpoints saved to `run_dir/ex_setting_latest_target_model.pt`
- State pickles saved to `run_dir/ex_setting_latest_state.pkl`
- Different run = different directory = no collision

---

## 5. Experiment Traceability

### 5.1 Config Preservation
- `config_used.yml` — exact copy of YAML config used for this run
- Saved via `shutil.copy2()` preserving timestamps

### 5.2 Run Metadata (`run_metadata.json`)
```json
{
  "timestamp": "2026-11-09T18:00:00.123456",
  "seed": 1,
  "device": "mps",
  "config_file": "./conf/ResNet18_cifar10_SL_badnets_rate010.yml",
  "method": "RCE",
  "ref_method": "RCE",
  "dataset": "cifar10",
  "python_version": "3.10.11",
  "pytorch_version": "2.14.0",
  "cuda_available": false,
  "mps_available": true
}
```

### 5.3 Complete Artifact Inventory per Run
```
run_dir/
├── config_used.yml              # Exact YAML used
├── run_metadata.json            # Full metadata
├── ex_setting.txt               # Training log
├── ex_setting_latest_state.pkl  # State pickle
├── ex_setting_latest_target_model.pt  # Model checkpoint
├── datamap/
│   └── cifar10_full.csv         # Datamap confidence CSV
├── tsne/
│   ├── model_tSNE_... .npy      # Activations
│   ├── model_tSNE_... .pdf      # TSNE plot (PDF)
│   └── model_tSNE_... .png      # TSNE plot (PNG)
├── rce_histograms/
│   ├── ..._probs.npy            # Probability arrays
│   ├── ..._otsu_th.npy          # Thresholds
│   ├── ..._labels.npy           # Labels
│   ├── ..._SCEhist_... .png     # Histogram plots
│   └── ..._DataDist_... .png    # Poison rate plots
├── ex_setting_latest_state.pkl  # Final state
└── ex_setting_latest_target_model.pt  # Final model
```

---

## 6. Tests Executed and Results

| Test | Status | Description |
|------|--------|-------------|
| `test_mps.py` | ✅ PASS | Core MPS compatibility |
| `test_rce.py` | ✅ PASS | RCE query strategy |
| `test_mixmatch_quick.py` | ✅ PASS | MixMatch training |
| `test_reproducibility.py` | ✅ PASS | Seeds 1,2,3 deterministic |
| `test_metrics.py` | ✅ PASS | Corrected precision/recall |
| `test_partitioning.py` | ✅ PASS | Dataset splitting logic |
| `test_output_safety.py` | ✅ PASS | All 6 safety tests |

### Output Safety Test Results
| Test | Status | Description |
|------|--------|-------------|
| Unique run directory creation | ✅ PASS | Each run gets unique directory |
| Collision handling | ✅ PASS | Numeric suffix on collision |
| Config copy | ✅ PASS | Exact YAML copied to run_dir |
| Run metadata | ✅ PASS | Full metadata JSON written |
| Overwrite protection | ✅ PASS | File existence detection |
| Existing files unchanged | ✅ PASS | Other run dirs untouched |

---

## 7. Exact Training Command

```bash
cd /Users/shashvatgarg/Desktop/Main/Shashvat_Garg/Study/DTU/Study/4th Year/Sem - 7/BTP/Harvey + Extra/HARVEY_MPS

python3 train.py \
  -c conf/ResNet18_cifar10_SL_badnets_rate010.yml \
  --seed 1 \
  --device mps
```

### Expected Output Location
```
results/
  cifar10/
    RCE-RCE/
      run_20261109_180000_seed1/    # ← This run's directory
```

### No Additional Arguments Required
- The `--device mps` flag works as before
- All output safety features are automatic
- No changes to command-line interface needed

---

## 8. Verification Checklist

- [x] All 6 output safety tests pass
- [x] All existing MPS tests pass (MPS, RCE, MixMatch, reproducibility, metrics, partitioning)
- [x] No algorithm changes introduced
- [x] No hyperparameter changes
- [x] No CATA integration
- [x] Original YAML configuration unchanged
- [x] Previous experiment results untouched
- [x] Original training command works without modification
- [x] Unique run directory created per execution
- [x] Collision handling verified
- [x] Config copy saved
- [x] Metadata written
- [x] All output paths under run_dir
- [x] Existing files in other directories protected

---

## 9. Remaining Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Timestamp resolution: seconds | Potential collision if 2 runs in same second | Numeric suffix fallback |
| `torch.save()` still overwrites within same run_dir | Expected behavior for checkpoint updates | Different runs use different directories |
| No distributed training support | Single-device only | Not in original scope |
| No automatic cleanup of old runs | Disk usage grows | Manual cleanup or external policy |

---

## 10. Conclusion

The HARVEY MPS port now has **complete output safety**:

1. **Every run gets a unique, timestamped directory** — no overwrites possible
2. **Collision detection with numeric suffixes** — even simultaneous runs are safe
3. **Full experiment traceability** — config, metadata, all artifacts preserved
4. **All outputs contained** — training logs, checkpoints, TSNE, RCE histograms, datamaps
5. **Zero risk to previous experiments** — completely isolated by directory
6. **Zero command-line changes** — works with original training command

The system is **ready for overnight full training** with complete confidence that no previous results will be affected.