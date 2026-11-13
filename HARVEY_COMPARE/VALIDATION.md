# Validation performed 2026-11-12

Environment: arm64 macOS, native Python 3.10.11, PyTorch 2.14.0, torchvision 0.29.0; MPS available. Used the existing Python environment for checks; `setup.sh` is provided for an isolated environment and was shell-syntax checked, but its pip-install step was not run.

Passed:

- Six CPU unit tests: global CATA masking with ties, sign election and zero ties, base/buffer preservation, hard row-space orthogonality, relaxed projection equality with explicit matrix multiplication, and UPGP loss/backward.
- Python compilation of runner, CATA, UPGP, comparison and benchmark; shell syntax for setup and all launchers.
- All three variants sequentially through `run_all.sh` on MPS using 80 synthetic training images, 40 synthetic test images, and one epoch per stage. The CATA and CATA+UPGP pre-postprocessing metrics and losses matched. These random-data metrics are not research results. In this synthetic pipeline no suspected poison remained after CATA, and UPGP correctly recorded a skip.
- A separate full 11.17M-parameter ResNet18 MPS synthetic check exercised UPGP's actual (non-skipped) loss, backward, CPU SVD, projection and optimizer update across 21 weight tensors. Finite parameters afterward; approximately 6.53 seconds. Details: `upgp_validation.json`.
- Intentionally interrupted CATA before a reference epoch checkpoint and resumed through final training. The replayed reference loss, CATA adaptation loss, final loss and metrics matched the uninterrupted synthetic run. Details: `runs/resume-validation/` and `runs/script-validation/`.
- Completed-run resume skipped work correctly.
- Temporary comparison fixtures checked same-setting paired deltas and suppression of deltas for a changed batch size. Fixtures were deleted. Synthetic smoke results are excluded from research comparisons.
- Vendored ResNet18 source matches `HARVEY_source/models/target_models.py` byte for byte. CATA source matches the notebook's extracted `utils/cata.py` byte for byte.

Real CIFAR-10 benchmark (12 training batches, first two discarded): batch 128, mean compute time approximately 0.139 seconds/batch. Approximately 0.906 minutes for 50,000-image training compute or 1.359 hours for 100 final epochs of 45,000 images. This excludes data preparation, augmentation/loading, all reference phases, CATA, SVD, evaluation, and disk writes; thermal throttling and other applications can change sustained speed. Details: `benchmark.json`.

No complete real-data training run was launched. No 93% reproduction or accuracy improvement is claimed. Full-run accuracy, ASR, overnight completion and long-run memory pressure remain to be measured. Multiple-cycle resume and interruption inside an epoch are designed to replay from the last durable checkpoint; exhaustive interruption points and cross-machine determinism were not tested.
