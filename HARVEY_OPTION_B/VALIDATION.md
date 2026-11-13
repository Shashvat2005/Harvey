# Checks performed on 12 November 2026

Native arm64 Python 3.10, PyTorch 2.14.0, torchvision 0.29.0, MPS available. No new package installation was required for the checks; the isolated setup installer was shell-syntax checked, not executed.

- Six CPU tests passed: disjoint holdouts and retained uncertain samples; aligned subtraction and preservation of BN buffers; validation guards and fallback; BN-only recalibration; balanced adaptation with frozen BN; postprocessing integration including proof that the selection file exists before any test-evaluation call.
- Python compilation and shell syntax checks passed.
- Full ResNet18 synthetic MPS pipeline completed through ordinary HARVEY detection, Option B group construction, base fitting, fallback, final evaluation and artifact saving. Random synthetic data produced no high-confidence suspects, correctly preserving the raw base. Metrics in these runs are not research results.
- A separate full ResNet18 MPS check used artificial fitting/validation memberships to exercise the eligible path: balanced branch adaptation, task-vector construction, all six alpha candidates, BN recalibration, validation selection, final evaluation and both re-exposure audits. All model parameters remained finite; details are in `eligible_path_validation.json`. It correctly selected alpha zero because suspicious prediction rate was already zero; a nonzero candidate was not forced.
- Intentional interruption before the first base epoch checkpoint was replayed via `--resume` from the saved post-meta checkpoint. The completed synthetic run matched the uninterrupted smoke run's final loss and selected/base metrics. Completed-run resume also skips work.
- Four-method reporting handles absent full-run outputs without inventing results and excludes synthetic results. No real-data table can be produced until full experiments finish.
- All input/source/configuration files recorded in `original_runner_hashes.json` still match their recorded SHA-256 values. The original three runners and original source/notebooks were not edited.

Not verified: full CIFAR-10 accuracy/ASR or target-class preservation, improvement relative to any existing method, sustained memory/runtime, multi-seed variation, every possible interruption point, or statistical validity of the proposed validation proxies. No full training run was started. This is a configured exploratory implementation, not a demonstrated backdoor-removal result.
