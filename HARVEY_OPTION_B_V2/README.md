# Option B V2 on Apple Silicon

This is a new exploratory implementation of HARVEY plus the Word document's Option B. It leaves the first Option B run, baseline code, earlier notebooks and results untouched. It aims to improve accuracy and ASR, but cannot guarantee either. Controls and fallback choices are explicit.

## Run on this Mac

From the repository root:

```bash
cd HARVEY_OPTION_B_V2
caffeinate -i bash run_option_b_v2.sh --run-name option-b-v2-protected --seed 1 --resume
```

The launcher uses the environment in `HARVEY_OPTION_B/.venv` or `HARVEY_COMPARE/.venv` if present. Otherwise run `bash setup.sh` inside this folder first. An explicit interpreter can be selected with `PYTHON=/absolute/path/to/python`.

For seed 1, the default reuses your existing `HARVEY_OPTION_B/runs/option-b-full/harvey_option_b/seed1/base_model.pt` **and its original fitting/validation groups**. This avoids repeating reference detection and the 100-epoch base training. It verifies the seed, protocol, exact poison IDs and group disjointness, then writes only into this V2 folder. The old 92.73%/0.51% result remains intact. A reused base is a continuation, not an independent replication. The first internal V2 trial under `runs/option-b-v2/` used the earlier unprotected/no-recovery revision and selected utility only (92.78% ACC / 0.50% ASR). Its source snapshot is preserved with that run; the final protected revision uses a NEW run name and must not resume that older code state.

To train a new base from scratch instead:

```bash
caffeinate -i bash run_option_b_v2.sh --fresh --run-name option-b-v2-fresh --seed 1 --resume
```

For seeds without an existing matching base, use `--fresh` or supply `--reuse-source /path/to/that/seedN`. Release protocol remains the default (14 reference epochs per cycle); `--protocol paper-text` selects 10. Use a new run name whenever settings, source or environment change. Run one process at a time on the 16 GB Mac.

A synthetic execution check:

```bash
bash run_option_b_v2.sh --profile smoke --fresh --run-name smoke --resume
```

## Changes from the first Option B run

1. **Preserve BatchNorm buffers.** The old alpha-zero recalibration reduced validation ACC by about 2.38 percentage points. V2 performs no recalibration; all utility, direction and candidate stages freeze running means, variances and counters. BN affine parameters may receive gradients.
2. **Filter the suspicious group using semantic agreement.** Start with HARVEY's estimated suspicious fitting/validation groups separately. A frozen base teacher must predict the same non-target class, at confidence >=0.85, under a horizontal flip and a small reflected translation. This uses neither true poison IDs nor the known trigger mask. It assumes a declared single target (class 0) and stable weak views. The teacher can still make correlated errors; this is not proof of poisoning.
3. **Separate utility fine-tuning from subtraction.** Evaluate class-balanced likely-clean fitting with CE plus a frozen-base teacher KL penalty, LR 0.001 and 0.005, up to two epochs per independent arm. Each arm starts at the same raw base. Rejected FITTING images can additionally re-enter utility fitting only when the original image and both weak views confidently (>=0.95) agree with the OBSERVED label. This is a separate data-recovery refinement; it never touches validation samples or corrects labels from poison truth. These candidates are NOT verified clean and may still contain triggers. Disable it with `--no-recover-clean` for an ablation. This targets both class imbalance and lost label-consistent fitting data without attributing utility/recovery gains to Option B subtraction.
4. **Compare literal and differential directions.** From the chosen utility anchor, one branch learns the balanced suspicious/likely-clean mixture (the document's Option B). A matched clean-only branch uses the same learning rate, number of steps and per-step example count. The differential direction is `delta_mixture - delta_clean`; it is a proposed extra ablation, not established pure backdoor isolation. Both branches use SGD 0.001 for two epochs on up to 1,024 examples per group. Each also has a target-protected variant: remove the component whose subtraction would increase the estimated clean target-class fitting loss to first order. This uses an averaged gradient from the ORIGINAL likely-clean fitting pool, not from newly recovered images. It approximates the document's optional target protection with estimated labels, not an independently verified clean target dataset. Higher-order interference is still possible.
5. **Bound changes and retain controls.** Try alpha 0.25, 0.5, 1 and 2 from the same anchor; zero is the unchanged control. Bound parameter displacement to 0.2% of the anchor parameter L2 norm. This is a global bound, not a per-layer guarantee. Each candidate is evaluated only on the fixed validation groups.
6. **Validation-selected fallback.** Require at least 32 filtered suspicious fitting and validation samples and 20 likely-clean target-class validation examples. Accept no overall validation-accuracy drop or suspicious target-prediction-rate increase relative to the raw base. Allow at most 1 percentage point target-class validation drop and 0.001 absolute mean target-probability increase. Rank allowed candidates by accuracy, then suspicious response, then a meaningful confidence reduction. If no candidate improves the incumbent, keep the control. These noisy proxies do not guarantee test ASR/ACC.

The utility-stage anchor can differ from the raw base. `selected_edit.kind` records `base`, `clean_polish`, or `option_b`; direction and alpha are included for subtraction. `recovered_fitting_count` and `recovery_enabled` identify the additional utility data policy. A utility-only gain must not be described as successful Option B subtraction. If insufficient trustworthy validation examples exist, all refinement is skipped, including utility fitting.

## Fresh training and comparison assumptions

Fresh V2 runs reserve 5% of training IDs, stratified by **observed** labels, before any fitting. Reference learning and base training exclude these IDs. Likely-clean and suspicious validation subsets are then derived from this fixed reserve after ordinary HARVEY meta splitting. In reuse mode, the original Option B holdouts are retained: its reference detection already saw those images, but its base fitting did not. These designs are different and are recorded as fresh versus reused.

Fresh Option B base fitting follows the document's safer broader-pool version: exclude high-confidence suspicious images and reserved validation IDs while retaining uncertain images. Architecture, poisoning, preprocessing and the 100-epoch classifier schedule match the original implementation; known original CUDA/MPS reproduction differences still apply. The method is Option B alone plus explicitly reported utility/differential ablations; it does not include CATA, UPGP, trigger repair or feature-space voting.

Candidate selection never receives test images or poison truth. Only after the selection file is written are the raw base, utility anchor and selected weights evaluated on the clean and triggered test sets. Test-set values are reported even if they regress. Because earlier test outcomes informed this development, treat these V2 trials as exploratory; freeze settings and confirm on additional paired seeds before a strong improvement claim. A matched fresh control can be run via `training.py --variant harvey_v2_control --fresh` with the same settings and name in a separate variant directory.

## Outputs

```text
HARVEY_OPTION_B_V2/runs/option-b-v2-protected/harvey_option_b_v2/seed1/
```

- `result.json`: final/base/utility test ACC, ASR, target ACC, chosen edit and runtime.
- `selection.json`: full validation candidate table and selected operation; no test selection.
- `refinement_groups.npz`: separate filtered fitting/validation groups and teacher pseudo labels.
- `base_model.pt`, `pre_refinement.pt`, `final_model.pt`: base, utility anchor and chosen weights.
- `task_vectors.pt`: literal, differential and their target-protected Option B parameter directions when eligible.
- `metadata.json`, `metrics.jsonl`, `checkpoint.pt`: reproducibility and progress.
- `reuse_provenance.json`: source base checksum and original training batch for reused runs.
- Fresh runs additionally save reference models, scores, groups and split-quality audit.

Epoch-level checkpoints cover fresh warmup/reference/base training. Interruption during refinement replays the entire bounded refinement stage from the saved base and RNG state. Never change code/settings mid-run; no source/package hashes from `.venv` or generated runs are included. Runtime excludes the original base training in reuse mode and says so explicitly.

Compare descriptively against your own baseline:

```bash
../HARVEY_OPTION_B/.venv/bin/python assess_results.py \
  runs/option-b-v2-protected/harvey_option_b_v2/seed1/result.json \
  --baseline-result /path/to/your/harvey/result.json
```

If no baseline JSON is provided, the helper prints the published 93.34%/0.56% reference, clearly labelled as a non-matched reference. A number exceeding that reference is not by itself evidence of a statistically reliable improvement.

`VALIDATION.md` documents executed checks. No default full refinement/improvement trial has been run automatically.

## Optional post-selection reactivation audit

After a completed run:

```bash
bash audit_reactivation.sh runs/option-b-v2-protected/harvey_option_b_v2/seed1
```

This separately re-exposes copies of the raw base and chosen model to one balanced epoch of up to 128 filtered suspicious and 128 likely-clean fitting examples at LR 1e-4. It writes `reactivation_v2.json` and never overwrites selected weights or chooses an alpha. Like the first implementation's audit, it uses estimated suspicious samples rather than a verified-poison oracle. It tests the specified re-exposure sensitivity, not irreversible erasure. Both comparison copies use the same deterministic sampling.

## First measured result

The final protected seed-1 continuation has already completed locally: **92.83% clean ACC / 0.477778% ASR**, selecting target-protected mixed-direction subtraction at alpha 0.25. Its raw base was 92.73% / 0.511111%. This is a small one-seed gain over that base, not an established gain over your matched HARVEY baseline or the paper (93.34% ACC). `FIRST_RUN_RESULTS.md` separates utility and subtraction effects. The default command above skips this completed run with `--resume`; use a different run name for another trial or `--fresh` for full training.
