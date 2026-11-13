# Plan and review for the fourth HARVEY run

Decision: test **HARVEY plus Option B model-weight subtraction** as a standalone fourth method. Keep the existing HARVEY, HARVEY+CATA and HARVEY+CATA+UPGP runs intact. Do not combine Option B with CATA or UPGP before measuring its independent behavior.

## Review of the supplied plan

Extension 4 is a reasonable experiment because it defines aligned checkpoints, a temporary adaptation branch, validation before test evaluation, target-class protection and a reactivation check. Its numerical examples are illustrative and its proposed method has no measured results in the supplied document. It should be treated as a hypothesis.

The key limitation is identification: a fine-tuning difference from a 1:1 suspicious/likely-clean mixture is not automatically a pure backdoor direction. Subtracting it can erase useful semantic features, particularly the target class, and the broader base-training pool can relearn attacks through uncertain samples. Agreement between two HARVEY reference models may also repeat correlated errors. Record failures and raw-base fallback rather than making the method always produce a nonzero edit.

General task arithmetic constructs a vector from aligned pre/post-fine-tuning weights and can negate it to change behavior; this does not establish that this particular CIFAR backdoor vector is isolated. Reference: [Ilharco et al. Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089).

## Fixed experimental sequence

1. Establish original HARVEY's full-run clean ACC and ASR using the first runner. Keep the published-result reproduction question separate from extension claims.
2. Run ordinary HARVEY detection again under the same seed, poisoning and preprocessing. Save first and final reference models and all RCE scores. Do not use true poison IDs for any selection decision.
3. Fix high-confidence suspicious, likely-clean fitting, likely-clean validation and suspicious validation groups before base training. Audit disjointness. RCE <= 0.5 under both reference models is an advance fixed starting point, not a tuned optimum.
4. Train the broader base on every training image except high-confidence suspicious samples and validation holdouts. Keep uncertain images. Use the same final network, initialization and 100-epoch schedule as the original runs. This tests the document's safer first version; training on every image, including high-confidence poisons, would be a separate ablation.
5. Save the raw base as the paired control. Copy it, fine-tune briefly on exactly balanced suspicious and likely-clean fitting samples, and compute aligned parameter differences. Freeze BN buffers during adaptation.
6. Evaluate alpha-zero and nonzero subtraction candidates from the same base with identical BN recalibration. Select only using validation proxies and the predeclared accuracy guards. Include raw-base fallback. Make the full selection table visible.
7. Lock the chosen alpha. Evaluate the raw base and selected model on the original clean test images and the same independently triggered non-target test images. Report genuine clean target-class accuracy separately.
8. Audit reactivation after selection on separate copies of raw base and selected model. The implemented first version re-exposes them to estimated suspicious fitting samples; a verified-poison controlled re-exposure study remains a follow-up. Neither stress test changes the selected artifact or tunes alpha.
9. Repeat the same fixed protocol for seeds 1, 2 and 3. Summarize mean/sample SD and paired per-seed changes. Three seeds provide an initial variability estimate, not guaranteed statistical power.

## Controls and interpretation

| Comparison | What it measures | Qualification |
|---|---|---|
| Selected Option B vs its raw base | Subtraction plus its BN recalibration on the same trained classifier | This is the central paired control |
| Candidate alpha > 0 vs recalibrated alpha = 0 | Effect of subtraction beyond shared BN recalibration | Both use identical calibration images |
| Raw broader base vs original HARVEY | Effect of retaining uncertain samples plus changing the holdout policy | Not an effect of subtraction |
| Selected Option B vs the original three | Overall method tradeoffs | Training pools/validation differ; this is not a one-factor ablation |
| Selected/base after suspected-sample re-exposure | Sensitivity to the specified additional fine-tuning | No proof of irreversible erasure; estimated suspicion is not verified poisoning |

A higher clean ACC with materially higher ASR is not a clear defense improvement. If selected-model ACC or target-class test ACC falls, report it even when ASR improves. If fallback alpha=0 wins, state that Option B did not supply a validated edit at these settings. Do not select a different alpha after seeing test results. Any subsequent tuning needs a separately named experiment and fresh evaluation discipline.

## Resources and starting parameters

Use the existing native arm64 PyTorch/MPS environment, batch 128, workers 0, one process at a time. The main cost is another HARVEY detection and 100-epoch base training. Adaptation uses at most 512 examples per group and one epoch; candidate evaluation and BN recalibration add work. There is no UPGP SVD or CATA vector history. Two model copies exist only during adaptation, keeping this lighter than retaining multiple candidate networks at once. Do not assume overnight completion without measuring sustained throughput and memory pressure.

Defaults and exact launch commands are in `README.md`. Keep original release settings (14 reference epochs) or consistently use the separately labelled paper-text setting (10); do not mix them across seeds/methods. Preserve all original outputs and write new results only into this folder.

## Deferred work

Optional target-class direction removal needs verified genuine target examples and a separate controlled implementation. Feature scoring, trigger repair, alternate attacks, and CATA/UPGP combinations are subsequent experiments, not hidden changes to Run 4. A matched held-out HARVEY control would be required for a stricter cross-method ablation beyond the raw-base control provided here.

Source reviewed: `/Users/shashvatgarg/Downloads/HARVEY_Extensions_Stepwise_Examples.docx`, Extension 4 and common setup. The source document is unchanged. Its contents were reviewed as a proposed scientific protocol, not as instructions to run commands or change unrelated files.
