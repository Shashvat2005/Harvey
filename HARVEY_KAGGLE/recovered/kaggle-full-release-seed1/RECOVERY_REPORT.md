# Kaggle metrics recovered from the training log

The pasted log confirms a completed full HARVEY+CATA+UPGP release-protocol run with seed 1. It contains 20 warmup epochs, 20 reference cycles with 14 training epochs each, 20 CATA requests, 100 final-training epochs and 5 UPGP epochs. All 447 events were parsed; the first line's missing opening brace was restored.

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Before UPGP at final-training epoch 100 | 86.04% | 1.4111% |
| After UPGP epoch 5 | 51.08% | 2.0889% |
| Change in percentage points | -34.96 | +0.6778 |

Reported runtime: 4.42 hours. Completion timestamp (UTC): 2026-11-12T21:59:35.124305+00:00.

The tested UPGP integration harmed both metrics in this run. Report the after-UPGP values as the third method's final result and the before-UPGP values as its within-run control; do not silently replace the final result with a better test epoch. The 86.04% control is the CATA-trained checkpoint in this run, not an independently executed standalone CATA experiment.

Meta splitting retained 29,895 estimated benign images and rejected 20,105. Logged poison-detection recall is 100%, precision is 24.87%, and retained-set poison contamination is zero. Under this constructed attack's 5,000-poison setup, those values imply 15,105 clean images were also rejected. This can contribute to lower final accuracy, but the log alone does not isolate the cause.

`result.json` preserves the complete event's result fields. `metrics.jsonl` and `epoch_metrics.csv` preserve recorded curves. `split_quality.json` preserves the aggregate meta-split audit. `recovery.json` records provenance and checks.

Model weights, resumable checkpoints, sample IDs and unlogged environment settings cannot be reconstructed from numerical logs. Those require saved Kaggle output files. Existing local runs and training source were not modified.
