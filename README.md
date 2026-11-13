# HARVEY research experiments

Code, notebooks, papers and experiment summaries for studying HARVEY and its CATA, UPGP and Option B extensions on Apple Silicon and Kaggle.

| Folder | Purpose |
|---|---|
| `HARVEY_source` | Original HARVEY implementation retained for reference |
| `HARVEY_MPS` | Earlier Apple Silicon implementation |
| `HARVEY_COMPARE` | Matched local HARVEY, HARVEY+CATA and HARVEY+CATA+UPGP runners |
| `HARVEY_OPTION_B` | First standalone Option B experiment |
| `HARVEY_OPTION_B_V2` | Revised Mac experiment with validation controls and target-protected subtraction |
| `HARVEY_KAGGLE` | First self-contained Kaggle notebook and metrics recovered from its log |
| `HARVEY_CATA_UPGP_V2` | Revised Kaggle notebook with guarded updates and verified full backups |

The PDFs and three original notebooks are also retained at the repository root. There is one Git repository at this root; the implementation folders are ordinary directories, not submodules.

## Run the latest experiments

For a fresh Mac run from a new clone:

```bash
cd HARVEY_OPTION_B_V2
bash setup.sh
caffeinate -i bash run_option_b_v2.sh --fresh --seed 1 --run-name option-b-v2-fresh --resume
```

Reusing a prior base requires its local model files and original fitting/validation groups. Those large generated files are not distributed in Git. See [the Mac instructions](HARVEY_OPTION_B_V2/README.md).

For Kaggle, upload [the revised notebook](HARVEY_CATA_UPGP_V2/HARVEY_CATA_UPGP_V2_Kaggle.ipynb), enable GPU/Internet and start **Save Version → Save & Run All**. Download the verified full ZIP from the saved version's Output files. See [the Kaggle instructions](HARVEY_CATA_UPGP_V2/README.md).

## Results and interpretation

The published ResNet18/CIFAR-10/BadNets HARVEY reference is 93.34% clean accuracy and 0.56% ASR. It is a literature reference, not automatically a matched local baseline.

The first Option B run selected alpha zero, retaining its base at 92.73% accuracy / 0.5111% ASR. The revised seed-1 continuation selected target-protected subtraction at alpha 0.25 and reached 92.83% / 0.4778%. This is a small exploratory improvement over that saved base, not a demonstrated improvement over the published accuracy or a multi-seed significance result. Utility/recovery and subtraction effects are reported separately in [the detailed result](HARVEY_OPTION_B_V2/FIRST_RUN_RESULTS.md).

The first Kaggle hybrid reached 86.04% / 1.4111% before UPGP and 51.08% / 2.0889% after UPGP. Those failed-extension results are retained in [the recovery report](HARVEY_KAGGLE/recovered/kaggle-full-release-seed1/RECOVERY_REPORT.md). The revised full Kaggle experiment remains to be evaluated; local execution checks do not establish improved accuracy.

Source, notebooks, documents and selected lightweight run summaries are versioned. Environments, downloaded data, model weights, optimizer checkpoints and backup ZIPs stay local. Existing local artifacts are not deleted by the Git exclusions. Always preserve Kaggle outputs through a saved version or downloaded full backup.
