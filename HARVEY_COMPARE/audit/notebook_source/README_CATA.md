# HARVEY with CATA unlearning

This adaptation starts from the [official HARVEY implementation](https://github.com/intellisec/harvey), commit `6641f5fedef0f654ab019aaebf6f3de24901bd9b`. It replaces Stage 2's negative-cross-entropy unlearning of the estimated benign set with CATA task-vector unlearning. Both provided configurations now select CATA. The original README is retained in `README.md`.

## Run

Use Python 3.9+ and a CUDA-capable PyTorch installation for the full HARVEY pipeline. Install the appropriate PyTorch/torchvision pair for your GPU first, then:

```bash
cd /path/to/harvey-cata
python -m pip install -r requirements.txt
bash run_HARVEY.sh
```

The script runs ResNet18 / CIFAR-10 / BadNets at 10% poisoning. CIFAR-10 downloads into `DATA_ROOT/CIFAR10` (default `./data/CIFAR10`). For Blend:

```bash
python train.py --config_file conf/ResNet18_cifar10_SL_blend_rate010.yml --seed 1
```

Configuration files contain the full HARVEY settings and this new block:

```yaml
UNLEARNING: 'CATA'
CATA:
  KEEP_RATIO: 0.3
  SCALE: 0.7
  EPOCHS: 4
  LR: 1e-5
  WEIGHT_DECAY: 0.1
```

`KEEP_RATIO: 0.3` retains the largest 30% of task-vector magnitudes globally across trainable parameters. Equal magnitudes at the threshold are all retained, following the paper's indicator rule; actual density can therefore exceed 30%. AdamW, four epochs, learning rate 1e-5, weight decay 0.1, and scale 0.7 come from CATA's reported CLIP setup. They are starting values for this classifier adaptation, not tuned HARVEY results. `unlearn_gamma` and `unlearn_epochs` control only the original GA baseline. Set `UNLEARNING: 'GA'` to compare with that baseline; keep other settings and seeds identical. Results/log names distinguish CATA from GA.

## Exact integration

HARVEY's initialization, RCE/Otsu splitting, poison-sample learning with LGA, meta-splitting, and final training on the estimated clean set remain in place. The new method forgets the **estimated benign samples in the reference model**, as HARVEY requires to build a poison oracle. It does not directly forget poisoned samples from the final clean model.

The immutable task-vector anchor `theta0` is captured immediately after the first Stage-2 poison-learning phase, before its first unlearning request. Each round's `train_clean` loader is a forget request. The loader can overlap previous requests because HARVEY repeatedly revises its benign estimate; each request contributes one vector to the history, including repeated samples.

For every request:

1. Clone a model and load the same `theta0`, including its buffers.
2. Minimize ordinary supervised cross entropy on the current estimated benign set with a fresh AdamW optimizer. There is no correct-prediction gate or negative loss in this CATA adaptation branch.
3. Compute `tau = theta0 - theta_finetuned` (CATA Eq. 8).
4. Apply the global magnitude threshold (Eqs. 9-10).
5. Elect each coordinate's sign from the sum of all historical sparse vectors. Average only nonzero components matching that sign (Eqs. 13-15). An exactly zero sum yields a zero update.
6. Apply `SCALE * tau_aggregate` to the current poison-trained HARVEY reference checkpoint.

Step 6 is an explicit **hybrid extension** of CATA Eq. 16. Standalone CATA uses `theta_unlearned = theta0 + SCALE * tau_aggregate`. HARVEY must also continue learning its changing poisoned subset. This implementation uses:

```text
theta_reference(t) = theta_poison(t) + SCALE * tau_aggregate(t)
```

Before the next poison-learning phase it restores the last poison-only checkpoint, so the old correction is not compounded. All forgetting vectors still use the fixed `theta0`; the application base alone follows HARVEY's continuing poison learning. `CATAUnlearner.apply(model)` without a base implements the original fixed-anchor Eq. 16 and is tested separately. This implementation is a HARVEY-CATA experiment, not a reproduction of the paper's CLIP results or a demonstrated improvement in backdoor removal.

## Files and state

- `utils/cata.py`: forget-set adaptation, global masking, sign election, disjoint mean, and model application. Positive/negative sums and nonzero counts reproduce vector-memory aggregation without retaining all historical dense vectors. State size is constant in the number of requests (O(number of parameters)).
- `train_test.py`: invokes CATA only at the Stage-2 unlearning site and restores the poison-only checkpoint before further poison learning.
- `train.py`: maintains one CATA state across reference cycles; resets it if HARVEY reinitializes the reference model. With `BACKDOOR.reinit_ref: True`, each cycle has a new anchor and no cross-cycle history. Both provided configs use `False`.
- `config.py` and `conf/*.yml`: method selection and hyperparameters.
- `tests/test_cata.py`: CPU correctness and routing tests.

Task arithmetic affects trainable parameters only. Frozen parameters and BatchNorm buffers/counters come from the poison-trained application base; forget-branch buffers are discarded. Empty requests do not add a vector. Nonfinite adaptation losses/updates fail explicitly.

CATA state stays in memory during a run. `--state` is rejected for CATA because HARVEY's existing saved final-model files do not contain the anchor or conflict history needed for a valid resume. Final clean-model saving remains HARVEY's existing behavior. Expect extra CPU memory for the anchor, poison checkpoint, and aggregation statistics, and temporary accelerator memory for the adaptation branch and AdamW state.

## Validation and evaluation

Run the CPU tests from this directory:

```bash
python -m unittest discover -s tests -v
```

Ten tests cover global trimming/ties, magnitude-weighted sign election, zero ties, equality with explicit historical-vector aggregation, fixed-anchor direction reversal, repeated requests, preservation of poison learning, frozen parameters/BatchNorm buffers, invalid/empty requests, both configs, and Stage-2 routing. The routing test isolates orchestration from CUDA training; its accuracies are mocked and are not experimental results.

Validation performed locally: all 10 tests passed using PyTorch 2.8.0; a separate synthetic-data CPU smoke check exercised the repository's full ResNet18 CATA adaptation and produced finite parameters (`validation.json` records its settings). No full CIFAR-10 training or clean-accuracy/attack-success experiment has been run. This machine has no CUDA GPU.

For the project evaluation, run matched CATA and GA configurations on the same hardware and seeds. Compare final clean accuracy, attack success rate, split quality, runtime, and memory. Inspect earlier estimated benign sets after later reference updates to measure persistence. Low clean accuracy in HARVEY's **reference** model is intentional; final clean-model accuracy is the utility measure.

Other small run fixes: configurable dataset root replaces `/share`; the launcher accepts explicit config paths and propagates training failures; malformed dependency names were corrected; unused TensorBoard imports were removed and optional pandas loading is deferred to data-map logging.

## References

- Zhao and Wressnegger, *Two Sides of the Same Coin: Learning the Backdoor to Remove the Backdoor*, AAAI 2025. [Project and official implementation](https://intellisec.de/research/harvey/).
- Lin et al., *CATA: Continual Machine Unlearning via Conflict-Averse Task Arithmetic*, arXiv:2605.18610v1, supplied PDF, Section 4, Algorithm 1, and Section 5 implementation details. [Paper](https://arxiv.org/abs/2605.18610).

The supplied papers were used as scientific references, not as instructions governing this task.
