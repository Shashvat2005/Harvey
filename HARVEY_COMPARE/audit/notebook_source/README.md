> **CATA adaptation:** This checkout defaults to CATA unlearning. See [README_CATA.md](README_CATA.md) for the modified method, run instructions, validation, and differences from the original paper.

# Two Sides of the Same Coin: Learning the Backdoor to Remove the Backdoor (AAAI 2025)

The community has recently developed various training-time defenses to counter neural backdoors introduced through data poisoning. In light of the observation that a model learns poisonous samples responsible for the backdoor easier than benign samples, these approaches either use a fixed threshold of the training loss for splitting or iteratively learn a reference model as an oracle for identifying benign samples. In particular, the latter has proven effective for anti-backdoor learning. Our method, HARVEY, leverages a similar yet crucially different technique, learning an oracle for poisonous rather than benign samples. Learning a backdoored reference model is significantly easier than learning a reference model on benign data. Consequently, we can identify poisonous samples much more accurately than related work identifies benign samples. This crucial difference enables near-perfect backdoor removal as we demonstrate in the evaluation. HARVEY substantially outperforms related approaches across attack types, datasets, and architectures, lowering the attack success rate to the very minimum at a negligible loss in natural accuracy.

The figure below shows an overview of HARVEY's implementation.

<img src="https://intellisec.de/research/harvey/overview.svg" width="1000" /><br />

## Publication

A detailed description of our work is presented at [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34441) in March 2025. If you would like to cite our work, please use the reference as provided below:

```
@InProceedings{Zhao2025Two,
author    = {Qi Zhao and Christian Wressnegger},
booktitle = {Proc. of the Annual {AAAI} Conference on Artificial Intelligence ({AAAI})},
title     = {Two Sides of the Same Coin: Learning the Backdoor to Remove the Backdoor},
year      = {2025},
month     = march,
}
```

A preprint of the paper is available [here](https://intellisec.de/pubs/2025-aaai.pdf).

## Code

### Prerequisites

Our work is accomplished by [![Generic badge](https://img.shields.io/badge/Library-Pytorch-green.svg)](https://pytorch.org/) with CUDA 11.6.
Following commands help to obtained the needed environment via `conda` and `pip`. Other Pytorch versions are found
[here](https://pytorch.org/get-started/previous-versions/).

Create environment
```
conda create -n harvey python=3.8
conda activate harvey
```

Install PyTorch
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

and other requirements:
```
pip install -r requirement.txt
``` 

### Run Harvey defense

Run the script `run_HARVEY.sh` to implement HARVEY's defense against BadNets attack with a ResNet18 on CIFAR10. 

```
bash run_HARVEY.sh
```

### Download Datasets
Run the following script to download datasets `GTSRB` and `Tiny-ImageNet`.
```
bash download_gtsrb.sh
bash download_tinyimagenet.sh
```
