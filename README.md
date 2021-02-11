# Reproducibility Challenge 2020
This project seeks to reproduce some of the claims presented in the paper [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness](https://arxiv.org/abs/1905.10626)(ICLR 2020)

Tianyu Pang, Kun Xu, Yinpeng Dong, Chao Du, Ning Chen and Jun Zhu

## Environment setup and Libraries used:

We reproduce the results presented in the original paper using the following environment settings:
- OS: Ubuntu 18.
- GPU: Geforce 1080 Ti or Tesla P100
- Cuda: 9.0, Cudnn: v7.03
- Python: 3.6.0
- cleverhans: 2.1.0
- Keras: 2.2.4
- tensorflow-gpu: 1.9.0
- pytorch:
- scipy:
- argparse:

## Addition Features Added:
We included the following additional features apart from the original code:
- **Python Center Generation:** We have implemented a python version of MMC center generation code under `center_gen_python/mmc_center_gen.py`
```shell
python mmc_center_gen.py --var=10 --dim_dense=256 --num_class=10
```
`var` is distance between the MMC centers `$C_{MMC}$`

$ \sum_{\forall i}{x_i^{2}} $
## Usage:

### Training:

### Inference:
