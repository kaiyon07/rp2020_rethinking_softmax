# Reproducibility Challenge 2020
This project seeks to reproduce some of the claims presented in the paper [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness](https://arxiv.org/abs/1905.10626)(ICLR 2020)

Tianyu Pang, Kun Xu, Yinpeng Dong, Chao Du, Ning Chen and Jun Zhu

## Environment setup and Libraries used:

We reproduce the results presented in the original paper using the following environment settings:
- OS: Ubuntu 18.04
- GPU: Geforce 2080 Ti or Tesla V100
- Cuda: 9.0
- Python: 3.6.0
- cleverhans: 2.1.0
- Keras: 2.2.4
- tensorflow-gpu: 1.9.0
- pytorch: 1.4.0
- scipy: 0.17.0
- argparse: 1.4.0

## Addition Features Added:
We included the following additional features apart from the original code:
- **Python Center Generation:** We have implemented a python version of MMC center generation code under `center_gen_python/mmc_center_gen.py`.
```shell
python mmc_center_gen.py --var=10 --dim_dense=256 --num_class=10
```
`var` is the distance between the MMC centers `$C_{MMC}$`, `dim_dense` is the dimension of the final dense layer, `num_class` is the number of the classes present in the dataset. 

We have also implemented a `no_args` version of `mmc_center_gen.py` under `center_gen_python/mmc_center_gen_no_args.py`.
```shell
python mmc_center_gen.py
```
changes to be made on the `Line 17` for `var`, `dim_dense` and `num_class` respectively.

- **Hierarchical MMC Centers (HMMC):** HMMC is a variant of MMC center loss, it is mainly used to create hierarchical tree structure in which a super-class can have multiple sub-class inside it. All these sub-classses have similar representation as compared to the other super-class. Ex- CIFAR-100 has 100 classes having 20 super-classes and each super-class has corresponding 5 sub-classes. Code is provided under `center_gen_python/hmmc_gen.py`.
```shell
python hmmc_gen.py --var=10 --dim_dense=256 --num_class=100 --var_2=1 --num_sup_class=20
```
`var` is the distance between the super-class MMC centers `$C_{MMC}$`, `dim_dense` is the dimension of the final dense layer, `num_class` is the number of the classes present in the dataset, `var_2` is the distance between the sub-class MMC centers and `num_sup_class` is number of super-classes present in the dataset.  

- **Isometric Centers:** For the model with `dim_dense` less than `num_class` then we cannot directly use MMC center loss, we then use isometric center loss. Implementation can be found under `center_gen_python/isometric.py`. 

- **Distorted MMC Centers (DMMC):** DMMC is also a variant of MMC center loss, where two centers can be brought closer to each other and vice versa depending on the similarity index value. Implementation can be found under `center_gen_python/joint_mmc.py`. 
 ```shell
python joint_mmc.py --var=10 --dim_dense=256 --num_class=10 
```

- **PyTorch MMC Centers loss:** We have also implemented a pytorch version of MMC center loss under `mmc_torch`.

**Note:** All the generated centers (`.mat` file) are present under `center_gen_python/generated_centers`.


## Usage:
The original implementation of MMC centers loss can be found at [Max-Mahalanobis-Training](https://github.com/P2333/Max-Mahalanobis-Training). The training procedure is same as mentioned in the original repository. We additional provide inference code for testing purpose as there was no support for inference in the original repository because the validation split was same as testing split.

All the experiments are performed on `MNIST`, `CIFAR-10` and `CIFAR-100` datasets. 

### Training:
We train the models the same way the authors of the original paper have done. 
- **Training the models using SCE loss**
Models can be trained with SCE loss by the following commands:
```shell
python train.py --batch_size=50 --dataset=[dataset] --optimizer='mom' --lr=0.01 --version=2 --use_MMLDA=False --use_BN=True --use_dense=True --use_leaky=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author.
Also `version` refers to Resnet architecure.
- **Training the models using MMC loss**
Models can be trained with MMC loss by the following commands:
```shell
python train.py --batch_size=50 --mean_var=10 --dataset=[dataset] -optimizer='mom' --lr=0.01 --version=2 --use_MMLDA=True --use_ball=True --use_BN=True --use_random=False --use_dense=True --use_leaky=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author.
The flag `use_ball` can be changed to `False` if you want to train the model on MMLDA loss.
The `meanvar` parameter can be used to change the distance between the optimal centers.

- **Training the models adversarially using SCE loss**
Models can be trained with MMC loss by the following commands:
```shell
python advtrain.py --batch_size=50 --dataset=[dataset] --optimizer='mom' --lr=0.01 --version=2 --adv_ratio=1.0 --use_MMLDA=False --use_target=False --attack_method='MadryEtAl' --use_BN=True --use_random=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author. The flag `use_ball` can be changed to `False` if you want to train the model on MMLDA loss.
The `meanvar` parameter can be used to change the distance between the optimal centers.
 The parameter `attack_method` can be changed to `MadryEtAl`, `FastGradientMethod` or `MomentumIterativeMethod`. The flag `use_target` if `True` uses targeted attacks to train the model and if `False` uses untargeted attacks to train it.

- **Training the models adversarially using MMC loss**
Models can be trained with MMC loss by the following commands:
```shell
python advtrain.py --batch_size=50 --mean_var=10 --dataset=[dataset] --optimizer='mom' --lr=0.01 --version=2 --adv_ratio=1.0 --use_MMLDA=True --use_ball=True --use_target=True --attack_method='MadryEtAl' --use_BN=True --use_random=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author. The parameter `attack_method` can be changed to `MadryEtAl`, `FastGradientMethod` or `MomentumIterativeMethod`. The flag `use_target` if `True` uses targeted attacks to train the model and if `False` uses untargeted attacks to train it.
### Inference:
