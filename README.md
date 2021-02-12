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
```shell
python isometric_args.py --var=10 --dim_dense=256 --num_class=100 --lr=0.0001 --steps=100000
```
`var`, `dim_dense`, `num_class` are the usual parameters, `lr` is learning and `steps` is the number of the optimization steps.

- **Distorted MMC Centers (DMMC):** DMMC is also a variant of MMC center loss, where two centers can be brought closer to each other and vice versa depending on the similarity index value. Implementation can be found under `center_gen_python/joint_mmc.py`. 
 ```shell
python joint_mmc_args.py --var=10 --dim_dense=256 --num_class=10 --sim=0.1 --class1=4 --class2=6 --alpha1=0.4 --alpha2=0.6  
```
`var`, `dim_dense`, `num_class` are the usual parameters, `class1` and `class2` are the indices of the class whose distance are being manipulated. `alpha1` and `alpha2` are the corresponding weights of class and `sim` is the similarity value between `class1` and `class2`.

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
Models can be trained adversarially with SCE loss by the following commands:
```shell
python advtrain.py --batch_size=50 --dataset=[dataset] --optimizer='mom' --lr=0.01 --version=2 --adv_ratio=1.0 --use_MMLDA=False --use_target=False --attack_method='MadryEtAl' --use_BN=True --use_random=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author. The flag `use_ball` can be changed to `False` if you want to train the model on MMLDA loss.
The `meanvar` parameter can be used to change the distance between the optimal centers.
 The parameter `attack_method` can be changed to `MadryEtAl`, `FastGradientMethod` or `MomentumIterativeMethod`. The flag `use_target` if `True` uses targeted attacks to train the model and if `False` uses untargeted attacks to train it.

- **Training the models adversarially using MMC loss**
Models can be trained adversarially with MMC loss by the following commands:
```shell
python advtrain.py --batch_size=50 --mean_var=10 --dataset=[dataset] --optimizer='mom' --lr=0.01 --version=2 --adv_ratio=1.0 --use_MMLDA=True --use_ball=True --use_target=True --attack_method='MadryEtAl' --use_BN=True --use_random=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author. The parameter `attack_method` can be changed to `MadryEtAl`, `FastGradientMethod` or `MomentumIterativeMethod`. The flag `use_target` if `True` uses targeted attacks to train the model and if `False` uses untargeted attacks to train it.
### Inference:
We have written a test.py file to evaluate the models seperately after the training is done.
- **Evaluation of trained models**
After training, weights of the trained models are saved for future use. These weights can be used to evaluate the models.
```shell
python test.py --batch_size=50 --mean_var=10 --dataset=[dataset] -optimizer='mom' --lr=0.01 --version=2 --use_MMLDA=True --use_ball=True --use_BN=True --use_random=False --use_dense=True --use_leaky=False --load_weights=[location]
```
All the parameters are same as used in training commands. A new parameter `load_weights` is given the weights of the model that needs to be evaluated. The parameters should be same as the ones used to train the model.

- **White-box L-infinity attack (PGD)**
Evaluation of models on these attacks can be done using this command 
```shell
python advtest_iterative.py --batch_size=50 --attack_method='MadryEtAl' --attack_method_for_advtrain=None --dataset=[dataset] --target=True --num_iter=10 --use_ball=True --use_MMLDA=True --use_advtrain=False --epoch=[epoch] --use_BN=True --normalize_output_for_ball=False --use_random=False --use_target=False
```
Here `dataset` can be either `mnist`, `cifar10` or `cifar100` as done by original author. The parameter `attack_method` can be changed to `MadryEtAl`, `FastGradientMethod`, `BasicIterativeMethod` or `MomentumIterativeMethod`. 
The parameter The parameter `use_advtrain` should be set to False if attacks are to be done on models trained with standard SCE or MMC loss and True if the attacks are to be done on adversarially trained models. 

The parameter `attack_method_for_advtrain` would be changed according to the attack that was used to train the said model. The `target` indicates whether use targeted or untargeted attack. The parameter
`normalize_output_for_ball` is a bool flag to decide whether apply a softmax function to return predictions in the inference phase.

**Note:** The authors of the original paper have used cleverhans: 2.1.0 to perform their adaptive attack experiments. This requires one to change some code in the installed repo itself. The details of the changes are mentioned in the original paper and the original repo itself.

- **White-box L-2 attack (C&W)**
The command used to evaluate the models on these attacks is given below
```shell
python advtest_others.py --mean_var=10 --batch_size=50 --attack_method='CarliniWagnerL2' --attack_method_for_advtrain=None --dataset=[dataset] --target=True --use_ball=True --use_MMLDA=True --use_advtrain=False --adv_ratio=1.0 --use_target=False --epoch=[epoch] --use_BN=True --normalize_output_for_ball=False --use_random=False --use_dense=True --use_leaky=False --CW_confidence=0.
```
The parameters here can be used similarly to the parameters used in above subsection.
The attack_method could also be `ElasticNetMethod` to perform EAD attack.

- **Black-box transfer-based attack (MIM & PGD)**
The command used to evaluate the models on these attacks is given below
```shell
python advtest_iterative_blackbox.py --batch_size=50 --optimizer='Adam' --attack_method='MadryEtAl' --dataset=[dataset] --target=False --num_iter=10 --use_random=False --use_dense=True --use_leaky=False --epoch=[epoch] --use_BN=True --model_1='AT-MMC-100' --model_2='SCE'
```
For black box attacks, a substitute model is used to create examples. The parameter `model_1` is the model that is used to craft the adversarial examples while `model_2` refers to the model that the attacks is done on. Both the parameters take similar values and can take the values `SCE`, `MMC-10`, `MMC-100`, `AT-SCE`, `AT-MMC-10`, `AT-MMC-100`.

- **Black-box gradient-free attack (SPSA)**
The command used to evaluate the models on these attacks is given below
```shell
python advtest_others.py --mean_var=10 --batch_size=50 --attack_method='SPSA' --attack_method_for_advtrain=None --dataset=[dataset] --target=False --use_ball=True --use_MMLDA=True --use_advtrain=False --adv_ratio=1.0 --use_target=False --epoch=[epoch] --use_BN=True -normalize_output_for_ball=False --use_random=False --use_dense=True --use_leaky=False --SPSA_epsilon=8
```

- **General-purpose attack**
The command used to evaluate the models on these attacks is given below
```shell
python advtest_simple_transform.py --mean_var=10 --batch_size=50  --attack_method='Rotation' --attack_method_for_advtrain='MadryEtAl' --dataset=[dataset] --use_ball=True --use_MMLDA=True --use_advtrain=True --epoch=[epoch] --adv_ratio=1.0 --use_target=False --normalize_output_for_ball=False
```
These methods are there to check the general robustness of the models to transformations such as Gaussian noise and rotations in the input.

The `attack_method` could be 'Rotation' for rotation transformation or 'Gaussian' for Gaussian noise.