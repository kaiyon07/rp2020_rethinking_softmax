import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD

import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.4914, 0.4822, 0.4465 ], [ 0.2023, 0.1994, 0.2010 ], 

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


cifar10_train = dsets.CIFAR10(root='./data', train=True,
                                       download=True, transform=train_transform)
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=50,
                                         shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=50,
                                        shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def generate_opt_means(C, p, L): 
    """
    input
        C = constant value
        p = dimention of feature vector
        L = class number
    """
    opt_means = np.zeros((L, p))
    opt_means[0][0] = 1
    for i in range(1,L):
        for j in range(i): 
            opt_means[i][j] = - (1/(L-1) + np.dot(opt_means[i],opt_means[j])) / opt_means[j][j]
        opt_means[i][i] = np.sqrt(1 - np.linalg.norm(opt_means[i])**2)
    for k in range(L):
        opt_means[k] = C * opt_means[k]
        
    return opt_means
