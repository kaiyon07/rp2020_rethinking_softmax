import os
import sys
import numpy as np

import torch

from torchvision import models, datasets, transforms
import argparse
from scipy.io import loadmat

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--file", required=True,
                    help="The .mat file which has all the center coordinates.")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ])

kernel_dict = loadmat(args.file)
mean_logits_np = kernel_dict['mean_logits']
num_classes, num_of_features = np.shape(mean_logits_np)
if num_classes == 10:
    train_data = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform)
elif num_classes == 100:
    train_data = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False,
                                  download=True, transform=transform)
else:
    print("Wrong number of classes.")
    sys.exit()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,
                                           shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50,
                                          shuffle=False, num_workers=1)


class MMLDA_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        opt_means = mean_logits_np
        self.mean_expand = torch.tensor(
            opt_means).unsqueeze(0).double().to(device)

    def forward(self, x):
        b, p = x.shape
        x = (x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10)) * np.linalg.norm(
            mean_logits_np[0])  # Comment this line if you do not want to normalise the input.

        x_expand = x.repeat(1, num_classes).view(b, num_classes, p).double()

        logits = - torch.sum((x_expand - self.mean_expand)**2, dim=2)

        return logits


class dot_loss(torch.nn.Module):
    def __init__(self):
        super(dot_loss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = torch.nn.functional.one_hot(
            y_true, num_classes=y_pred.size(1)).double()
        loss = - torch.sum(y_pred * y_true, dim=1)
        return loss.mean()


# Training
def train(model, train_loader, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correctly_predicted = 0
    total = 0
    for _, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total += targets.size(0)
        train_loss += loss.item() * targets.size(0)
        _, predictions = outputs.max(1)
        correctly_predicted += predictions.eq(targets).sum().item()

    print('- Train Loss: %.4f, Acc: %.3f%%'
          % (train_loss/total, 100.*correctly_predicted/total))


def test(model, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correctly_predicted = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total += targets.size(0)
            test_loss += loss.item() * targets.size(0)
            _, predictions = outputs.max(1)
            correctly_predicted += predictions.eq(targets).sum().item()

        print('- Test Loss: %.4f, Acc: %.3f%%'
              % (test_loss/total, 100.*correctly_predicted/total))


model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(512, num_of_features)  # used_dense
model = torch.nn.Sequential(model, MMLDA_layer())
model = model.to(device)

# Criterion has to be changed if you want to perform SCE loss.
criterion = dot_loss()


def run(model, epoch_list, lr_list):
    for epochs, lr in zip(epoch_list, lr_list):
        for epoch in range(epochs):
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9)
            train(model, train_loader, criterion,
                  optimizer, epoch)
            test(model, test_loader, criterion,
                 epoch)


# run(model, [100,50,50],[0.01,0.001,0.0001])
# run(model, [50], [0.0001])
