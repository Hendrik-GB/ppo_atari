import torch
from torch import nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNN(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 8, 3))
        self.norm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(3)

        self.conv2 = layer_init(nn.Conv2d(8, 16, 3))
        self.norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3)

        self.conv3 = layer_init(nn.Conv2d(16, 32, 3))
        self.norm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(3)

        self.flatten = nn.Flatten()

        self.ff = nn.Linear(128, out_dims)

    def forward(self, x):
        x = x / 255.0

        x = self.pool1(self.norm1(self.conv1(x)))
        x = self.pool2(self.norm2(self.conv2(x)))
        x = self.pool3(self.norm3(self.conv3(x)))
        x = self.flatten(x)
        return self.ff(x)
