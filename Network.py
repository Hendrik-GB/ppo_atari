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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.flatten = nn.Flatten()
        self.ff = nn.Linear(512, out_dims)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x / 255.0
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.reshape((-1, 7 * 7 * 64))
        x = self.activation(self.lin(x))

        return self.ff(x)
