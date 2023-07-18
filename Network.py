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
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 16, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.ff = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x / 255.0
        x = self.network(x)
        return self.ff(x)
