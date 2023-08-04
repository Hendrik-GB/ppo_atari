import torch
from torch import nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNN(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3))
        self.norm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3))
        self.norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        self.norm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=3)
        self.lin = nn.Linear(in_features=2 * 2 * 64, out_features=128)

        self.actor = nn.Linear(128, out_dims)
        self.critic = nn.Linear(128, 1)

        self.actor_activation = nn.Sigmoid()
        self.activation = nn.Tanh()

    def action_only(self, x):
        x = self.forward(x)
        x = self.actor_activation(self.actor(x))
        return x

    def action_score(self, x):
        x = self.forward(x)
        action = self.actor_activation(self.actor(x))
        score = self.activation(self.critic(x))
        return action, score.squeeze()

    def forward(self, x):
        # unsqueeze without framestack
        x = x.unsqueeze(dim=-3)
        x = x / 255.0
        # print(torch.min(self.conv1.weight.data), torch.max(self.conv1.weight.data))
        x = self.pool1(self.norm1(self.conv1(x)))
        x = self.pool2(self.norm2(self.conv2(x)))
        x = self.pool3(self.norm3(self.conv3(x)))
        x = x.reshape((-1, 2 * 2 * 64))
        x = self.activation(self.lin(x))
        return x

