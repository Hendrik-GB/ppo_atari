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
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, out_dims), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def action_only(self, x):
        x = self.network(x)
        x = self.actor_activation(self.actor(x))
        return x

    def action_score(self, x):
        x = self.network(x)
        action = self.actor(x)
        score = self.critic(x)
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

