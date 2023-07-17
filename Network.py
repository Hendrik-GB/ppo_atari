import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.linear1 = nn.Linear(7056, out_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        return x
