import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.conv1_0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.norm1 = nn.BatchNorm2d(num_features=4)

        self.conv2_0 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.norm2 = nn.BatchNorm2d(num_features=8)

        self.conv3_0 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=3)
        self.norm3 = nn.BatchNorm2d(num_features=16)

        self.ff = nn.Linear(64, out_dims)

    def forward(self, x):
        x = self.norm1(self.pool1(self.conv1_0(x)))
        x = self.norm2(self.pool2(self.conv2_0(x)))
        x = self.norm3(self.pool3(self.conv3_0(x)))
        x = torch.flatten(x, start_dim=1, end_dim=3)

        x = self.ff(x)
        return x
