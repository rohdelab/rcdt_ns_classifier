from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class MNISTNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.pool = nn.AdaptiveAvgPool2d((12, 12))

        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x) # output size (N, 32, 26 26)
        x = F.relu(x)
        x = self.conv2(x) # output size (N, 64, 24, 24)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/sovrasov/flops-counter.pytorch/issues/16
model = MNISTNet(3, 10)
input = torch.randn(1, 3, 28, 28)
macs, params = profile(model, inputs=(input, ))
print('GMACs {}, GFLOPs {}'.format(macs/1e9, macs/2e9))

