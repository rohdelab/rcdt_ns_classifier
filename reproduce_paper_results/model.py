import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
# https://github.com/pytorch/examples/tree/master/mnist
    def __init__(self, input_channels, num_classes, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        if img_size == 28:
            self.pool = nn.MaxPool2d(2)
        else:
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