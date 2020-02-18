from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from thop import profile
from model import MNISTNet

# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/sovrasov/flops-counter.pytorch/issues/16

def gflops(model, epochs=50, num_train_samples=1, input_size=28):
    assert model in ['shallowcnn', 'resnet18', 'vgg11']
    if model == 'shallowcnn':
        model = MNISTNet(3, 10, img_size=28)
    if model == 'resnet18':
        model = models.resnet18(num_classes=10)
    if model == 'vgg11':
        model = models.vgg11_bn(num_classes=10)
    input = torch.randn(1, 3, input_size, input_size)
    macs, params = profile(model, inputs=(input, ))
    gflops = epochs * num_train_samples * 2 * macs/2e9
    return gflops
    # print('GMACs {}, GFLOPs {}'.format(macs/1e9, macs/2e9))

print(gflops('resnet18'))

