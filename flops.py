from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from thop import profile
from model import MNISTNet
import numpy as np

# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/sovrasov/flops-counter.pytorch/issues/16

def test_gflops(model, epochs=1, num_train_samples=1, input_size=28):
    gflops = epochs * num_train_samples * 2 * test_gflops(model, 1, input_size)
    return gflops


def test_gflops(model, input_size):
    assert model in ['shallowcnn', 'resnet18', 'vgg11']
    if model == 'shallowcnn':
        model = MNISTNet(3, 10, img_size=input_size)
    if model == 'resnet18':
        model = models.resnet18(num_classes=10)
    if model == 'vgg11':
        model = models.vgg11_bn(num_classes=10)
    input = torch.randn(1, 3, input_size, input_size)
    macs, params = profile(model, inputs=(input, ))
    gflops = macs/2e9
    return gflops

costs = []
for model in ['shallowcnn', 'resnet18', 'vgg11']:
    model_costs = []
    for img_size in [28, 32, 64, 128]:
        print('model {} input size {}'.format(model, img_size))
        if model == 'vgg11' and img_size == 28:
            model_costs.append(0)
        else:
            model_costs.append(test_gflops(model, img_size))
    costs.append(model_costs)
costs = np.stack(costs, axis=0)
print(costs.shape)



barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(costs[0]))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

rect = plt.bar(r1, costs[0], color='tab:red', width=barWidth, edgecolor='white', label='shallowcnn')
# height = rect.get_height()
# ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')
rect = plt.bar(r2, costs[1], color='tab:green', width=barWidth, edgecolor='white', label='resnet18')
rect = plt.bar(r3, costs[2], color='tab:blue', width=barWidth, edgecolor='white', label='vgg11')

# Add xticks on the middle of the group bars
plt.ylabel('GFLOPS', fontweight='bold')
plt.xlabel('input size', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(costs[0]))], ['28x28', '32x32', '64x64', '128x128'])

# Create legend & Show graphic
plt.legend()
plt.title('Single Input Test GFLOPS of Nueral Networks Models')
plt.show()

