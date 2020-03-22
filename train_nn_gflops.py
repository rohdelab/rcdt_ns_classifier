import os
import pickle
from functools import lru_cache
from datetime import datetime
from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from IPython.core.debugger import set_trace
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from utils import *
from model import MNISTNet
from sklearn.metrics import accuracy_score
# from cifar_models import resnet18
from pypapi import events, papi_high as high
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='shallowcnn', type=str, choices=['vgg11', 'shallowcnn', 'resnet18'])
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

if args.dataset == 'MNIST':
    assert args.model not in ['vgg11']

num_classes, img_size, po_train_max, _ = dataset_config(args.dataset)

# Adhoc code to make it works on AffMNIST
if args.dataset == 'AffMNIST':
    po_train_max = 7

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cpu'

if __name__ == '__main__':
    if args.model == 'vgg11':
        model = models.vgg11_bn(num_classes=num_classes).double().to(device)
    elif args.model == 'shallowcnn':
        model = MNISTNet(3, num_classes, img_size).double().to(device)
    if args.model == 'resnet18':
        model = models.resnet18(num_classes=num_classes).double().to(device)
    torch.save(model.state_dict(), './model_init.pth')

    model.eval()
    with torch.no_grad():
        high.start_counters([events.PAPI_DP_OPS,])
        x_test_batch = torch.rand(1, 3, img_size, img_size, dtype=torch.float64)
        test_logit = model(x_test_batch)
        test_gflops =high.stop_counters()[0] / 1e9
        print('test gflops: {}'.format(test_gflops))

    all_train_gflops = []
    for n_samples_perclass in [2**i for i in range(0, po_train_max+1)]:
        model.load_state_dict(torch.load('./model_init.pth'))
        x_val_size = 0 if n_samples_perclass < 16 else int(n_samples_perclass*0.1) * num_classes
        x_train_sub_size = n_samples_perclass * num_classes - x_val_size

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)

        high.start_counters([events.PAPI_DP_OPS,])
        # train
        model.train()
        for i in range(0, x_train_sub_size, args.batch_size):
            inputs = torch.rand(args.batch_size, 3, img_size, img_size, dtype=torch.float64)
            targets = torch.randint(low=0, high=num_classes, size=(args.batch_size,))
              
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # validation
        if x_val_size > 0:
            model.eval()
            with torch.no_grad():
                for i in range(0, x_val_size, 10):
                  x_val_batch = torch.rand(10, 3, img_size, img_size, dtype=torch.float64)
                  batch_logit = model(x_val_batch)

        train_gflops = high.stop_counters()[0] / 1e9
        print('============== perclass samples {} x_train_sub_size {} x_val_size {} train_gflops {} ============'.format(n_samples_perclass, x_train_sub_size, x_val_size, train_gflops))
        all_train_gflops.append(train_gflops)

    results_dir = 'results/final_gflops/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'nn_{}.hdf5'.format(args.model))
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('train_gflops', data=np.array(all_train_gflops))
        f.create_dataset('test_gflops', data=test_gflops)
    print('saved to ' + result_file)
