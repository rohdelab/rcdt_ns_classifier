import pickle
from torchvision.utils import make_grid
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import skorch
import skorch.dataset
from skorch import NeuralNetClassifier
from resnet import *

from model import MNISTNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', type=str, default='OASIS1_age')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='shallowcnn', type=str, choices=['vgg11', 'shallowcnn', 'resnet18'])
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

if args.dataset == 'MNIST':
    assert args.model not in ['vgg11']

num_classes, img_size, po_train_max, _ = dataset_config(args.dataset)

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make3dnet(input_channels):
    layers = [
        nn.Conv3d(input_channels, 16, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),

        nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),

        nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),

        nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),

        nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),

        nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1),
        nn.BatchNorm3d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),
    ]
    return nn.Sequential(*layers)

class Net3D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.features = make3dnet(input_channels)
        self.classifier = nn.Sequential(nn.Linear(128, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    assert args.dataset == 'OASIS1_age'
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes)
    # x_train shape: (class_idx*n_samples_perclass, args.img_size, args.img_size)
    x_train = (x_train.astype(np.float32) / 255. - 0.5) / 0.5
    x_test = (x_test.astype(np.float32) / 255. - 0.5) / 0.5
    x_train = x_train[:, np.newaxis, ...]
    x_test = x_test[:, np.newaxis, ...]
    print('loaded data, x_train.shape {}, x_test.shape {}'.format(x_train.shape, x_test.shape))

    for n_samples_perclass in [1, 2, 4, 8, 16, 32, 64]:
        x_train_partial = np.concatenate([x_train[y_train==0][:n_samples_perclass], x_train[y_train==1][:n_samples_perclass]])
        y_train_partial = np.concatenate([y_train[y_train==0][:n_samples_perclass], y_train[y_train==1][:n_samples_perclass]])

        print('train samples {} test samples {}'.format(x_train_partial.shape[0], x_test.shape[0]))

        for repeat in range(10):
            # model = MNISTNet(x_train.shape[1], num_classes, img_size, with_softmax=True)
            # model = Net3D(1, num_classes=2)
            model = resnet10(num_classes=2)

            train_split = None
            # if n_samples_perclass >= 16:
            #     train_split = skorch.dataset.CVSplit(10, stratified=True)
            classifier = NeuralNetClassifier(
                model,
                max_epochs=20,
                lr=5e-4,
                optimizer=torch.optim.SGD,
                optimizer__momentum=0.9,
                iterator_train__shuffle=True,
                device='cuda',
                train_split=train_split,
                batch_size=5
            )

            classifier.fit(x_train_partial, y_train_partial)
            preds = classifier.predict(x_test)
            acc = accuracy_score(y_test, preds)
            print('samples per class {} repeat {} test acc {}'.format(n_samples_perclass, repeat, acc))
