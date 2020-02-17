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
from scipy.io import loadmat
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
# from cifar_models import resnet18


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', type=str, choices=['data699', 'data711', 'data705_s3', 'data705_s3_t10', 'data704', 'data701', 'data700', 'data706', 'data703', 'data701_rot', 'data707', 'data707_hog', 'data708', 'data709', 'data710', 'data710_full'], required=True)
parser.add_argument('--img_size', default=84, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--model', default='vgg11', type=str, choices=['vgg11', 'shallowcnn', 'resnet18'])
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

if args.dataset in ['data700', 'data704']:
    args.img_size = 28
if args.dataset in ['data701', 'data701_rot']:
    args.img_size = 84
if args.dataset in ['data705_s3_t10', 'data705_s3']:
    args.img_size = 151
    args.num_classes = 32
if args.dataset == 'data706':
    args.img_size = 64
    args.num_classes = 6
if args.dataset == 'data703':
    args.img_size = 130
    args.num_classes = 2
if args.dataset in ['data707', 'data707_hog']:
    args.img_size = 128
    args.num_classes = 5
if args.dataset == 'data708':
    args.img_size = 120
    args.num_classes = 2
if args.dataset == 'data709':
    args.img_size = 32
    args.num_classes = 4
if args.dataset == 'data710':
    args.img_size = 128
    args.num_classes = 3
if args.dataset == 'data710_full':
    args.img_size = 128
    args.num_classes = 24
if args.dataset == 'data711':
    args.img_size = 64
    args.num_classes = 10
if args.dataset == 'data699':
    args.img_size = 128
    args.num_classes = 1000

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MNISTNet(nn.Module):
# https://github.com/pytorch/examples/tree/master/mnist
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        if args.img_size == 28:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AdaptiveAvgPool2d((12, 12))

        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

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



if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data_3D(args.dataset)
    print('loaded data, x_train.shape {}, x_test.shape {}'.format(x_train.shape, x_test.shape))
    if args.plot:
        fig, axes = plt.subplots(nrows=args.num_classes, ncols=1)
        for k in range(args.num_classes):
            class_data = x_train[y_train == k][:64]
            class_data = class_data.reshape(class_data.shape[0], 3, *class_data.shape[2:])
            axes[k].imshow(make_grid(torch.from_numpy(class_data), nrow=16, pad_value=1).permute(1, 2, 0))
        plt.savefig('samples.pdf', dpi=400)
        plt.show()


    # x_train shape: (class_idx*n_samples_perclass, args.img_size, args.img_size)
    x_train = (x_train.astype(np.float32) / 255. - 0.5) / 0.5
    x_test = (x_test.astype(np.float32) / 255. - 0.5) / 0.5

    if args.model == 'vgg11':
        model = models.vgg11_bn(num_classes=args.num_classes).to(device)
    elif args.model == 'shallowcnn':
        model = MNISTNet(input_channels=3).to(device)
    if args.model == 'resnet18':
        model = models.resnet18(num_classes=args.num_classes).to(device)
    torch.save(model.state_dict(), './model_init.pth')

    for n_samples_perclass in [2**i for i in range(1, 13)]:
    # for n_samples_perclass in [512]:
        for repeat in range(5):
            print('============== num samples {} repeat {} ============'.format(n_samples_perclass, repeat))
            model.load_state_dict(torch.load('./model_init.pth'))
            (x_train_sub, y_train_sub), (x_val, y_val) = train_val_split(x_train, y_train, n_samples_perclass, args.num_classes, repeat)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=5e-4)

            save_path = 'results-new-validation/{}/samples-{}-model-{}/'.format(args.dataset, n_samples_perclass, type(model).__name__)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(save_path, 'repeat{}.pkl'.format(repeat))
            best_val_acc = 0.0

            for epoch in range(args.epochs):
                perm = np.random.permutation(x_train_sub.shape[0])
                x_train_sub_perm = x_train_sub[perm]
                y_train_sub_perm = y_train_sub[perm]

                # train
                model.train()
                for i in range(0, x_train_sub_perm.shape[0], args.batch_size):
                    inputs = x_train_sub_perm[i: i + args.batch_size]
                    targets = y_train_sub_perm[i: i + args.batch_size]
                    inputs = torch.from_numpy(inputs).to(device)
                    targets = torch.from_numpy(targets).to(device)
                    # if inputs.shape[0] != args.batch_size:
                    #     break

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    _, max_indices = torch.max(outputs, 1)

                    train_acc = (max_indices == targets).type(torch.float).mean()
                    if (i//args.batch_size) % 10 == 0:
                        print('epoch {} iter {} train loss {:.5f} acc {:.5f}'.format(epoch, i//args.batch_size, loss.item(), train_acc))

                # validation
                if x_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_logits = []
                        for i in range(0, x_val.shape[0], 100):
                          batch_logit = model(x_val[i:i+100])
                          val_logits.append(batch_logit.cpu().numpy())
                        val_logits = np.concatenate(val_logits)
                        val_acc = (np.argmax(val_logits, axis=1) == y_val).mean()
                        print('epoch {} val acc {:.5f}'.format(epoch, val_acc))
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            state = dict(model=model.state_dict(), best_val_acc=val_acc, epoch=epoch)
                            torch.save(state, ckpt_path)
                            print('saved to ' + ckpt_path)
                else:
                    state = dict(model=model.state_dict(), best_val_acc=-1, epoch=epoch)
                    torch.save(state, ckpt_path)
                    print('saved to ' + ckpt_path)
            # test
            model.eval()
            with torch.no_grad():
                state = torch.load(ckpt_path)
                model.load_state_dict(state['model'])
                print('recovered from {}'.format(ckpt_path))
                print('samples {} repeat {} best val acc {}, epoch {}'.format(n_samples_perclass, repeat, state['best_val_acc'],
                                                                           state['epoch']), end=' ')
                logit = []
                for i in range(0, x_test.shape[0], 100):
                  x_test_batch = torch.from_numpy(x_test[i:i+100]).to(device)
                  test_logit = model(x_test_batch)
                  logit.append(test_logit.cpu().numpy())
                logit = np.concatenate(logit)
                y_pred = np.argmax(logit, axis=1)
                test_acc = (y_pred == y_test).mean()
                del state['model']
                state['test_acc'] = test_acc
                state['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                print('test acc {:.5f}'.format(test_acc))
                print(state['confusion_matrix'])
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(state, f)
                # # to load the data
                # with open(ckpt_path, 'rb') as f:
                #     state = pickle.load(f)

                print('saved to {}'.format(ckpt_path))


    # plt.imshow(make_grid(torch.from_numpy(x_train[:16]).view(-1, 1, args.img_size, args.img_size), pad_value=1).permute(1, 2, 0))
    # plt.savefig('out.png')
    # plt.show()

