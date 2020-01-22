import os
import pickle
from functools import lru_cache
from datetime import datetime
from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@lru_cache()
def load_data():
    cache_file = 'data/customizedAffNIST.npz'
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(10):
            datafile = 'data/{}/dataORG_{}.mat'.format(split, classidx)
            data = loadmat(datafile)['xxO'].transpose([2, 0, 1])
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)

    x_train = x_train / x_train.max(axis=(1, 2), keepdims=True)
    x_test = x_test / x_test.max(axis=(1, 2), keepdims=True)

    x_train = (x_train * 255.).astype(np.uint8)
    x_test = (x_test * 255.).astype(np.uint8)

    np.savez(cache_file, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    return (x_train, y_train), (x_test, y_test)

def take_samples(data, index):
    data_reshape = data.reshape(10, -1, 3, 84, 84)
    sub = np.take(data_reshape, index, axis=1)
    return sub.reshape(-1, 3, 84, 84)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    # x_train shape: (class_idx*n_samples, 84, 84)
    x_train = (x_train.astype(np.float32) / 255. - 0.5) / 0.5
    x_train_3d = np.repeat(x_train.reshape(-1, 1, 84, 84), axis=1, repeats=3)

    x_test = (x_test.astype(np.float32) / 255. - 0.5) / 0.5
    x_test_3d = np.repeat(x_test.reshape(-1, 1, 84, 84), axis=1, repeats=3)
    x_test_3d = torch.from_numpy(x_test_3d).to(device)

    indices = loadmat('Ind_tr.mat')['indtr'] - 1 # index start from 1
    val_indices = indices[50:, 0]
    x_val = take_samples(x_train_3d, val_indices)
    x_val = torch.from_numpy(x_val).to(device)
    y_val = np.repeat(np.arange(10), val_indices.shape[0])

    model = models.vgg11_bn(num_classes=10).to(device)
    # torch.save(model.state_dict(), 'data/model_init.pth')

    for n_samples in range(5, 55, 5):
        for run in range(10):
            print('============== num samples {} run {} ============'.format(n_samples, run))
            model.load_state_dict(torch.load('data/model_init.pth'))

            sub_index = indices[:n_samples, run]
            x_train_sub = take_samples(x_train_3d, sub_index)
            y_train_sub = np.repeat(np.arange(10), n_samples)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

            save_path = 'results/samples-{}-model-{}/'.format(n_samples, type(model).__name__)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(save_path, 'run{}.pkl'.format(run))
            best_val_acc = 0.0

            for epoch in range(50):
                perm = np.random.permutation(x_train_sub.shape[0])
                x_train_sub_perm = x_train_sub[perm]
                y_train_sub_perm = y_train_sub[perm]

                # train
                for i in range(0, x_train_sub_perm.shape[0], args.batch_size):
                    inputs = x_train_sub_perm[i: i + args.batch_size]
                    targets = y_train_sub_perm[i: i + args.batch_size]
                    inputs = torch.from_numpy(inputs).to(device)
                    targets = torch.from_numpy(targets).to(device)
                    if inputs.shape[0] != args.batch_size:
                        break

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    _, max_indices = torch.max(outputs, 1)

                    train_acc = (max_indices == targets).type(torch.float).mean()

                    print('epoch {} iter {} train loss {:.5f} acc {:.5f}'.format(epoch, i//args.batch_size, loss.item(), train_acc))

                # validation
                with torch.no_grad():
                    val_acc = (np.argmax(model(x_val).cpu().numpy(), axis=1) == y_val).mean()
                    print('epoch {} val acc {:.5f}'.format(epoch, val_acc))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        state = dict(model=model.state_dict(), best_val_acc=val_acc, epoch=epoch)
                        torch.save(state, ckpt_path)
                        print('saved to ' + ckpt_path)
            # test
            with torch.no_grad():
                state = torch.load(ckpt_path)
                model.load_state_dict(state['model'])
                print('recovered from {}'.format(ckpt_path))
                print('samples {} run {} best val acc {}, epoch {}'.format(n_samples, run, state['best_val_acc'],
                                                                           state['epoch']), end=' ')
                logit = []
                for i in range(0, 10000, 2000):
                  test_logit = model(x_test_3d[i:i+2000])
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


    # plt.imshow(make_grid(torch.from_numpy(x_train[:16]).view(-1, 1, 84, 84), pad_value=1).permute(1, 2, 0))
    # plt.savefig('out.png')
    # plt.show()

