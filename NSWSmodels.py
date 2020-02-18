#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:58:26 2020

@author: Hasnet, Xuwang Yin
"""

import numpy as np
import numpy.linalg as LA
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import multiprocessing as mp
from utils import *
import os
import h5py
from pathlib import Path

from pytranskit.optrans.continuous.radoncdt import RadonCDT

dataset = 'MNIST'  # 'MNIST'/'AffMNIST'/'OAM'/'SignMNIST'/'Synthetic'

if dataset in ['MNIST']:
    data_folder = '../DATA/data700'
    print(dataset + ': ' + data_folder)
    rm_edge = True
    num_classes = 10
    classes = range(num_classes)
    # classes = [1,5]
    po_max = 12  # maximum train samples = 2^po_max
elif dataset in ['AffMNIST']:
    data_folder = '../DATA/data701'
    print(dataset + ': ' + data_folder)
    rm_edge = True
    num_classes = 10
    classes = range(num_classes)
    # classes = [1,5]
    po_max = 12  # maximum train samples = 2^po_max
elif dataset in ['OAM']:
    data_folder = '../DATA/data705_s3'
    print(dataset + ': ' + data_folder)
    rm_edge = False
    num_classes = 32
    classes = range(num_classes)
    po_max = 9  # maximum train samples = 2^po_max
elif dataset in ['SignMNIST']:
    data_folder = '../DATA/data710'
    print(dataset + ': ' + data_folder)
    rm_edge = False
    num_classes = 3
    classes = range(num_classes)
    po_max = 10  # maximum train samples = 2^po_max
elif dataset in ['Synthetic']:
    data_folder = '../DATA/data699'
    print(dataset + ': ' + data_folder)
    rm_edge = True
    num_classes = 1000
    classes = range(num_classes)
    po_max = 7  # maximum train samples = 2^po_max

po = range(0, po_max + 1, 1)
tr_split = np.power(2, po)
N_exp = 1

eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]
Rdown = 4  # downsample radon projections (w.r.t. angles)
theta = np.linspace(0, 176, 180 / Rdown)
radoncdt = RadonCDT(theta)


def fun_rcdt_single(I):
    # I: (width, height)
    template = np.ones(I.shape, dtype=I.dtype)
    Ircdt = radoncdt.forward(x0_range, template / np.sum(template), x_range, I / np.sum(I), rm_edge)
    return Ircdt
    # return Ircdt.reshape([Ircdt.shape[0]*Ircdt.shape[1]],order='F')


def fun_rcdt_batch(data):
    # data: (n_samples, width, height)
    dataRCDT = [fun_rcdt_single(data[j, :, :] + eps) for j in range(data.shape[0])]
    return np.array(dataRCDT)


class SubSpaceClassifier:
    def __init__(self):
        self.num_classes = None
        self.subspaces = []

    def fit(self, X, y, num_classes):
        """Fit linear model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_proj, n_angles))
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        self :
            Returns an instance of self.
        """
        self.num_classes = num_classes
        for class_idx in range(num_classes):
            # generate the bases vectors
            # TODO check class_data is normalized (column mean = 0)
            class_data = X[y == class_idx]
            class_data_trans = add_trans_samples(class_data)
            flat = class_data_trans.reshape(class_data_trans.shape[0], -1)
            # print(flat.shape)
            u, s, vh = LA.svd(flat)
            # print(vh.shape, s.shape, u.shape)
            # Only use the largest 512 eigenvectors
            # Each row of basis is a eigenvector
            basis = vh[:flat.shape[0]][s > 1e-8][:512]
            self.subspaces.append(basis)

    def predict(self, X):
        """Predict using the linear model
        Parameters
        ----------
        X : array-like, sparse matrix, shape (n_samples, n_proj, n_angles))
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        X = X.reshape([X.shape[0], -1])
        D = []
        for class_idx in range(self.num_classes):
            basis = self.subspaces[class_idx]
            proj = X @ basis.T  # (n_samples, n_basis)
            projR = proj @ basis  # (n_samples, n_features)
            D.append(LA.norm(projR - X, axis=1))
        D = np.stack(D, axis=0)  # (num_classes, n_samples)
        preds = np.argmin(D, axis=0)  # n_samples
        return preds


def rcdt_parallel(X):
    # X: (n_samples, width, height)
    # template for RCDT

    # calc RCDT of test images
    print('Calculate RCDT of test images')
    splits = np.array_split(X, mp.cpu_count(), axis=0)
    pl = mp.Pool(mp.cpu_count())

    dataRCDT = pl.map(fun_rcdt_batch, splits)
    rcdt_features = np.vstack(dataRCDT)  # (n_samples, proj_len, num_angles)
    pl.close()
    pl.join()

    return rcdt_features


def add_trans_samples(rcdt_features):
    # rcdt_features: (n_samples, proj_len, num_angles)
    # deformation vectors for  translation
    Rdown = 4  # downsample radon projections (w.r.t. angles)
    theta = np.linspace(0, 176, 180 // Rdown)
    v1, v2 = np.cos(theta), np.sin(theta)
    v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
    v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
    return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])


if __name__ == '__main__':
    dataset = 'data700'
    img_size, num_classes = dataset_info(dataset)
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data('data700', num_classes=num_classes)
    cache_file = os.path.join(dataset, 'rcdt.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_traion {} x_test {}'.format(x_train.shape, x_test.shape))
    else:
        with h5py.File(cache_file, 'w') as f:
            x_train = rcdt_parallel(x_train)
            x_test = rcdt_parallel(x_test)
            f.create_dataset('x_train', data=x_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('x_test', data=x_test)
            f.create_dataset('y_test', data=y_test)
            print('saved to {}'.format(cache_file))

    num_repeats = 10
    accs = []
    all_preds = []
    for n_samples_perclass in [2 ** i for i in range(0, 2)]:
        for repeat in range(num_repeats):
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            classifier = SubSpaceClassifier()
            classifier.fit(x_train_sub, y_train_sub, num_classes)
            preds = classifier.predict(x_test)
            print('n_samples_perclass {} repeat {} acc {}'.format(n_samples_perclass, repeat, (preds == y_test).mean()))
            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
    accs = np.array(accs).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])
    results_dir = 'results/final/{}/'.format(dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'nsws.hdf5')
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)