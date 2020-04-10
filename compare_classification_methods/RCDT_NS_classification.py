#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:58:26 2020

@author: Hasnat, Xuwang Yin
"""

import argparse
import multiprocessing as mp
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as LA

from pytranskit.optrans.continuous.radoncdt import RadonCDT
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils import *
from pytranskit.classification.rcdt_ns import RCDT_NS

import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--use_image_feature', action='store_true')
parser.add_argument('--no_deform_model', action='store_true')
parser.add_argument('--classifier', default='subspace', choices=['mlp', 'subspace'])
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--count_flops', action='store_true')
args = parser.parse_args()


if args.count_flops:
   from pypapi import events, papi_high as high

if args.classifier == 'mlp':
    assert not args.use_gpu
    import torch
    import skorch
    from torch import nn
    from skorch import NeuralNetClassifier

if args.use_gpu:
    import cupy as cp

num_classes, img_size, po_train_max, rm_edge = dataset_config(args.dataset)

eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]
Rdown = 4  # downsample radon projections (w.r.t. angles)
theta = np.linspace(0, 176, 180 // Rdown)
radoncdt = RadonCDT(theta)

def fun_rcdt_single(I):
    # I: (width, height)
    template = np.ones(I.shape, dtype=I.dtype)
    Ircdt = radoncdt.forward(x0_range, template / np.sum(template), x_range, I / np.sum(I), rm_edge)
    return Ircdt

def fun_rcdt_batch(data):
    # data: (n_samples, width, height)
    dataRCDT = [fun_rcdt_single(data[j, :, :] + eps) for j in range(data.shape[0])]
    return np.array(dataRCDT)

def rcdt_parallel(X):
    # X: (n_samples, width, height)
    # template for RCDT

    # calc RCDT of test images
    print('Calculating RCDT ...')
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
    v1, v2 = np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)
    v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
    v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
    return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])



if __name__ == '__main__':
    datadir = '../data'
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes, datadir)

    if args.count_flops:
        for n_samples in [1, 10, 100]:
            high.start_counters([events.PAPI_DP_OPS,])
            rcdt_test = x_train[:n_samples]
            rcdt_test = fun_rcdt_batch(rcdt_test)
            x=high.stop_counters()[0]
            print('rcdt_test.shape {} GFLOPS {}'.format(rcdt_test.shape, x/1e9))
            rcdt_gflops = (x / 1e9) / n_samples
        print('rcdt_gflops: {}'.format(rcdt_gflops))

    if not args.use_image_feature:
        cache_file = os.path.join(datadir, args.dataset, 'rcdt.hdf5')
        if os.path.exists(cache_file):
            with h5py.File(cache_file, 'r') as f:
                x_train, y_train = f['x_train'][()], f['y_train'][()]
                x_test, y_test = f['x_test'][()], f['y_test'][()]
                x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
                print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
                # Adhoc code for flops counting on AffMNIST dataset
                if args.count_flops and args.dataset == 'AffMNIST':
                    from sklearn.model_selection import train_test_split
                    _, x_train, _, y_train = train_test_split(x_train, y_train, test_size=1100*10, random_state=42, stratify=y_train)
                    print(x_train.shape, y_train.shape)
                    print([(y_train == i).sum() for i in range(10)])
                    po_train_max = 10 # up to 1024 samples per class
        else:
            with h5py.File(cache_file, 'w') as f:
                x_train = rcdt_parallel(x_train)
                x_test = rcdt_parallel(x_test)
                x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
                f.create_dataset('x_train', data=x_train)
                f.create_dataset('y_train', data=y_train)
                f.create_dataset('x_test', data=x_test)
                f.create_dataset('y_test', data=y_test)
                print('saved to {}'.format(cache_file))

    num_repeats = 10
    accs = []
    all_preds = []
    if args.use_gpu:
        x_test = cp.array(x_test)

    if args.count_flops:
       all_train_gflops, all_test_gflops = [], []
    for n_samples_perclass in [2 ** i for i in range(0, po_train_max+1)]:
        for repeat in range(num_repeats):
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            if args.classifier == 'subspace':
                classifier = RCDT_NS(theta=theta, 
                                     no_deform_model=args.no_deform_model, 
                                     use_image_feature=args.use_image_feature, 
                                     count_flops=args.count_flops,
                                     use_gpu=args.use_gpu)
                tic = time.time()
                if args.count_flops:
                    x_train_sub = x_train_sub.astype(np.float64)
                    x_test = x_test.astype(np.float64)
                    high.start_counters([events.PAPI_DP_OPS,])
                    classifier.fit(x_train_sub, y_train_sub, num_classes)
                    train_gflops = high.stop_counters()[0] / 1e9 + n_samples_perclass * num_classes * rcdt_gflops

                    high.start_counters([events.PAPI_DP_OPS,])
                    preds = classifier.predict(x_test)
                    test_gflops = (high.stop_counters()[0] / 1e9) / x_test.shape[0] + rcdt_gflops

                    all_train_gflops.append(train_gflops)
                    all_test_gflops.append(test_gflops)
                else:
                    classifier.fit(x_train_sub, y_train_sub, num_classes)
                    preds = classifier.predict(x_test)
            else:
                tic = time.time()
                x_train_sub_flat = x_train_sub.reshape(x_train_sub.shape[0], -1)
                x_test_flat = x_test.reshape(x_test.shape[0], -1)
                
                module = nn.Sequential(nn.Linear(x_train_sub_flat.shape[1], 500), nn.ReLU(), nn.Linear(500, num_classes), nn.Softmax(dim=1))
                classifier = NeuralNetClassifier(
                    module,
                    max_epochs=500,
                    lr=5e-4,
                    optimizer=torch.optim.Adam,
                    iterator_train__shuffle=True,
                    train_split=None,
                    device='cuda',
                    verbose=0
                )
                classifier.fit(x_train_sub_flat, y_train_sub)
                preds = classifier.predict(x_test_flat)
            toc = time.time()
            # print('Runtime of fit+predict functions: {} seconds'.format(toc-tic))
            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
            print('n_samples_perclass {} repeat {} acc {}'.format(n_samples_perclass, repeat, accuracy_score(y_test, preds)))
            if args.count_flops:
                print('train GFLOPS {:.5f} test GFLOPS {:.5f}'.format(train_gflops, test_gflops))

    accs = np.array(accs).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])

    results_dir = 'results/final/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if args.use_image_feature:
        result_file = os.path.join(results_dir, 'nsws_image.hdf5')
    else:
        if args.classifier == 'mlp':
            result_file = os.path.join(results_dir, 'nsws_{}.hdf5'.format(args.classifier))
        else:
            result_file = os.path.join(results_dir, 'nsws.hdf5')
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)
        if args.count_flops:
            train_gflops = np.array(all_train_gflops).reshape(-1, num_repeats)
            test_gflops = np.array(all_test_gflops).reshape(-1, num_repeats)
            f.create_dataset('train_gflops', data=train_gflops)
            f.create_dataset('test_gflops', data=test_gflops)
