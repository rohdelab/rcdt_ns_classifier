#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:58:26 2020

@author: Hasnet, Xuwang Yin
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from pypapi import events, papi_high as high


from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as LA

from pytranskit.optrans.continuous.radoncdt import RadonCDT
from utils import *

import time

__GPU__ = False

if __GPU__:
    import cupy as cp
    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

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
    print('Calculating RCDT cpu count {}...'.format(mp.cpu_count()))
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
    assert rcdt_features.dtype == np.float64
    v1, v2 = np.cos(theta*np.pi/180).astype(np.float64), np.sin(theta*np.pi/180).astype(np.float64)
    v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
    v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
    return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])


class SubSpaceClassifier:
    def __init__(self):
        self.num_classes = None
        self.subspaces = []
        self.len_subspace = 0

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
        assert X.dtype == np.float64
        self.num_classes = num_classes
        for class_idx in range(num_classes):
            # generate the bases vectors
            # TODO check class_data is normalized (column mean = 0)
            class_data = X[y == class_idx]
            class_data_trans = add_trans_samples(class_data)
            flat = class_data_trans.reshape(class_data_trans.shape[0], -1)
            # flat = class_data.reshape(class_data.shape[0], -1)
            
            u, s, vh = LA.svd(flat)
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            #max_basis = max(np.where(cum_s<0.99)[0])+2
            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            # print('# basis with atleast 99% variance: '+str(max_basis))
            # print('singular values: ' +str(s))

            if max_basis > self.len_subspace:
                self.len_subspace = max_basis

            basis = vh[:flat.shape[0]]
            assert basis.dtype == np.float64
            
            #if __GPU__:
                #basis = cp.array(basis)
                #using SVD
                #u, s, vh = cp.linalg.svd(cp.array(flat),full_matrices=False)
                #basis = vh[:flat.shape[0]][s > 1e-8][:512]
                
                #using Gram-Schmidt Ortho-Normalization
                #vh = gs_orthogonalization(cp.array(flat))
                #vh = vh/cp.linalg.norm(vh,axis=1).reshape(vh.shape[0],1)
                #basis = vh[:flat.shape[0]][:512]
            # else:
                # using SVD
                # u, s, vh = LA.svd(flat)
                # basis = vh[:flat.shape[0]][s > 1e-8][:512]
                
                # using Gram-Schmidt Ortho-Normalization
                # vh = gs_orthogonalization(flat)
                # vh = vh/LA.norm(vh,axis=1).reshape(vh.shape[0],1)
                # basis = vh[:flat.shape[0]][:512]

            # Only use the largest 512 eigenvectors
            # Each row of basis is a eigenvector
            
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
            basis = basis[:self.len_subspace,:]
            
            if __GPU__:
                D.append(cp.linalg.norm(cp.matmul(cp.matmul(X, cp.array(basis).T), cp.array(basis)) -X, axis=1))
                #basis = cp.array(basis)
                #proj = cp.matmul(X,basis.T)
                #projR = cp.matmul(proj,basis)
                #D.append(cp.linalg.norm(projR - X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if __GPU__:
            preds = cp.argmin(cp.stack(D, axis=0), axis=0)
            #D = cp.stack(D, axis=0)  # (num_classes, n_samples)
            #preds = cp.argmin(D, axis=0)  # n_samples
            return cp.asnumpy(preds)
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            return preds


if __name__ == '__main__':
    datadir = './data'
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes, datadir)
    
    for n_samples in [1, 10, 100]:
        high.start_counters([events.PAPI_DP_OPS,])
        rcdt_test = x_train[:n_samples]
        rcdt_test = fun_rcdt_batch(rcdt_test)
        x=high.stop_counters()[0]
        print('rcdt_test.shape {} GFLOPS {}'.format(rcdt_test.shape, x/1e9))
        rcdt_gflops = (x / 1e9) / n_samples
    print('rcdt_gflops: {}'.format(rcdt_gflops))

    high.start_counters([events.PAPI_DP_OPS,])
    rcdt_test = rcdt_parallel(rcdt_test)
    x=high.stop_counters()[0]
    print('rcdt_test.shape {} GFLOPS {} (parallel)'.format(rcdt_test.shape, x/1e9))

    cache_file = os.path.join(datadir, args.dataset, 'rcdt.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
            # Adhoc code for AffMNIST
            if args.dataset == 'AffMNIST':
                from sklearn.model_selection import train_test_split
                _, x_train, _, y_train = train_test_split(x_train, y_train, test_size=1100*10, random_state=42, stratify=y_train)
                print(x_train.shape, y_train.shape)
                print([(y_train == i).sum() for i in range(10)])
                po_train_max = 10 # up to 1024 samples per class
    else:
        with h5py.File(cache_file, 'w') as f:
            x_train = rcdt_parallel(x_train)
            x_test = rcdt_parallel(x_test)
            f.create_dataset('x_train', data=x_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('x_test', data=x_test)
            f.create_dataset('y_test', data=y_test)
            print('saved to {}'.format(cache_file))

    num_repeats = 1
    accs = []
    all_preds = []
    all_train_gflops, all_test_gflops = [], []
    if __GPU__:
        x_test = cp.array(x_test)
    for n_samples_perclass in [2 ** i for i in range(0, po_train_max+1)]:
        for repeat in range(num_repeats):
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            classifier = SubSpaceClassifier()
            tic = time.time()

            high.start_counters([events.PAPI_DP_OPS,])
            classifier.fit(x_train_sub.astype(np.float64), y_train_sub, num_classes)
            train_gflops = high.stop_counters()[0] / 1e9 + n_samples_perclass * num_classes * rcdt_gflops
            
            high.start_counters([events.PAPI_DP_OPS,])
            preds = classifier.predict(x_test.astype(np.float64))
            test_gflops = (high.stop_counters()[0] / 1e9) / x_test.shape[0] + rcdt_gflops

            toc = time.time()

            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
            all_train_gflops.append(train_gflops)
            all_test_gflops.append(test_gflops)

            print('n_samples_perclass {} repeat {} acc {} train GFLOPS {:.5f} test GFLOPS {:.5f} train+predict time {} seconds'.format(n_samples_perclass, repeat, accuracy_score(y_test, preds), train_gflops, test_gflops, toc-tic))

    accs = np.array(accs).reshape(-1, num_repeats)
    train_gflops = np.array(all_train_gflops).reshape(-1, num_repeats)
    test_gflops = np.array(all_test_gflops).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])

    results_dir = 'results/final_gflops/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'nsws.hdf5')
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)
        f.create_dataset('train_gflops', data=train_gflops)
        f.create_dataset('test_gflops', data=test_gflops)
