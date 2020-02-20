#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:58:26 2020

@author: Hasnet, Xuwang Yin
"""

import argparse
import multiprocessing as mp
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as LA

from pytranskit.optrans.continuous.radoncdt import RadonCDT
from utils import *

import time

__GPU__ = True

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


def gs_cofficient(v1, v2):
    if __GPU__:
        return cp.dot(v2, v1) / cp.dot(v1, v1)
    else:
        return np.dot(v2, v1) / np.dot(v1, v1)

def proj(v1, v2):
    if __GPU__:
        return cp.multiply(gs_cofficient(v1, v2), v1)
    else:
        return np.multiply(gs_cofficient(v1, v2), v1)
    
def gs_orthogonalization(X):
    for i in range(len(X)):
        if i==0:
            Y = X[i]
            if __GPU__:
                Y = Y[cp.newaxis]
            else:
                Y = Y[np.newaxis]
            continue
        else:
            temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = temp_vec - proj_vec
        if __GPU__:
            Y = cp.concatenate((Y,temp_vec[cp.newaxis]),axis=0)
        else:
            Y = np.concatenate((Y,temp_vec[np.newaxis]),axis=0)
    return Y



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
            
            u, s, vh = LA.svd(flat)
            basis = vh[:flat.shape[0]][s > 1e-8][:512]
            
            if __GPU__:
                basis = cp.array(basis)
                # using SVD
                # u, s, vh = cp.linalg.svd(cp.array(flat),full_matrices=False)
                # basis = vh[:flat.shape[0]][s > 1e-8][:512]
                
                # using Gram-Schmidt Ortho-Normalization
                # vh = gs_orthogonalization(cp.array(flat))
                # vh = vh/cp.linalg.norm(vh,axis=1).reshape(vh.shape[0],1)
                # basis = vh[:flat.shape[0]][:512]
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
            
            if __GPU__:
                proj = cp.matmul(X,basis.T)
                projR = cp.matmul(proj,basis)
                D.append(cp.linalg.norm(projR - X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if __GPU__:
            D = cp.stack(D, axis=0)  # (num_classes, n_samples)
            preds = cp.argmin(D, axis=0)  # n_samples
            return cp.asnumpy(preds)
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            return preds


if __name__ == '__main__':
    datadir = './data'
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes, datadir)
    cache_file = os.path.join(datadir, args.dataset, 'rcdt.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
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
    if __GPU__:
        x_test = cp.array(x_test)
    for n_samples_perclass in [2 ** i for i in range(0, po_train_max+1)]:
        for repeat in range(num_repeats):
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            classifier = SubSpaceClassifier()
            tic = time.time()
            classifier.fit(x_train_sub, y_train_sub, num_classes)
            
            preds = classifier.predict(x_test)
            toc = time.time()
            print('Runtime of fit+predict functions: {} seconds'.format(toc-tic))
            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
            print('n_samples_perclass {} repeat {} acc {}'.format(n_samples_perclass, repeat, accuracy_score(y_test, preds)))

    accs = np.array(accs).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])

    results_dir = 'results/final/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'nsws.hdf5')
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)
