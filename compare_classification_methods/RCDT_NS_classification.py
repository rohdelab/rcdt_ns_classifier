
"""
Created on Tue Apr  7 22:09:19 2020

@author: A. H. M. Rubaiyat, X. Yin, M. Shifat-E-Rabbi
"""

import argparse
import multiprocessing as mp
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as LA

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import sys
sys.path.append('../../')   # path to 'pytranskit' package

# Import RCDT-NS class from pytranskit package.
# This contains all the necessary functions to run the classifier
from pytranskit.classification.rcdt_ns import RCDT_NS
from utils import *

import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--use_image_feature', action='store_true')
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

num_classes, img_size, po_train_max, rm_edge = dataset_config(args.dataset)
theta = np.linspace(0, 176, 45) 


# Functions to calculate RCDT when MLP classifier is used
from pytranskit.optrans.continuous.radoncdt import RadonCDT
eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]
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
    print('Calculating RCDT ...')
    n_cpu = np.min([mp.cpu_count(), X.shape[0]])
    splits = np.array_split(X, n_cpu, axis=0)
    pl = mp.Pool(n_cpu)

    dataRCDT = pl.map(fun_rcdt_batch, splits)
    rcdt_features = np.vstack(dataRCDT)  # (n_samples, proj_len, num_angles)
    pl.close()
    pl.join()

    return rcdt_features

# Nearest Subspace classifier opearting on image samples
class NS:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.subspaces = []
        self.len_subspace = 0

    def fit(self, X, y):
        """Fit linear model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_rows, n_columns)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        self :
            Returns an instance of self.
        """
        if args.count_flops:
            assert X.dtype == np.float64

        for class_idx in range(self.num_classes):
            # generate the bases vectors
            class_data = X[y == class_idx]
            
            flat = class_data.reshape(class_data.shape[0], -1)
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            self.subspaces.append(basis)

            if args.count_flops:
               assert basis.dtype == np.float64

    def predict(self, X):
        """Predict using the linear model
        Parameters
        ----------
        X : array-like, sparse matrix, shape (n_samples, n_rows, n_columns)
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        X = X.reshape([X.shape[0], -1])
        if args.use_gpu:
            import cupy as cp
            X = cp.array(X)
        D = []
        for class_idx in range(self.num_classes):
            basis = self.subspaces[class_idx]
            basis = basis[:self.len_subspace,:]
            
            if args.use_gpu:
                D.append(cp.linalg.norm(cp.matmul(cp.matmul(X, cp.array(basis).T), 
                                                  cp.array(basis)) -X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if args.use_gpu:
            preds = cp.argmin(cp.stack(D, axis=0), axis=0)
            return cp.asnumpy(preds)
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            return preds




if __name__ == '__main__':
    datadir = '../data'
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes, datadir)

    if args.count_flops:
        rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge)
        for n_samples in [1, 10, 100]:
            high.start_counters([events.PAPI_DP_OPS,])
            rcdt_test = x_train[:n_samples]
            rcdt_test = rcdt_ns_obj.fun_rcdt_batch(rcdt_test)
            x=high.stop_counters()[0]
            print('rcdt_test.shape {} GFLOPS {}'.format(rcdt_test.shape, x/1e9))
            rcdt_gflops = (x / 1e9) / n_samples
        print('rcdt_gflops: {}'.format(rcdt_gflops))

    num_repeats = 10
    accs = []
    all_preds = []

    if args.count_flops:
       all_train_gflops, all_test_gflops = [], []
    for n_samples_perclass in [2 ** i for i in range(0, po_train_max+1)]:
        for repeat in range(num_repeats):
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            if args.classifier == 'subspace':
                if args.use_image_feature:
                    classifier = NS(num_classes)    
                else:
                    classifier = RCDT_NS(num_classes, theta, rm_edge)
                tic = time.time()
                if args.count_flops:
                    x_train_sub = x_train_sub.astype(np.float64)
                    x_test = x_test.astype(np.float64)
                    high.start_counters([events.PAPI_DP_OPS,])
                    
                    classifier.fit(x_train_sub, y_train_sub)
                    train_gflops = high.stop_counters()[0] / 1e9 + n_samples_perclass * num_classes * rcdt_gflops

                    high.start_counters([events.PAPI_DP_OPS,])
                    if args.use_image_feature:
                        preds = classifier.predict(x_test)
                    else:
                        preds = classifier.predict(x_test, args.use_gpu)
                    test_gflops = (high.stop_counters()[0] / 1e9) / x_test.shape[0] + rcdt_gflops

                    all_train_gflops.append(train_gflops)
                    all_test_gflops.append(test_gflops)
                else:
                    if args.use_image_feature:
                        classifier.fit(x_train_sub, y_train_sub)
                        preds = classifier.predict(x_test)
                    else:
                        classifier.fit(x_train_sub, y_train_sub)
                        preds = classifier.predict(x_test, args.use_gpu)
            else:
                tic = time.time()
                x_train_rcdt = rcdt_parallel(x_train_sub)
                x_test_rcdt = rcdt_parallel(x_test)
                x_train_rcdt, x_test_rcdt = x_train_rcdt.astype(np.float32), x_test_rcdt.astype(np.float32)
                
                x_train_rcdt_flat = x_train_rcdt.reshape(x_train_rcdt.shape[0], -1)
                x_test_rcdt_flat = x_test_rcdt.reshape(x_test_rcdt.shape[0], -1)
                
                module = nn.Sequential(nn.Linear(x_train_rcdt_flat.shape[1], 500), nn.ReLU(), nn.Linear(500, num_classes), nn.Softmax(dim=1))
                classifier = NeuralNetClassifier(
                    module,
                    max_epochs=500,
                    lr=5e-4,
                    optimizer=torch.optim.Adam,
                    iterator_train__shuffle=True,
                    train_split=None,
                    device='cuda'
                )
                classifier.fit(x_train_rcdt_flat, y_train_sub)
                preds = classifier.predict(x_test_rcdt_flat)
            toc = time.time()
            # print('Runtime of fit+predict functions: {} seconds'.format(toc-tic))
            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
            print('n_samples_perclass {} repeat {} acc {}'.format(n_samples_perclass, repeat, accuracy_score(y_test, preds)))
            if args.count_flops:
                print('train GFLOPS {:.5f} test GFLOPS {:.5f}'.format(train_gflops, test_gflops))

    accs = np.array(accs).reshape(-1, num_repeats)
    print('Mean accuracies: {}'.format(np.mean(accs,axis=1)))
    
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
