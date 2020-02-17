#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:58:26 2020

@author: hasnat
"""

import numpy as np
import numpy.linalg as LA
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import multiprocessing as mp


from pytranskit.optrans.continuous.radoncdt import RadonCDT

dataset = 'MNIST' # 'MNIST'/'AffMNIST'/'OAM'/'SignMNIST'/'Synthetic'

if dataset in ['MNIST']:
    data_folder = '../DATA/data700'
    print(dataset +': '+data_folder)
    rm_edge=True
    numClass = 10
    classes = range(numClass)
    #classes = [1,5]
    po_max = 12   # maximum train samples = 2^po_max    
elif  dataset in ['AffMNIST']:
    data_folder = '../DATA/data701'
    print(dataset +': '+data_folder)
    rm_edge=True
    numClass = 2
    #classes = range(numClass)
    classes = [1,5]
    po_max = 12   # maximum train samples = 2^po_max
elif dataset in ['OAM']:
    data_folder = '../DATA/data705_s3'
    print(dataset +': '+data_folder)
    rm_edge=False
    numClass = 32
    classes = range(numClass)
    po_max = 9   # maximum train samples = 2^po_max
elif dataset in ['SignMNIST']:
    data_folder = '../DATA/data710';
    print(dataset +': '+data_folder)
    rm_edge=False
    numClass = 3
    classes = range(numClass)
    po_max = 10   # maximum train samples = 2^po_max    
elif dataset in ['Synthetic']:
    data_folder = '../DATA/data699';
    print(dataset +': '+data_folder)
    rm_edge=True
    numClass = 1000
    classes = range(numClass)
    po_max = 7   # maximum train samples = 2^po_max

po = range(0,po_max+1 ,1)
tr_split = np.power(2,po)
N_exp = 1

eps=1e-6
x0_range = [0,1]
x_range = [0,1]
Rdown = 4   # downsample radon projections (w.r.t. angles)
theta = np.linspace(0,176,180/Rdown)


radoncdt = RadonCDT(theta)

template = []

def fun_load_train_data_single(cl):
    data = loadmat(data_folder+'/org/training/dataORG_'+str(classes[cl])+'.mat')['xxO']
    data[data<0]=0
    label = cl*np.ones([1,data.shape[2]])
    return (data, label)

def fun_load_train_data_batch(nClass):
    data = [fun_load_train_data_single(j) for j in nClass]
    return data

def fun_load_test_data_single(cl):
    data = loadmat(data_folder+'/org/testing/dataORG_'+str(classes[cl])+'.mat')['xxO']
    data[data<0]=0
    label = cl*np.ones([1,data.shape[2]])
    return (data, label)

def fun_load_test_data_batch(nClass):
    data = [fun_load_test_data_single(j) for j in nClass]
    return data

def fun_rcdt_single(I): 
    Ircdt = radoncdt.forward(x0_range, template/np.sum(template), x_range, I/np.sum(I), rm_edge)
    return Ircdt.reshape([Ircdt.shape[0]*Ircdt.shape[1]],order='F')

def fun_rcdt_batch(data):
    dataRCDT = [fun_rcdt_single(data[:,:,j] + eps) for j in range(data.shape[2])]
    return np.array(dataRCDT)


def fun_classify(N_cv, dataTe, dataM, labels, indtr, template, v1, v2):
    eps = 1e-8
    dist = 10000*np.ones([numClass,dataTe.shape[1]])
    yPred = -1*np.ones([1,dataTe.shape[1]])
    
    for class_idx in range(numClass):
        ## Training phase
        cls_l = np.where(labels==class_idx)
        cls_l = cls_l[1]
        
        # take x_ax number of train samples
        ind = min(cls_l)+indtr[0:x_ax,N_cv]-1
        dataTr = []
        
        pl_count = min(mp.cpu_count(), len(ind))
        batchTr = dataM[:,:,ind]
        pl = mp.Pool(pl_count)
        splits = np.array_split(batchTr, pl_count,axis=2)
        
        dataRCDT = pl.map(fun_rcdt_batch, splits)
        dataTr = np.vstack(dataRCDT)
        dataTr = dataTr.T
        pl.close()
        pl.join()
        
        ######## need to add deformation########
        dataTr = np.concatenate((dataTr,v1),axis=1)
        dataTr = np.concatenate((dataTr,v2),axis=1)
        
        # generate the bases vectors
       
        u,s,vh = LA.svd(dataTr)
        # choose first 512 components if train sample>512
        s_num = min(512,len(s))
        s = s[0:s_num]
        s_ind = np.where(s>eps)
        basis = u[:,s_ind[0]]

        ## Testing Phase
        
        proj = basis.T @ dataTe

        # dataTe: (h, N), basis: (h, M), proj: (N, M)
        
        projR = basis @ proj # projR: (h, N)
        dist[class_idx] = LA.norm(projR - dataTe, axis=0)
            
    for i in range(dataTe.shape[1]):
        d = dist[:,i]
        yPred[0,i]=np.where(d==min(d))[0]
        
    return yPred


if __name__ == '__main__':
    
    start = time.time()
    
    # load train index
    indtr = loadmat(data_folder+'/Ind_tr.mat')['indtr']
    
    # load test data
    print('load test data')   
    pl_count = min(mp.cpu_count(), numClass)
    pl = mp.Pool(pl_count)
    splits = np.array_split(np.array(range(numClass)), pl_count,axis=0)
    
    data = pl.map(fun_load_test_data_batch, splits)
    dataN = (data[0])[0][0]
    yTe = (data[0])[0][1]
    pl.close()
    pl.join()
    
    for i in range(1,len(data)):
        dataN = np.concatenate((dataN,(data[i])[0][0]),axis=2)
        yTe = np.concatenate((yTe,(data[i])[0][1]),axis=1)
    
    # template for RCDT
    template = np.ones([dataN.shape[0], dataN.shape[1]]) + eps
    Ircdt = radoncdt.forward(x0_range, template/np.sum(template), x_range, template/np.sum(template), rm_edge)
    
    # calc RCDT of test images
    print('Calculate RCDT of test images')
    pl = mp.Pool(mp.cpu_count())
    splits = np.array_split(dataN, mp.cpu_count(),axis=2)
    
    dataRCDT = pl.map(fun_rcdt_batch, splits)
    dataTe = np.vstack(dataRCDT)
    dataTe = dataTe.T
    pl.close()
    pl.join()

    dataN = []
    
    # load train data
    print('load train data')
    pl_count = min(mp.cpu_count(), numClass)
    pl = mp.Pool(pl_count)
    splits = np.array_split(np.array(range(numClass)), pl_count,axis=0)
    
    data = pl.map(fun_load_train_data_batch, splits)
    dataM = (data[0])[0][0]
    yTr = (data[0])[0][1]
    pl.close()
    pl.join()
    
    for i in range(1,len(data)):
        dataM = np.concatenate((dataM,(data[i])[0][0]),axis=2)
        yTr = np.concatenate((yTr,(data[i])[0][1]),axis=1)
        
    # deformation vectors for  translation
    v1 = np.zeros([Ircdt.shape[0]*Ircdt.shape[1], 1])
    v2 = np.zeros([Ircdt.shape[0]*Ircdt.shape[1], 1])
    indx=0
    for th in theta:
        for j in range(Ircdt.shape[0]):
            v1[indx]=np.cos(th*np.pi/180)
            v2[indx]=np.sin(th*np.pi/180)
            indx=indx+1
    
    
    print('Run the classifier')
    acc = np.zeros([len(tr_split), N_exp])
    indx = 0
    
    for x_ax in tr_split:
        print('Train samples: '+str(x_ax))
        yPred = -1*np.ones([N_exp,yTe.shape[1]])
        for N_cv in range(N_exp):
            yPred[N_cv,:] = fun_classify(N_cv, dataTe, dataM, yTr, indtr, template, v1, v2)
            acc[indx,N_cv]=accuracy_score(yTe[0],yPred[N_cv,:])
            print(acc[indx,N_cv])   

        indx = indx+1
        savemat(data_folder+'/RESULTS/predictions_tr-'+str(x_ax)+'.mat',{'yPred': yPred, 'yTe':yTe})
    savemat(data_folder+'/RESULTS/accuracy.mat',{'acc': acc})
    
    end = time.time()
    print('Elapsed time: '+str(end - start) + ' seconds')
    print(np.mean(acc,axis=1))
        


