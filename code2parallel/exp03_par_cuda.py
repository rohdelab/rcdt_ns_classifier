#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:50:20 2020

@author: ar3fx
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:06:17 2020

@author: hasnat
"""

import numpy as np
import numpy.linalg as LA
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import multiprocessing
from datetime import datetime
import cupy as cp

data_folder = 'data700'
Rdown = 4   # downsample radon projections (w.r.t. angles)
numClass = 10


def fun_classify(N_cv, dataN, labels, indtr, dataM, v1, v2):
    eps = 1e-4
    dist = 10000*np.ones([numClass,dataN.shape[1]])
    yPred = -1*np.ones([1,dataN.shape[1]])
    
    for class_idx in range(numClass):
        ## Training phase
        cls_l = np.where(labels==class_idx)
        cls_l = cls_l[1]
        
        # take x_ax number of train samples
        ind = min(cls_l)+indtr[0:x_ax,N_cv]-1
        dataTr = dataM[:,ind]
        ######## need to add deformation########
        dataTr = np.concatenate((dataTr,v1),axis=1)
        dataTr = np.concatenate((dataTr,v2),axis=1)

        dataTr_cu = cp.asarray(dataTr)
        # generate the bases vectors
        print('=========== compute basis')
        print(datetime.now())
        print(dataTr.shape)
        # u,s,vh = LA.svd(dataTr)
        u,s,vh = cp.linalg.svd(dataTr_cu)
        u,s,vh = cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)

        s_ind = np.where(s>eps)
        basis = u[:,s_ind[0]]

        ## Testing Phase
        print(datetime.now())
        print('testing phase 1')
        proj = np.transpose(basis.T @ dataN)

        # dataN: (h, N), basis: (h, M), proj: (N, M)
        
        print(datetime.now())
        print('testing phase 2')
        projR = basis @ proj.T # projR: (h, N)
        dist[class_idx] = LA.norm(projR - dataN, axis=0)
        # for i in range(dataN.shape[1]):
        #     projR = sum((proj[i,:]*basis).T).T
        #     dist[class_idx,i]= LA.norm(projR- dataN[:,i])
        print(datetime.now())
        print('finished')
            
    for i in range(dataN.shape[1]):
        d = dist[:,i]
        yPred[0,i]=np.where(d==min(d))[0]
        
    return yPred


if __name__ == '__main__':
    
    # load train index
    indtr = loadmat(data_folder+'/Ind_tr.mat')['indtr']
    
    # load train data
    class_idx = 0
    data = loadmat(data_folder+'/tbm/training/dataTBM_'+str(class_idx)+'.mat')['xxT']
    data = data[:,range(0,180,Rdown),:]
    dataM = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]),order='F')
    labels = class_idx*np.ones([1,data.shape[2]])
    
    for class_idx in range(1,numClass):
        data = loadmat(data_folder+'/tbm/training/dataTBM_'+str(class_idx)+'.mat')['xxT']
        data = data[:,range(0,180,Rdown),:]
        dataM = np.concatenate((dataM, np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]),order='F')), axis=1)
        labels = np.concatenate((labels, class_idx*np.ones([1,data.shape[2]])), axis=1)
        
    # deformation vectors for  translation
    v1 = np.zeros([data.shape[0]*data.shape[1], 1])
    v2 = np.zeros([data.shape[0]*data.shape[1], 1])
    indx=0
    for theta in range(0,180,Rdown):
        for j in range(data.shape[0]):
            v1[indx]=np.cos(theta*np.pi/180)
            v2[indx]=np.sin(theta*np.pi/180)
            indx=indx+1
        
    # load test data
    class_idx = 0
    data = loadmat(data_folder+'/tbm/testing/dataTBM_'+str(class_idx)+'.mat')['xxT']
    data = data[:,range(0,180,Rdown),:]
    dataN = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]),order='F')
    yTe = class_idx*np.ones([1,data.shape[2]])
    
    for class_idx in range(1,numClass):
        data = loadmat(data_folder+'/tbm/testing/dataTBM_'+str(class_idx)+'.mat')['xxT']
        data = data[:,range(0,180,Rdown),:]
        dataN = np.concatenate((dataN, np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]),order='F')), axis=1)
        yTe = np.concatenate((yTe, class_idx*np.ones([1,data.shape[2]])), axis=1)
        
    
    indx = 0
    
    po = range(13)
    tr_split = np.power(2,po)
    N_exp = 5
    
    acc = np.zeros([len(tr_split), N_exp])
    
    start = time.time()
    for x_ax in [512]:
        print('Train samples: '+str(x_ax))
        
        yPred = -1*np.ones([N_exp,yTe.shape[1]])
        for N_cv in range(1):
            yPred[N_cv,:] = fun_classify(N_cv, dataN, labels, indtr, dataM, v1, v2)
            acc[indx,N_cv]=accuracy_score(yTe[0],yPred[N_cv,:])
            print(acc[indx,N_cv])            


        indx = indx+1
        savemat(data_folder+'/RESULTS/predictions_tr-'+str(x_ax)+'.mat',{'yPred': yPred, 'yTe':yTe})
    savemat(data_folder+'/RESULTS/accuracy.mat',{'acc': acc})
    
    end = time.time()
    print(end - start)
    
    
    
    
    
    
    
    
