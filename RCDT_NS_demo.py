
"""
This demo code presents a handwritten digit (MNIST) recognition example using 
RCDT-NS classifier.

To use the classifer, users need to install 'pytranskit' package using 
'pip install pytranskit'. 
This package can also be cloned/downloaded from github link 
'https://github.com/rohdelab/PyTransKit.git', in that case users need to
specify the directory to 'pytranskit' using sys.path.append(/path/to/pytranskit)
in the codes.

For training the model, we have used randomly selected samples (specified by 
'n_samples_perclass') from the whole training set of the MNIST dataset. 
For testing, we have used all the test samples of the dataset.


Created on Tue Apr  7 19:09:19 2020

@author: A. H. M. Rubaiyat, X. Yin, M. Shifat-E-Rabbi
"""


# Import some necessary libraries
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Following two lines are used to locate the pytranskit package directory.
# If the package has already been installed, then these two lines are not needed.
import sys
sys.path.append('../')   # path to 'pytranskit' package

# Import RCDT-NS class from pytranskit package.
# This contains all the necessary functions to run the classifier
from pytranskit.classification.rcdt_ns import RCDT_NS

# The following library comes with the pytranskit package.
# In this example, this library has been used to load data from the dataset.
# User can implement there own load_data function, in that case this library will not be required.
#from pytranskit.classification.utils import *
from pytranskit.classification.utils import load_data, take_train_samples

# RCDT-NS classifier can run both in CPU and GPU.
# Following flag is used to select CPU(False)/GPU(True)
use_gpu = False

# data info
datadir = './data'        # directory of the dataset
dataset = 'MNIST'         # dataset name, there is a folder with same name inside datadir
num_classes = 10          # total number of classes in the dataset

# In this example we are using a subset of the total train samples to train 
# the model. Following variable specifies the total number of training samples 
# per class to be used in training phase.
n_samples_perclass = 128


theta = np.linspace(0, 176, 45)     # angles in degrees that will be used 
                                    # to calculate Radon projections
    
                                
if __name__ == '__main__':
    
    # load image data and labels from datadir
    # data: n_samples x n_rows x n_columns
    # labels: n_samples x 1
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes, datadir)
    
    # Take 'n_samples_perclass' randomly chosen samples per class to train the model.
    # If the user wants to use all the samples for training, this step is not required. 
    x_train_sub, y_train_sub = take_train_samples(x_train, y_train, n_samples_perclass, 
                                              num_classes, repeat=0)
    
    # Create an instance of the RCDT-NS class from pytranskit
    rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge=True)
    
    
    # Training phase::
    # Train the model using the randomle selected train samples
    # If the user wants to use all the training samples from dataset,
    # simply use 'x_train' and 'y_train' instead of 'x_train_sub' and 
    # 'y_train_sub' respectively.
    rcdt_ns_obj.fit(x_train_sub, y_train_sub)
    
    
    # Testing phase::
    # This function returns predicted class labels for the test samples using 
    # the trained model
    y_pred = rcdt_ns_obj.predict(x_test, use_gpu)
    
    # Calculate accuracy in %
    acc = accuracy_score(y_test, y_pred)*100
    print('\nTest accuracy: {:.2f}%'.format(acc))
    
    # Confusion matrix
    conf_mtx = confusion_matrix(y_test, y_pred)
    print('\n\nConfusion Matrix\n')
    print(conf_mtx)
    
    
    
    