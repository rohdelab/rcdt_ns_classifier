# Radon cumulative distribution transform subspace models for image classification

This repository contains the Python language codes for reproducing the results in the paper titled "Radon cumulative distribution transform subspace models for image classification" using the Radon cumulative distribution transform nearest subspace (RCDT-NS) classifier. To use this classifier users need to install PyTransKit (Python Transport Based Signal Processing Toolkit) from: https://github.com/rohdelab/PyTransKit.

## Installation of PyTransKit

The library can be installed through pip
```
pip install pytranskit
```
Alternately, one can clone/download the repository from [[github](https://github.com/rohdelab/PyTransKit)] and add the `pytranskit` directory to your Python path.
```python
import sys
sys.path.append('path/to/pytranskit')
```

## RCDT-NS Classifier Demo

1. First, import the RCDT_NS class from PyTransKit.
```python
from pytranskit.classification.rcdt_ns import RCDT_NS
```
2. Load/read image data in a 3d array ```x_train``` with shape ```[#samples x #rows x #columns]```. Create another 1d array ```y_train``` containing class labels of corresponding images. 

3. Similarly, load test images ```x_test```

4. Create an instance of the RCDT_NS class. Users need to specify total number of class and directions (in degrees) of the Radon projections.
```python
import numpy as np
theta = np.linspace(0,180,45) .     # 45 equidistant angles for Radon projections
num_class = 10                      # for MNIST

rcdt_ns_obj = RCDT_NS(num_classes, theta)
```

5. Train the classifier using ```x_train``` and ```y_train```.
```python
rcdt_ns_obj.fit(x_train_sub, y_train_sub)
```

6. Test the classifier using ```x_test```.
```python
preds = rcdt_ns_obj.predict(x_test, use_gpu)
```
If ```use_gpu = True```, testing phase will run in GPU. Otherwise, CPU will be used.
```predict``` function returns the predicted class labels (in 1d array ```preds```) for the test images. To calculate the accuracy one can use ```accuracy_score``` function from [[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)] package.
```python
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(y_test, preds) * 100.
```

#### The above steps have also been compiled in a single python notebook ```Demo_RCDT_NS.ipynb``` which runs the RCDT-NS classifier on MNIST dataset.

## RCDT-NS Classifier in MATLAB
The MATLAB implementation of the RCDT-NS classifier can be found here [[rcdt_ns (matlab)](https://github.com/rohdelab/rcdt_ns_classifier_matlab)]

# Publication for Citation
Please cite the following publication when publishing findings that benefit from the codes provided here.

#### Shifat-E-Rabbi M, Yin X, Rubaiyat AHM, Li S, Kolouri S, Aldroubi A, Nichols JM, Rohde GK. "Radon cumulative distribution transform subspace modeling for image classification." arXiv preprint arXiv:2004.03669 (2020). [[Paper](https://arxiv.org/abs/2004.03669)]


## Reproduce Results from the Paper     

Python scripts for reproducing the results shown in the paper have been provided inside the ```compare_classification_methods/``` directory. To generate the results of the classification methods, first ```cd``` to this directory and then use the following commands:

1. Generate the results of the RCDT-NS classification method:
    - Use `python RCDT_NS_classification.py --dataset DATASET` to generate the results of the classification method based on Radon cumulative distribution transform subspace models. Example: `python RCDT_NS_classification.py --dataset MNIST` (here, images from the MNIST dataset have been organized in ```data/MNIST``` directory).

2. Generate the results of the CNN-based classification methods: 
    - Use `python CNN_classification.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

3. Floating point operation (FLOP) count results: 
    - Use `RCDT_NS_classification.py ----count_flops` and `CNN_flopcount.py` to generate the FLOPs counting results for the classification method based on Radon cumulative distribution transform subspace models and the classification methods based on convolutional neural networks, respectively.

4. Ablation study:
    - Use `python RCDT_NS_classification.py --dataset DATASET --classifier mlp` to generate the results of RCDT + MLP classification.
    - Use `python RCDT_NS_classification.py --dataset DATASET --use_image_feature` to generate the results of image feature + nearest subspace classification.
    
We also provide a bash script "MNIST_classification.sh" for a demonstration of how to do RCDT-NS classification and neural network classification on MNIST dataset.

### Dependencies

See "requirements.txt".

### Organize datasets

Organize an image classification dataset as follows:

1. Download the image dataset, and seperate it into the `training` and `testing` sets.
2. For the `training` set: 
    - Save images from different classes into separate `.mat` files. Dimension of the each `.mat` file would be `M x N x K`, where `M x N` is the size of the images and `K` is the number of samples per class.
    - Name of the mat file would be `dataORG_<class_index>.mat`. For example, `dataORG_0.mat` and `dataORG_1.mat` would be two mat files for a binary class problem.
    - Save the mat files in the `./data/training` directory.
3. For the `testing` set:
    - The first two steps here are the same as the first two steps for the `training` set.
    - Save the mat files in the `./data/testing` directory.
4. Update the `dataset_config` in `utils.py` with few informations of the dataset (e.g. image size, number of classes, maximum number of training samples).
