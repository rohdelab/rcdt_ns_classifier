# Radon cumulative distribution transform subspace models (RCDT-SUBS) for image classification

This repo contains the Python language code for reproducing the results in the paper titled "Radon cumulative distribution transform subspace models for image classification".

## Usage     

To generate the results of the classification methods, use the following commands:

1. Generate the results of the RCDT-SUBS classification method:
    - Use `python RCDTSUBS_classification.py --dataset DATASET` to generate the results of the classification method based on Radon cumulative distribution transform subspace models.

2. Generate the results of the CNN-based classification methods: 
    - Use `python CNN_classification.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

3. Floating point operation (FLOP) count results: 
    - Use `RCDTSUBS_flopcount.py` and `CNN_flopcount.py` to generate the FLOPs counting results for the classification method based on Radon cumulative distribution transform subspace models and the classification methods based on convolutional neural networks, respectively.

4. Ablation study:
    - Use `python RCDTSUBS_classification.py --dataset DATASET --classifier mlp` to generate the results of RCDT + MLP classification.
    - Use `python RCDTSUBS_classification.py --dataset DATASET --use_image_feature` to generate the results of image feature + nearest subspace classification.

## Dependencies

See "requirements.txt".

## Organize datasets

Organize an image classification dataset using the following steps:

1. Download the image dataset, and separate the `training` and `testing` sets.
2. For the `training` set: 
    - Save images from different classes into separate `.mat` files. Size of each `.mat` file would be $MxNxK$, where MxN is the size of the images and K is the number of samples per class.
    - Name of the mat file would be `dataORG_<class_index>.mat`. For example, `dataORG_0.mat` and `dataORG_1.mat` would be two mat files for a binary class problem.
    - Save the mat files in the `./data/training` directory.
3. For the `testing` set:
    - The first two steps are the same as the first two steps for the `training` set.
    - Save the mat files in the `./data/testing` directory.
4. Update the `dataset_config` in `utils.py` with the information (e.g. image size, number of classes, maximum number of training samples) of the dataset.   
