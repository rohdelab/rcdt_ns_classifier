# Radon cumulative distribution transform subspace models (RCDT-SUBS) for image classification

This repo contains the Python language code for reproducing the results in the paper titled "Radon cumulative distribution transform subspace models for image classification".

## Dependencies

See "requirements.txt".

## Usage

1. Setup the datasets: for the dataset used in the paper, use this link to download the datasets, and put them in the `./data` folder. To use a new dataset, organize the data using following steps:
    - Download the image dataset, and separate `training` and `testing` sets.
    - First consider `training` set: \\
        a. save images from different classes into separate `.mat` files. Size of each matfile would be MxNxK, where MxN is the size of the images and K is the number of samples per class
        b. name of the mat file would be `dataORG_<class_index>.mat`. Example: `dataORG_0.mat` and `dataORG_1.mat` would be two mat files for a binary class problem
        c. save the mat files in the `./data/training` directory.
    - For `testing` set:
        a. and b. same as `training`
        c. save the mat files in the `./data/testing` directory.
    - Update the `dataset_config` in `utils.py` with the information (e.g. image size, number of classes, maximum number of training samples) of new dataset.        

2. Generate the results of the RCDT-SUBS classification method:
    - Use `python RCDTSUBS_classification.py --dataset DATASET` to generate the results of the classification method based on Radon cumulative distribution transform subspace models.

3. Generate the results of the CNN-based classification methods: 
    - Use `python CNN_classification.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

4. Floating point operation (FLOP) count results: 
    - Use `RCDTSUBS_flopcount.py` and `CNN_flopcount.py` to generate the FLOPs counting results for the classification method based on Radon cumulative distribution transform subspace models and the classification methods based on convolutional neural networks, respectively.

5. Ablation study:
    - Use `python RCDTSUBS_classification.py --dataset DATASET --classifier mlp` to generate the results of RCDT + MLP classification.
    - Use `python RCDTSUBS_classification.py --dataset DATASET --use_image_feature` to generate the results of image feature + nearest subspace classification.
