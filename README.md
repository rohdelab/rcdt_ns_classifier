# Radon cumulative distribution transform subspace models for image classification

This repo contains the Python language code for reproducing the results in the paper titled "Radon cumulative distribution transform subspace models for image classification".

## Dependencies

See "requirements.txt".

## Usage

First setup the datasets: download all the data from this link, and put them in the `data` folder.

1. Reproduce the results of the classification method:
    - Use `python train_nsws.py --dataset DATASET` to reproduce the results of the classification method based on Radon cumulative distribution transform subspace models.

2. Reproduce the results of neural network-based classification methods: 
    - Use `python train_nn.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

3. Floating point operation (FLOP) count results: 
    - Use `train_nn_gflops.py` and `train_nsws_gflops.py` to reproduce the FLOPs counting results for neural network-based classification methods and the classification method based on Radon cumulative distribution transform subspace models.

4. Ablation study:
    - Use `python train_nsws.py --dataset DATASET --classifier mlp` to reproduce the results of RCDT + MLP classification.
    - Use `python train_nsws.py --dataset DATASET --use_image_feature` to reproduce the results of image feature + nearest subspace classification.
