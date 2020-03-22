# Radon cumulative distribution transform subspace models (RCDT-SUBS) for image classification

This repo contains the Python language code for reproducing the results in the paper titled "Radon cumulative distribution transform subspace models for image classification".

## Dependencies

See "requirements.txt".

## Usage

First setup the datasets: download all the data from this link, and put them in the `data` folder.

1. Generate the results of the classification method:
    - Use `python RCDTSUBS_classification.py --dataset DATASET` to generate the results of the classification method based on Radon cumulative distribution transform subspace models.

2. Generate the results of neural network-based classification methods: 
    - Use `python CNN_classification.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

3. Floating point operation (FLOP) count results: 
    - Use `RCDTSUBS_classification.py` and `CNN_classification.py` to generate the FLOPs counting results for the classification method based on Radon cumulative distribution transform subspace models and the classification methods based on convolutional neural networks, respectively.

4. Ablation study:
    - Use `python RCDTSUBS_classification.py --dataset DATASET --classifier mlp` to generate the results of RCDT + MLP classification.
    - Use `python RCDTSUBS_classification.py --dataset DATASET --use_image_feature` to generate the results of image feature + nearest subspace classification.
