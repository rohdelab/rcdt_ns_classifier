# Radon cumulative transform subspace models for image classification

This repo contains the code for reproducing the resulst in the paper "Radon cumulative distribution transform subspace models for image classification".

## Dependencies

See requirements.txt.

## Usage

First setup the datasets: download all the data from this link, and put them in the `data` folder.

Reproduce the results of the proposed method:
  - Use `python train_nsws.py --dataset DATASET` to reproduce results of the RCDT + subspace classification. 
  - Use `python train_nsws.py --dataset DATASET --classifier mlp` to reproduce the result of RCDT + MLP classification.
  - Use `python train_nsws.py --dataset DATASET --use_image_feature` to reproduce the result of image feature + subspace classification.

Reproduce the results of neural network models: `python train_nn.py --dataset DATASET --model MODEL`, where `MODEL` could be `shallowcnn`, `resnet18`, and `vgg11`.

Use `train_nn_gflops.py` and `train_nsws_gflops.py` to reproduce the FLOPs counting results for neural network models and the proposed model.
