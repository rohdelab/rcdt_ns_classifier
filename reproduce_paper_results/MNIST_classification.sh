#!/bin/bash

# RCDT-SUBS classification on MNIST dataset
python RCDT_NS_classification.py --dataset MNIST

# Use shallowcnn model to do classification on MNIST dataset
python CNN_classification.py --dataset MNIST --model shallowcnn
