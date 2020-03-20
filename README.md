# Radon cumulative transform subspace models for image classification

This repo contains the code for reproducing the resulst in the paper "Radon cumulative distribution transform subspace models for image classification".

## Dependencies

See requirements.txt.

## Usage

First setup the datasets: download all the data from this link, and put them in the `data` folder.

use `train_nsws.py` to reproduce results of the proposed method. 

```
usage: main.py [-h] [--dataset DATASET] --space {image,wndchrm,rcdt} --model
               {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3,ResNet,DenseNet}
               [-T] [-U] [--splits {2,3,4,5,6,7,8,9,10}]
               [--SVM-kernel {rbf,linear}] [--preprocessed]
               --target_image_size {32,64,75,128,256}

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --space {image,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3,ResNet,DenseNet}
  -T, --transfer-learning
                        neural network use pretrained weights instead of
                        training from scratch
  -U, --data_augmentation
                        use data augmentation for neural network based
                        approaches
  --splits {2,3,4,5,6,7,8,9,10}
                        number of splits for cross-validation
  --SVM-kernel {rbf,linear}
  --preprocessed        reproduce the results on Hela dataset reported in the
                        paper
  --target_image_size {32,64,75,128,256}
                        image size used for classification
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space image --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space image --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

