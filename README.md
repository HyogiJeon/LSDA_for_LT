# Improving Latent Space Data Augmentation for Long-Tail Distributions in Instance Segmentation

This repository contains the implementation of the Hyogi Jeon`s M.S. thesis:
> **Improving Latent Space Data Augmentation for Long-Tail Distributions in Instance Segmentation**<br>
> Hyogi Jeon, Soochahn Lee (Kookmin University)

### Abstract
 Recently, as training data is collected based on web crawling, a problem of long-tail distributions occurs, which degrades the performance of the deep learning model. Previous studies have attempted to solve the data imbalance problem by improving the loss function. However, the loss function improvement is not a fundamental solution because the long-tail distributions problem is fundamentally the lack of training data. Therefore, in this paper, to solve the fundamental problem, the problem was solved using data augmentation technique. Existing data augmentation create new data by transforming the original image. However, since the existing data augmentation technique cannot change the posture or appearance of an object, there is a limit in increasing the amount of training data. In this paper, to solve this problem, the problem was solved by using the data augmentation technique in the Latent Space. Based on the previously studied techniques, we propose a new technique to identify the problems of the technique and solve the problems.
 
**Note: This repo is the code for classification task (Not Segmentation task)**<br>

## Running Environment
We tested on the following settings:
- pytorch
- torchvision
- numpy 
- scikit-learn
- matplotlib
 
## Getting Stareted
### Datasets
- CIFAR10/100-LT(Long-tail): This dataset is the long-tailed version of CIFAR10/100. We provide the code for converting to long-tail version
 
### Training
Training on default settings:
```
python main.py
```

if you want to use tsne, tensorboard and save weights, Use the following commands:
```
python main.py --run_train_tsne True --run_test_tsne True --run_writer True --save_weights True
```

## Results
Results of CIFAR10-LT
| Method   | Models       | Accuracy |
| :------: | :----------: | :------: |
| Baseline | ResNet34     | 77.33%   |
| [FASA](https://arxiv.org/abs/2102.12867)| ResNet34     | 80.11%   |
| Ours     | ResNet34     | 80.93%   |
