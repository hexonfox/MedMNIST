#!/bin/bash

python train.py --gpu 1 --model resnet20 --dataset pathmnist
python train.py --gpu 1 --model vgg16 --dataset pathmnist
python train.py --gpu 1 --model convnet --dataset pathmnist

python train.py --gpu 1 --model resnet20 --dataset octmnist
python train.py --gpu 1 --model vgg16 --dataset octmnist
python train.py --gpu 1 --model convnet --dataset octmnist

python train.py --gpu 1 --model resnet20 --dataset tissuemnist
python train.py --gpu 1 --model vgg16 --dataset tissuemnist
python train.py --gpu 1 --model convnet --dataset tissuemnist