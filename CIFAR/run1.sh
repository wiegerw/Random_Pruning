#!/bin/bash

for model in cifar_resnet_20 cifar_resnet_32 cifar_resnet_44 cifar_resnet_56 cifar_resnet_110 
do
    python3 main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model $model --data cifar10 --epoch 160
done
