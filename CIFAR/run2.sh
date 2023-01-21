#!/bin/bash

for model in cifar_resnet_20_8 cifar_resnet_20_16 cifar_resnet_20_24
do
    python3 main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model $model --data cifar10 --epoch 160
done
