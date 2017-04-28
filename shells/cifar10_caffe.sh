#!/usr/bin/env sh
model=caffe_cifar
dataset=cifar10
epochs=180

CUDA_VISIBLE_DEVICES=0 python main.py ./data/cifar.python \
        --dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} \
        --schedule 120 150 --gammas 0.1 0.1 --learning_rate 0.01 --decay 0.004 --batch_size 128 --workers 2 --ngpu 1
