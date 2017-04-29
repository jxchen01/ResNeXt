#!/usr/bin/env sh
model=resnet110
dataset=cifar10
epochs=166

CUDA_VISIBLE_DEVICES=0,1 python main.py ./data/cifar.python \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} \
	--schedule 1 83 124 --gammas 10 0.1 0.1 --learning_rate 0.01 --decay 0.0001 --batch_size 128 --workers 4 --ngpu 2
