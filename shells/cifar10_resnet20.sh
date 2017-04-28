#!/usr/bin/env sh
model=resnet20
dataset=cifar10
epochs=164

CUDA_VISIBLE_DEVICES=0,1 python main.py ./data/cifar.python \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} \
       	--schedule 82 123 --gammas 0.1 0.1 --learning_rate 0.1 --decay 0.0001 --batch_size 128 --workers 2 --ngpu 2
