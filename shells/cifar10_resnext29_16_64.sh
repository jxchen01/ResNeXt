#!/usr/bin/env sh
model=resnext29_16_64
dataset=cifar10
epochs=300

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ./data/cifar.python \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} --learning_rate 0.05 \
	--schedule 150 225 --gammas 0.1 0.1 --batch_size 64 --workers 2 --ngpu 4
