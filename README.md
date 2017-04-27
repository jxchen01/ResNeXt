# ResNeXt & ResNet Pytorch Implementation
ResNeXt (Aggregated Residual Transformations for Deep Neural Networks)
ResNet (Deep Residual Learning for Image Recognition)

- [x] Train on Cifar10 and Cifar100 with ResNeXt29-8-64d and ResNeXt29-16-64d
- [x] Train on Cifar10 and Cifar100 with ResNet20,32,44,56,110
- [ ] Train Imagenet

## Usage
To train on Cifar-10 using 4 gpu:

```bash
python main.py ./data/cifar.python --dataset cifar10 --arch resnext29_8_64 --save_path ./snapshots/cifar10_resnext29_8_64_310 --epochs 310 --learning_rate 0.05 --schedule 150 225 300 --gammas 0.1 0.1 0.1 --batch_size 128 --workers 4 --ngpu 4
```

## Configurations
From the original [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) papers:

| depth | cardinality | base width | parameters | accuracy cifar10 | accuracy cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.65       |       17.77       |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |       3.58       |       17.31       |   ResNeXt    |
|  20   |      -      |     -      |    0.27M   |       8.75       |         -         |   ResNet     |
|  32   |      -      |     -      |    0.46M   |       7.51       |         -         |   ResNet     |
|  44   |      -      |     -      |    0.66M   |       7.17       |         -         |   ResNet     |
|  56   |      -      |     -      |    0.85M   |       6.97       |         -         |   ResNet     |
| 110   |      -      |     -      |    1.7M    |  6.43(6.61^0.16) |         -         |   ResNet     |
| 1202  |      -      |     -      |   19.4M    |       7.93       |         -         |   ResNet     |

## Our Results
| depth | cardinality | base width | parameters | accuracy cifar10 | accuracy cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.86       |                   |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |        -         |                   |   ResNeXt    |
|  20   |      -      |     -      |    0.27M   |       8.66       |       33.17       |   ResNet     |
|  32   |      -      |     -      |    0.46M   |       7.47       |         -         |   ResNet     |
|  44   |      -      |     -      |    0.66M   |        -         |         -         |   ResNet     |
|  56   |      -      |     -      |    0.85M   |                  |         -         |   ResNet     |
| 110   |      -      |     -      |    1.7M    |                  |         -         |   ResNet     |
| 1202  |      -      |     -      |   19.4M    |        -         |         -         |   ResNet     |

## Other frameworks
* [torch (@facebookresearch)](https://github.com/facebookresearch/ResNeXt). (Original) Cifar and Imagenet
* [MXNet (@dmlc)](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k). Imagenet
* [pytorch (@prlz77)](https://github.com/prlz77/ResNeXt.pytorch). Cifar

## Cite
```
@article{xie2016aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```
