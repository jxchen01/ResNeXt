# ResNeXt & ResNet Pytorch Implementation
ResNeXt (Aggregated Residual Transformations for Deep Neural Networks)
ResNet (Deep Residual Learning for Image Recognition)

- [x] Train on Cifar10 and Cifar100 with ResNeXt29-8-64d and ResNeXt29-16-64d
- [x] Train on Cifar10 and Cifar100 with ResNet20,32,44,56,110
- [ ] Train Imagenet

## Usage
To train on Cifar-10 using 4 gpu:

```bash
python main.py ./data/cifar.python --dataset cifar10 --arch resnext29_8_64 --save_path ./snapshots/cifar10_resnext29_8_64_300 --epochs 300 --learning_rate 0.05 --schedule 150 225 --gammas 0.1 0.1 --batch_size 128 --workers 4 --ngpu 4
```

Or there are some off-the-shelf scripts can dirrectly be used for training.

```bash
sh ./shells/cifar10_resnet20.sh
```

And a simplified caffenet-like model for cifar10, obtaining 89.5 top1 accuracy.

```
sh ./shells/cifar10_caffe.sh
```

## Configurations
From the original [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) papers:

| depth | cardinality | base width | parameters |  error   cifar10 |   error  cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.65       |       17.77       |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |       3.58       |       17.31       |   ResNeXt    |
|  20   |      *      |     *      |    0.27M   |       8.75       |         -         |   ResNet     |
|  32   |      *      |     *      |    0.46M   |       7.51       |         -         |   ResNet     |
|  44   |      *      |     *      |    0.66M   |       7.17       |         -         |   ResNet     |
|  56   |      *      |     *      |    0.85M   |       6.97       |         -         |   ResNet     |
| 110   |      *      |     *      |    1.7M    |       6.61       |         -         |   ResNet     |
| 1202  |      *      |     *      |   19.4M    |       7.93       |         -         |   ResNet     |

## My Results {Last Epoch Error (Best Error)}
| depth | cardinality | base width | parameters |  error   cifar10 |   error  cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.67       |    17.66(17.47)   |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |    3.59(3.39)    |                   |   ResNeXt    |
|  20   |      *      |     *      |    0.27M   |       8.47       |       32.99       |   ResNet     |
|  32   |      *      |     *      |    0.46M   |       7.67       |       30.80       |   ResNet     |
|  44   |      *      |     *      |    0.66M   |       7.23       |       29.45       |   ResNet     |
|  56   |      *      |     *      |    0.85M   |       6.86       |       28.89       |   ResNet     |
| 110   |      *      |     *      |    1.7M    |       6.62       |       27.62       |   ResNet     |

## Other frameworks
* [torch (@facebookresearch)](https://github.com/facebookresearch/ResNeXt). (Original) Cifar and Imagenet
* [MXNet (@dmlc)](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k). Imagenet
* [pytorch (@prlz77)](https://github.com/prlz77/ResNeXt.pytorch). Cifar

## Cite
```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
@article{xie2016aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```
