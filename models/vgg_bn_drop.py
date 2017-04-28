from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class ConvBNReLU(nn.Module):
    def __init__(self, nInputPlane, nOutputPlane):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nInputPlane, nOutputPlane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nOutputPlane)

    def forward(self, x):
        out = self.conv(x)
	out = self.bn(out)
        return F.relu(out, inplace=True)

class CifarVGGNet(nn.Module):
    def __init__(self, block, num_classes):
        super(CifarVGGNet, self).__init__()

        self.num_classes = num_classes

	self.block_0 = nn.Sequential() # 32 * 32
        self.block_0.add_module('vgg_0.0', block( 3, 64))
        self.block_0.add_module('vgg_0.0_drop', nn.Dropout(p=0.3, inplace=True))
        self.block_0.add_module('vgg_0.1', block(64, 64))
        self.block_0.add_module('vgg_0_pool', nn.MaxPool2d(2, 2))
	
	self.block_1 = nn.Sequential() # 16 * 16
        self.block_1.add_module('vgg_1.0', block( 64, 128))
        self.block_1.add_module('vgg_1.0_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_1.add_module('vgg_1.1', block(128, 128))
        self.block_1.add_module('vgg_1_pool', nn.MaxPool2d(2, 2))

	self.block_2 = nn.Sequential() # 08 * 08
        self.block_2.add_module('vgg_2.0', block(128, 256))
        self.block_2.add_module('vgg_2.0_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_2.add_module('vgg_2.1', block(256, 256))
        self.block_2.add_module('vgg_2.1_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_2.add_module('vgg_2.2', block(256, 256))
        self.block_2.add_module('vgg_2_pool', nn.MaxPool2d(2, 2))

	self.block_3 = nn.Sequential() # 04 * 04
        self.block_3.add_module('vgg_3.0', block(256, 512))
        self.block_3.add_module('vgg_3.0_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_3.add_module('vgg_3.1', block(512, 512))
        self.block_3.add_module('vgg_3.1_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_3.add_module('vgg_3.2', block(512, 512))
        self.block_3.add_module('vgg_3_pool', nn.MaxPool2d(2, 2))

	self.block_3 = nn.Sequential() # 02 * 02
        self.block_3.add_module('vgg_4.0', block(512, 512))
        self.block_3.add_module('vgg_4.0_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_3.add_module('vgg_4.1', block(512, 512))
        self.block_3.add_module('vgg_4.1_drop', nn.Dropout(p=0.4, inplace=True))
        self.block_3.add_module('vgg_4.2', block(512, 512))
        self.block_3.add_module('vgg_4_pool', nn.MaxPool2d(2, 2))

	
	self.classifier = nn.Sequential() # 512
	self.classifier.add_module('vgg_cls_drop1', nn.Dropout(p=0.5, inplace=True))
	self.classifier.add_module('vgg_cls_fc1', nn.Linear(512, 512))
	self.classifier.add_module('vgg_cls_bn1', nn.BatchNorm1d(512))
	self.classifier.add_module('vgg_cls_relu', nn.ReLU())
	self.classifier.add_module('vgg_cls_drop2', nn.Dropout(p=0.5, inplace=True))
	self.classifier.add_module('vgg_cls_prediction', nn.Linear(512, self.num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block_0.forward(x)
        x = self.block_1.forward(x)
        x = self.block_2.forward(x)
        x = self.block_3.forward(x)
        return self.classifier(x)

def vgg_cifar(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarVGGNet(ConvBNReLU, num_classes)
    return model
