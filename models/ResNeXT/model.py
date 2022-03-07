### import package ###

# model 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display image
from torchvision import utils
import matplotlib.pyplot as plt
%matplotlib inline 

# utils
import numpy as np
from torchsummary import summary
import time
import copy 


### Model structure ###
# BottleNeck (Figure 3)
class ResNeXtBottleNeck(nn.Module):
    mul = 2
    def __init__(self, in_planes, group_width, cardinality, stride = 1):
        super(ResNeXtBottleNeck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(group_width)
        
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size = 3, stride = 1, padding = 1, groups = cardinality, bias = False)
        self.bn2 = nn.BatchNorm2d(group_width)
        
        self.conv3 = nn.Conv2d(group_width, group_width*self.mul, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(group_width*self.mul)
        self.shortcut = nn.Sequential()
        
        # identifier 
        if stride != 1 or in_planes != group_width * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, group_width*self.mul, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(group_width*self.mul)
                )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2d(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out 
    
# ResNeXT
class ResNeXT(nn.Module):
    def __init__(self, block, num_blocks, cardinality = 32, width = 4, num_classes = 10):
        super(ResNeXt, self).__init__()
        self.in_planes = 64
        self.group_conv_width = cardinality * width
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.make_layer(block, cardinality, num_blocks[0], stride = 1)
        self.layer2 = self.make_layer(block, cardinality, num_blocks[1], stride = 2)
        self.layer3 = self.make_layer(block, cardinality, num_blocks[2], stride = 2)
        self.layer4 = self.make_layer(block, cardinality, num_blocks[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.group_conv_width, num_classes)
        
    def make_layer(self, block, cardinality, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, self.group_conv_width, cardinality, strides[i]))
            self.in_planes = block.mul * self.group_conv_width
        self.group_conv_width *= block.mul
        return nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out 
    
# Add ResNeXt architecture 
def ResNeXt50():
    return ResNeXT(ResNeXtBottleNeck, [3, 4, 6, 3])

def ResNeXt101():
    return ResNeXT(ResNeXtBottleNeck, [3, 4, 23, 3])

def ResNeXt152():
    return ResNeXT(ResNeXtBottleNeck, [3, 8, 36, 3])

