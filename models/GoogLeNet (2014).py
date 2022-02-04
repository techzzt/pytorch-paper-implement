# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:17:51 2022

@author: keun
"""

# 논문 설명 

# import package 

# Model 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os 

# display images 
from torchvision import utils 
import matplotlib.pyplot as plt
%matplotlib inline

# utils 
import numpy as np
from torchsummary import summary
import time
import copy


# load dataset
# splits = ('train, 'train + unlabeled', 'unlabeled', 'test')

train_ds = datasets.STL10("./data", split = "train", download = True, transform = transforms.ToTensor())
val_ds = datasets.STL10("./data", split = "test", download = True, transform = transforms.ToTensor())


# normalize dataset (mean, std)
train_mean_RGB = [np.mean(x.numpy(), axis = (1, 2)) for x, _ in train_ds]
train_std_RGB = [np.std(x.numpy(), axis = (1, 2)) for x, _ in train_ds]

train_mean_R = np.mean([m[0] for m in train_mean_RGB])
train_mean_G = np.mean([m[1] for m in train_mean_RGB])
train_mean_B = np.mean([m[2] for m in train_mean_RGB])

train_std_R = np.mean([s[0] for s in train_std_RGB])
train_std_G = np.mean([s[1] for s in train_std_RGB])
train_std_B = np.mean([s[2] for s in train_std_RGB])

val_mean_RGB = [np.mean(x.numpy(), axis = (1, 2)) for x, _ in val_ds]
val_std_RGB = [np.std(x.numpy(), axis = (1, 2)) for x, _ in val_ds]

val_mean_R = np.mean([m[0] for m in val_mean_RGB])
val_mean_G = np.mean([m[1] for m in val_mean_RGB])
val_mean_B = np.mean([m[2] for m in val_mean_RGB])

val_std_R = np.mean([s[0] for s in val_std_RGB])
val_std_G = np.mean([s[1] for s in val_std_RGB])
val_std_B = np.mean([s[2] for s in val_std_RGB])

# define image transformation
train_transformation = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(224),
                                           transforms.Normalize([train_mean_R, train_mean_G, train_mean_B], [train_std_R, train_std_G, train_std_B]), 
                                           transforms.RandomHorizontalFlip(),
                                           ])

val_transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(224),
                                         transforms.Normalize([train_mean_R, train_mean_G, train_mean_B], [train_std_R, train_std_G, train_std_B]),
                                         ])

# apply transformation
# https://pytorch.org/vision/stable/datasets.html
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/stl10.html#STL10
# datasets.transform -> self.transform(img), default = None
# verify_str_arg = input(split option)

train_ds.transform = train_transformation 
val_ds.transform = val_transformation

# create DataLoader 
train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True)
val_dl = DataLoader(val_ds, batch_size = 64, shuffle = True)

# display sample images
def show(img, y = None, color = True):
    img = img.numpy()
    img_tr = np.transpose(img, (1, 2, 0))
    plt.imshow(img_tr)
    
    if y is not None: 
        plt.title("labels:" + str(y))
        
np.random.seed(2022)
torch.manual_seed(2022)

grid_size = 4
rand_idx = np.random.randint(0, len(train_ds), grid_size)
print("image indices:", rand_idx)

x_grid = [train_ds[i][0] for i in rand_idx]
y_grid = [train_ds[i][0] for i in rand_idx]

x_grid = utils.make_grid(x_grid, nrow = 4, padding = 2)

plt.figure(figsize = (10, 10))
show(x_grid, y_grid)

# Model (GoogLeNet)
# inception v3 have two aux branch

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits = True, num_classes = 10, init_weights = True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False 
        self.aux_logits = aux_logits
        
        self.conv1 = conv_block(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = conv_block(64, 192, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        
        # auxiliary classifier 
        
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        
        # auxiliary classifier 
        
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AvgPool2d(7, 1)
        self.dropout = nn.Dropout(p = 0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        
        if self.aux_logits: 
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
            
        # weight initialization
        
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
            
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        
        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x
        
    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                    

# define convolution block 

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )
    
    def forward(self, x):
        return self.conv_layer(x)
    
    
# define inception block

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size = 1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size = 1),
            conv_block(red_3x3, out_3x3, kernel_size = 3, padding = 1),
            )
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size = 1),
            conv_block(red_5x5, out_5x5, kernel_size = 5, padding = 2),
            ) 
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            conv_block(in_channels, out_1x1pool, kernel_size = 1)
            )
        
    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x
    

# define auxiliary classifier 

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            conv_block(in_channels, 128, kernel_size = 1),
            )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
            )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    

x = torch.randn(3, 3, 224, 224)

model = GoogLeNet(aux_logits = True, num_classes = 10, init_weights = True)
output = model(x)
print(output)

# summarize model structure

summary(model, input_size = (3,224,224))
