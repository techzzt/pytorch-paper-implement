# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:06:00 2022

@author: keun
"""

# https://github.com/qubvel/classification_models/blob/master/classification_models/models/senet.py

#################################################
#############      SEResNet      ################
#################################################

# import package
import os 
import collections 
from keras_applications import imagenet_utils
# from classification_models import get_submodules_from_kwargs
# from ._common_blocks import GroupConv2D, ChannelSE
# from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple("ModelParams", 
                                     ["model_name", "repetitions", "residual_block", "groups",
                                      "reduction", "init_filters", "input_3x3", "dropout"]
                                     )


##### Additional Functions #####
def get_bn_params(**params):
    axis = 3 if backend.image_dat_format() == "channels_last" else 1
    default_bn_params = {
        "axis": axis,
        "epsilon": 9.999999747378752e-06,
        }
    default_bn_params.update(params)
    return default_bn_params

def get_num_channels(tensor):
    channels_last = 3 if backend.image_data_format() == "channels_last" else 1
    return backend.int_shape(tensor)[chanels_axis]



##### Residual Blocks ######

## Bottleneck ##
#--- ResNet ---#
def SEResNetBottleneck(filters, reduction = 16, strides = 1, **kwargs):
    bn_params = get_bn_params()
    
    def layer(input_tensor):
        x = input_tensor 
        residual = input_tensor
        
        # bottleneck 
        x = layers.Conv2D(filters // 4, (1, 1), kernel_initializer = "he_uniform",
                          strides = strides, use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.ZeroPadding2D(1)(x)
        x = layers.Conv2D(filters // 4, (3, 3), 
                          kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(filters, (1, 1), kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization("relu")(x)
        
        x = layers.Conv2D(filters, (1, 1), kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        
        # if number of filters or spatial dimensions changed make same manipulations with residual connection 
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)
        
        if strides != 1 or x_channels != r_channels:
            residual = layers.Conv2D(x_channels, (1, 1), strides = strides,
                                     kernel_initializer = "he_uniform", use_bias = False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)
            
        # apply attention module 
        x = ChannelSE(reduction = reduction, **kwargs)(x)
        
        # add residual connection 
        x = layers.Add()([x, residual])
        
        x = layers.Activation("relu")(x)
        
        return x
    
    return layer


## Bottleneck ##
#--- ResNeXt ---#
def SEResNeXtBottleneck(filters, reduction = 16, strides = 1, groups = 32, base_width = 4, **kwargs):

    bn_params = get_bn_params()
    
    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        width = (filters // 4) * base_width * groups // 64

        # bottleneck 

        x = layer.Conv2D(width, (1, 1), kernel_initializer = "he_uniform", use_bias = False)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.ZeroPadding2D(1)(x)
        x = GroupConv2D(width, (3, 3), strides = strides, groups = groups,
                        kernel_initializer = "he_uniform", use_bias = False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(filters, (1, 1), kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        
        # make same manipulations with residual connection (filters, dimensions changed)
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)
        
        if strides != 1 or x_channels != r_channels: 
            residual = layers.Conv2D(x_channels, (1, 1), strides = strides, 
                                     kernel_initializer = "he_uniform", use_bias = False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)
            
        # apply attention module
        x = ChannelSE(reduction = reduction, **kwargs)(x)
        
        # add residual connection 
        x = layers.Add()([x, residual])

        x = layers.Activation("relu")(x)
        return x        
        
    return layer 


## Bottleneck ##
#---   SE   ---#
# Add padding condition 
def SEBottleneck(filters, reduction = 16, strides = 1, groups = 64, is_first = False, **kwargs):
    bn_params = get_bn_params()
    module_kwargs = ({k: v for k, v in kwargs.items()
                      if k in ("backend", "layers", "models", "utils")})
    
    if is_first:
        downsample_kernel_size = (1, 1)
        padding = False
    else:
        downsample_kernel_size = (3, 3)
        padding = False
    
    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor
        
        # bottleneck
        x = layers.Conv2D(filters // 2, (1, 1), kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.ZeroPadding2D(1)(x)
        x = GroupConv2D(filters, (3, 3), strides = strides, groups = groups, 
                        kernel_initializer = "he_uniform", use_bias = False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(filters, (1, 1), kernel_initializer = "he_uniform", use_bias = False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)
        
        if strides != 1 or x_channels != r_channels:
            if padding:
                residual = layers.ZeroPadding2D(1)(residual)
            residual = layers.Conv2D(x_channels, downsample_kernel_size, strides = strides,
                                     kernel_initializer = "he_uniform", use_bias = False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)
            
        # apply attention module 
        x = ChannelSE(reduction = reduction, **kwargs)(x)
        x = layers.Add()[(x, residual)]
        x = layers.Activation("relu")(x)
        
        return x
    
    return layer
