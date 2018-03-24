# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
"""
Created on Fri Mar 16 09:51:50 2018

@author: xingshuli
"""
import warnings

from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K


CIFAR_TH_WEIGHTS_PATH = ''
CIFAR_TF_WEIGHTS_PATH = ''
CIFAR_TH_WEIGHTS_PATH_NO_TOP = ''
CIFAR_TF_WEIGHTS_PATH_NO_TOP = ''

IMAGENET_TH_WEIGHTS_PATH = ''
IMAGENET_TF_WEIGHTS_PATH = ''
IMAGENET_TH_WEIGHTS_PATH_NO_TOP = ''
IMAGENET_TF_WEIGHTS_PATH_NO_TOP = ''


def ResNext(input_shape = None, depth = 29, cardinality = 8, width = 64, weight_decay = 5e-4, 
            include_top = True, weights = None, input_tensor = None, 
            pooling = None, classes = 10):
    
    if weights not in {'cifar10', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')
    if weights == 'cifar10' and include_top and classes != 10:
        raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                         ' as true, `classes` should be 10')
    if type(depth) == int:
        if (depth - 2) % 9 != 0:
            raise ValueError('Depth of the network must be such that (depth - 2)'
                             'should be divisible by 9.')
    
    #determine proper input shape
    input_shape = _obtain_input_shape(input_shape, 
                                      default_size = 32, 
                                      min_size = 8, 
                                      data_format = K.image_data_format(), 
                                      require_flatten = include_top, 
                                      weights = weights)
    if input_tensor is None:
        img_input = Input(shape = input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
            
    x = __create_res_next(classes, img_input, include_top, depth, cardinality, width, 
                          weight_decay, pooling)
    
    #Ensure the model takes into account any potential predecessors of 'input_tensor'
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    #create model
    model = Model(inputs, x, name = 'resnext')
    
    #load weights
    if weights == 'cifar10':
        if (depth == 29) and (cardinality == 8) and (width == 64):
            if K.image_data_format() == 'channels_first':
                if include_top:
                    weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels.h5', 
                                            CIFAR_TH_WEIGHTS_PATH, 
                                            cache_subdir='models')
                else:
                    weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels_no_top.h5', 
                                            CIFAR_TH_WEIGHTS_PATH_NO_TOP, 
                                            cache_subdir='models')
                    
                model.load_weights(weights_path)
                
                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image dimension ordering convention '
                                  '(`image_dim_ordering="th"`). '
                                  'For best performance, set '
                                  '`image_dim_ordering="tf"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
                    
                    convert_all_kernels_in_model(model)
            else:
                if include_top:
                    weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels.h5', 
                                            CIFAR_TF_WEIGHTS_PATH, 
                                            cache_subdir='models')
                else:
                    weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels_no_top.h5', 
                                            CIFAR_TF_WEIGHTS_PATH_NO_TOP, 
                                            cache_subdir='models')
                model.load_weights(weights_path)
                if K.backend() == 'theano':
                    convert_all_kernels_in_model(model)
                
    return model

def ResNextImageNet(input_shape = None, depth = [3, 4, 6, 3], cardinality = 32, width = 4, weight_decay = 5e-4, 
                    include_top = True, weights = None, input_tensor = None, pooling = None, 
                    classes = 1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
        
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    
    if type(depth) == int and (depth - 2) % 9 != 0:
        raise ValueError('Depth of the network must be such that (depth - 2)'
                         'should be divisible by 9.')
    
    #determine the proper input shape
    input_shape = _obtain_input_shape(input_shape, 
                                      default_size = 224, 
                                      min_size = 112, 
                                      data_format = K.image_data_format(), 
                                      require_flatten = include_top, 
                                      weights = weights)
    
    if input_tensor is None:
        img_input = Input(shape = input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
            
    x = __create_res_next_imagenet(classes, img_input, include_top, depth, cardinality, width, 
                                   weight_decay, pooling)
    
    #Ensure the model takes into account any potentail preprocessors of 'input_tensor'
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    #create model
    model = Model(inputs, x, name = 'resnext')
    
    #load weights
    if weights == 'imagenet':
        if (depth == [3, 4, 6, 3]) and (cardinality == 32) and (width == 4):
            if K.image_data_format() == 'channels_first':
                if include_top:
                    weights_path = get_file('resnext_imagenet_32_4_th_dim_ordering_th_kernels.h5', 
                                            IMAGENET_TH_WEIGHTS_PATH, 
                                            cache_subdir='models')
                else:
                    weights_path = get_file('resnext_imagenet_32_4_th_dim_ordering_th_kernels_no_top.h5', 
                                            IMAGENET_TH_WEIGHTS_PATH_NO_TOP, 
                                            cache_subdir='models')
                    
                model.load_weights(weights_path)
                
                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image dimension ordering convention '
                                  '(`image_dim_ordering="th"`). '
                                  'For best performance, set '
                                  '`image_dim_ordering="tf"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
                    convert_all_kernels_in_model(model)
                
            else:
                if include_top:
                    weights_path = get_file('resnext_imagenet_32_4_tf_dim_ordering_tf_kernels.h5', 
                                            IMAGENET_TF_WEIGHTS_PATH, 
                                            cache_subdir='models')
                else:
                    weights_path = get_file('resnext_imagenet_32_4_tf_dim_ordering_tf_kernels_no_top.h5', 
                                            IMAGENET_TF_WEIGHTS_PATH_NO_TOP, 
                                            cache_subdir='models')
                model.load_weights(weights_path)
                
                if K.backend() == 'theano':
                    convert_all_kernels_in_model(model)
                    
    return model
    
def __initial_conv_block(input, weight_decay = 5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = Conv2D(64, (3, 3), padding = 'same', use_bias = False, kernel_initializer = 'he_normal', 
               kernel_regularizer = l2(weight_decay))(input)
    
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)
    
    return x

def __initial_conv_block_imagenet(input, weight_decay = 5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(64, (7, 7), padding = 'same', use_bias = False, kernel_initializer = 'he_normal', 
               kernel_regularizer = l2(weight_decay), strides = (2, 2))(input)
    
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
    
    return x

def __grouped_convolution_block(input, grouped_channels, cardinality, strides, 
                                weight_decay = 5e-4):
    
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    group_list = []
    
    if cardinality == 1:
        x = Conv2D(grouped_channels, (3, 3), padding = 'same', use_bias = False, strides = (strides, strides), 
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay))(init)
        x = BatchNormalization(axis = channel_axis)(x)
        x = Activation('relu')(x)
        return x
    
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else 
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
        
        x = Conv2D(grouped_channels, (3, 3), padding = 'same', use_bias = False, strides = (strides, strides), 
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay))(x)
        
        group_list.append(x)
        
    group_merge = concatenate(group_list, axis = channel_axis)
    x = BatchNormalization(axis = channel_axis)(group_merge)
    x = Activation('relu')(x)
    
    return x
    

def __bottleneck_block(input, filters = 64, cardinality = 8, strides = 1, weight_decay = 5e-4):
    init = input
    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding = 'same', strides = (strides, strides), 
                          use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(weight_decay))(init)
            init = BatchNormalization(axis = channel_axis)(init)
            
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis = channel_axis)(init)
            
    x = Conv2D(filters, (1, 1), padding = 'same', use_bias = False, 
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)
    
    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    
    x = Conv2D(filters * 2, (1, 1), padding = 'same', use_bias = False, 
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    
    x = BatchNormalization(axis = channel_axis)(x)
    
    x = add([init, x])
    
    x = Activation('relu')(x)
    
    return x
    
def __create_res_next(nb_classes, img_input, include_top, depth = 29, cardinality = 8, 
                      width = 4, weight_decay = 5e-4, pooling = None):
    
    if type(depth) is list or type(depth) is tuple:
        #if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        #otherwise, default to 3 blocks
        N = [(depth - 2) // 9 for _ in range(3)]
        
    filters = cardinality * width
    filters_list = []
    
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2
        
    x = __initial_conv_block(img_input, weight_decay)
    
    #block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides = 1, weight_decay = weight_decay)
        
    N = N[1:] #remove the first block from block definition list
    filters_list = filters_list[1:] #remove the first filter from filter list
    
    #block 2 (input_size becomes half)
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2, 
                                       weight_decay = weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1, 
                                       weight_decay = weight_decay)
    
    
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias = False, kernel_initializer='he_normal', 
                  kernel_regularizer = l2(weight_decay), activation = 'softmax')(x)
        
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        
    return x


def __create_res_next_imagenet(nb_classes, img_input, include_top, depth, cardinality = 32, 
                               width = 4, weight_decay = 5e-4,  pooling = None):
    
    if type(depth) is list or type(depth) is tuple:
        #if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        #otherwise, default to 3 blocks
        N = [(depth - 2) // 9 for _ in range(3)]
        
    filters = cardinality * width
    filters_list = []
    
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2
        
    x = __initial_conv_block(img_input, weight_decay)
    
    #block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides = 1, weight_decay = weight_decay)
        
    N = N[1:] #remove the first block from block definition list
    filters_list = filters_list[1:] #remove the first filter from filter list
    
    #block 2 (input_size becomes half)
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2, 
                                       weight_decay = weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1, 
                                       weight_decay = weight_decay)
    
    
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias = False, kernel_initializer='he_normal', 
                  kernel_regularizer = l2(weight_decay), activation = 'softmax')(x)
        
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    return x
    


if __name__ == '__main__':
    model = ResNext((32, 32, 3), depth=29, cardinality=8, width=64)
    model.summary()
        

    
    
            
    
    
                   
    
                    
                



