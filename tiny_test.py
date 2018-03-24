# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
"""
Created on Tue Mar 20 11:23:45 2018

@author: xingshuli
"""
import os
import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2

from keras.preprocessing import image

from resNeXt import ResNext

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

img_width, img_height = 32, 32

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

input_tensor = Input(shape = input_shape)

train_data_dir = os.path.join(os.getcwd(), 'tiny_test/train')
validation_data_dir = os.path.join(os.getcwd(), 'tiny_test/validation')

nb_train_samples = 10000
nb_validation_samples = 2500

num_class = 25
epochs = 10000
batch_size = 64

#base_model parameters
depth = 29
cardinality = 8
width = 16
weight_decay = 5e-4 

save_dir = os.path.join(os.getcwd(), 'tiny_ResNext_model')
model_name = 'keras_tiny_trained_model.h5'


base_model = ResNext(depth = depth, cardinality = cardinality, width = width, 
                     include_top = False, weights = None, input_tensor = input_tensor, 
                     pooling= 'avg', weight_decay = weight_decay)
                     
x = base_model.output

x = Dense(num_class, use_bias = False, kernel_initializer = 'he_normal', 
          kernel_regularizer = l2(weight_decay), activation = 'softmax')(x)

train_model = Model(base_model.input, outputs = x, name = 'tiny_test_model')

for layer in base_model.layers:
    layer.trainable = True

#set initial learning rate = 1e-3
#add nesterov momentum to accelerate speed of training
sgd = SGD(lr = 0.0001, momentum = 0.9, nesterov = True)
train_model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
train_model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255, 
                                   rotation_range=15, 
                                   width_shift_range=5./32, 
                                   height_shift_range=5./32, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

label_indices = validation_generator.class_indices
#label_indices (classXX:value), sort the label_indices by value
label_indices = sorted(label_indices.items(), key = lambda item:item[1])

#store the new_indices into dict_file.txt
dic_path = os.path.join(os.getcwd(), 'dic_file.txt')
with open(dic_path, 'w') as f0:
    for item in label_indices:
        f0.write(item[0] + ':' + str(item[1]))
        f0.write('\n')
f0.close()

#early-stopping 
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 12, mode = 'auto')

#learning rate schedule
lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, 
                               mode = 'auto', min_lr = 1e-6)

#set callbacks for model fit
callbacks = [early_stopping, lr_reducer]

#model fit
hist = train_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size, 
    callbacks=callbacks)

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

#the reasonable accuracy of model should be calculated based on
#the value of patience in EarlyStopping: accur = accur[-patience + 1:]/patience
Er_patience = 13  # Er_patience = patience + 1
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))
                                
#save model 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
train_model.save(model_path)
print('save trained model at %s' % model_path)


#predict a category of input image
img_path = '/home/xingshuli/Desktop/test_pictures/labrado.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /=255.
print('Input image shape:', x.shape)
preds = train_model.predict(x)



