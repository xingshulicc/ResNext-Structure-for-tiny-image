# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
"""
Created on Fri Mar 23 10:54:08 2018

@author: xingshuli
"""
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

#assign GPU for computation
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#model path
save_dir = os.path.join(os.getcwd(), 'tiny_ResNext_model')
model_name = 'keras_tiny_trained_model.h5'
model_path = os.path.join(save_dir, model_name)

#load model
model_test = load_model(model_path)

'''
keras -- ImageDataGenerator -- flow_from_directory(directory):
class name assigned to each sample based 
on the sorted order of folder names
'''
#get class_indices of training from dic_file.txt
dic_path = os.path.join(os.getcwd(), 'dic_file.txt')
list_1 = []
with open(dic_path, 'r') as f1:
    lines = f1.readlines()
    for line in lines:
        data = line.strip('\n').split(':')
        list_1.append(data)
f1.close()

class_indices = {}
for item in list_1:
    item[1] = int(item[1])
    class_indices.update({item[0]:item[1]})

#pre-parameters for folder prediction
accur_list = []
img_width, img_height = 32, 32
dir_path = os.path.join(os.getcwd(), 'validation_test')

dirs = os.listdir(dir_path) #dirs contains folders' name
order_dirs = sorted(dirs)

folder_path = []
for file in order_dirs:
    path = os.path.join(dir_path, str(file))
    folder_path.append(path)

#get prediction of each file in folde_path
for folder in folder_path:
    folder_name = os.path.split(folder)[1]
    files = os.listdir(folder)
    num_right = 0
    accur = 0.0
    for file in files:
        file_path = os.path.join(folder, str(file))
        img_input = image.load_img(file_path, target_size = (img_width, img_height))
        img_input = image.img_to_array(img_input)
        img_input = img_input / 255.
        img_input = np.expand_dims(img_input, axis = 0)
        prediction = model_test.predict(img_input)
        prediction = prediction.tolist()
        prediction = sum(prediction, [])
        prediction = np.array(prediction)
        max_index = np.argmax(prediction)
        if max_index == class_indices[folder_name]:
            num_right += 1
    
    accur = num_right / len(files)
    accur_list.append(accur)
    
print(accur_list)

xticks = order_dirs
x = np.arange(len(accur_list)) + 1            
y = np.array(accur_list)
plt.bar(x, y, width = 0.8, color = 'b', align = 'center')
plt.xticks(x, xticks)
plt.xlabel('Labels of Category')
plt.ylabel('Prediction Accuracy')

for x, y in zip(x, y):
    plt.text(x, y, '%.4f' % y, ha='center', va= 'bottom')

plt.show()




