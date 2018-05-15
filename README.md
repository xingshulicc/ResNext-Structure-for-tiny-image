# ResNext-Structure-for-tiny-image
Group Convolution, EarlyStopping and ReduceLROnPlateau Callbacks in Keras, A folder based prediction method 

1. Dataset:

The Dataset includes 25 categories: lobster, goldenfish, newt, crocodile, snake, scorpion, spider, koala, jellyfish, snail, penguin, seamew, chihuahua, labradors, bear, ladybug, fly, mantis, monarch, yak, gorilla, car, lemon, coral, orange. Each category contains 500 images. I selected 400 images from each category for training and the rest images are used for testing. The size of each image during training is restricted to 32 * 32.


2. Network Structure:

As for training tiny images, we cannot adopt very deep CNN models. Therefore, in order to improve the classification accuracy, increasing the width instead of depth of CNN becomes an reasonable choice. 
Here, I refered to the structure of ResNext (https://arxiv.org/pdf/1611.05431.pdf). At the same time, to accelerate training and overcome the problem of overfitting, I add EarlyStopping and ReduceLROnPlateau Callbacks in the code.


3. Experimental Platform:

Python(3.5)

Keras (1.2.0) -- Tensorflow backend (GPU version)

GPU:
1080 ti  (11G) * 2


4. A folder based prediction method:

Instead of generating .txt file, this method allows us to get the prediction results of each folder in test dataset. 

Notice: the folder names in test dataset should be same with that of training dataset.


5. some explanation about network structure:

The value of depth should satisfy (depth - 2) // 9 == 0

2: the first 3 by 3 Conv, 64 and the last Dense, nb_classes

9: In each block there are 3 bottleneck structures and each bottleneck structure has 3 convolutional layers

So if depth = 29, it means that there are 3 blocks in the network

For each block, Downsampling is implemented in the 3 by 3 Conv of the first bottleneck structure, which is a little different from ResNet-50: the downsampling is done in the first 1 by 1 Conv.

Note:

N = [3, 3, 3]

filters_list = 128, 256, 512
