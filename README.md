# ResNext-Structure-for-tiny-image
ResNext Struture, EarlyStopping and ReduceLROnPlateau Callbacks in Keras, A folder based prediction method 

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
