import cv2
import numpy as np
import os;
from random import shuffle
import tensorflow as tf
from keras.datasets import cifar10

path_to_ddsm = '../patchset/'
LR = 1e-3
IMG_SIZE = 224

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
import tensorflow as tf


### imports for new model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator



# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224,224,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))

model.add(Activation('softmax'))

def create_train_data():
        training_data = []
        for root, subFolders, file_names in os.walk(path_to_ddsm):
            for file_name in file_names:
                if '.jpg' in file_name:
                    dirname = os.path.basename(root)
                    file_path = os.path.join(root , file_name)
                    #print(dirname)
                    if  dirname == '1':
                        #print(dirname , file_name)
                        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                        #print img
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img), np.array([1,0,0,0])])
                    if dirname == '2':
                        #print(dirname , file_name);
                        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img), np.array([0 ,1,0,0])])
                    if dirname == '3':
                        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img), np.array([0,0,1,0])])
                        #print(dirname , file_name)
                    if dirname == '4':
                        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img), np.array([0,0,0,1])])

        shuffle(training_data)
        return training_data



traing_data = create_train_data()
train = traing_data[:-5]
test = traing_data[-5:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
print('x_train shape:', X.shape, test_x.shape)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])



print "done"
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X, Y, batch_size=5)
test_generator = test_gen.flow(test_x, test_y, batch_size=5)

model.fit_generator(train_generator, steps_per_epoch=4, epochs=5,
                    validation_data=test_generator, validation_steps=2)
