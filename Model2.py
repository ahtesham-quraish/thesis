import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import cv2
import numpy as np
import os;
from random import shuffle
from numpy import array
import tensorflow as tf
from keras.datasets import cifar10

path_to_ddsm = '../patchset/'
IMG_SIZE = 224
batch_size = 128 # in each iteration, we consider 128 training examples at once
num_epochs = 12 # we iterate twelve times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # there will be 128 neurons in both hidden layers

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)


def create_train_data():
    training_data = []
    for root, subFolders, file_names in os.walk(path_to_ddsm):
        for file_name in file_names:
            if '.jpg' in file_name:
                dirname = os.path.basename(root)
                file_path = os.path.join(root, file_name)
                # print(dirname)
                if dirname == 'CALCIFICATION_BENIGN':
                    # print(dirname , file_name)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    # print img
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img), np.array([1, 0, 0, 0])])
                if dirname == 'CALCIFICATION_MALIGNANT':
                    # print(dirname , file_name);
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img), np.array([0, 1, 0, 0])])
                if dirname == 'MASS_BENIGN':
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img), np.array([0, 0, 1, 0])])
                    # print(dirname , file_name)
                if dirname == 'MASS_MALIGNANT':
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img), np.array([0, 0, 0, 1])])

    shuffle(training_data)
    return training_data


traing_data = create_train_data()
train = traing_data[:-5]
test = traing_data[-5:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]
print('x_train shape:', X.shape, test_x.shape)

(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data
#print X_train[0], y_train[0]
X_train = X_train.reshape(X_train.shape[0], height, width, depth)
X_test = X_test.reshape(X_test.shape[0], height, width, depth)
X_train = X_train.astype('float32')
X = X.astype('float32')
X_test = X_test.astype('float32')
test_x = test_x.astype('float32')
X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range
#print  y_train.shape
X /=255;
test_x /=255

Y = array(Y)
test_y = array(test_y)
print(y_train[0] ,Y[1])
Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
print type(X) , type(X_train) , type(test_y)
print(X_train.shape , Y_train[0] , Y_train[1] , test_y[0] )
inp = Input(shape=(224, 224, depth)) # N.B. TensorFlow back-end expects channel dimension last
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
flat = Flatten()(drop_1)
hidden = Dense(hidden_size, activation='relu')(flat) # Hidden ReLU layer
drop = Dropout(drop_prob_2)(hidden)
out = Dense(4, activation='softmax')(drop) # Output softmax layer

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X, Y, # Train the model using the training set...
          batch_size=4, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(test_x, test_y, verbose=1) # Evaluate the trained model on the test set!