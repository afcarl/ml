import theano
from matplotlib.pyplot import imshow, show
import numpy as np
import sys
from loaders import *

np.random.seed(404)

# load mnist training data (60000 training images)
MNIST_PATH = './data'
WEIGHT_FILE = 'weights.hdf5'
X_train, y_train = load_mnist(MNIST_PATH, kind='train')
X_test, y_test = load_mnist(MNIST_PATH, kind='t10k')

# configure theano to use float32's and adjust training data format
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

from keras.utils import np_utils

# one hot encoding of output categories
y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train /= 255
X_test /= 255

print(' '.join([str(X_train.shape[0]), 'training samples', str(X_test.shape[0]), 'test samples']))

# define the model itself
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

try:
    model.load_weights(WEIGHT_FILE)
except:
    model.fit(X_train, y_train_ohe, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test_ohe))
    model.save_weights(WEIGHT_FILE, overwrite=True)

score = model.evaluate(X_test, y_test_ohe, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
