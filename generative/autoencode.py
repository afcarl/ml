import numpy as np
import theano
import theano.tensor as T

import generative

from mnist.loaders import load_mnist
from keras.utils import np_utils

# initialize random number generator
rng = np.random.RandomState(42)

# load the mnist data
MNIST_PATH = '../mnist/data'
X_train, y_train = load_mnist(MNIST_PATH, kind='train')
X_test, y_test = load_mnist(MNIST_PATH, kind='t10k')

# normalize
X_train = X_train / 255

def pmatlab(a):
    print(';'.join(','.join(str(u) for u in t) for t in a.reshape(28, 28)))

# convert to one-hot encoding
y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

gb = generative.GenerativeBackprop(n_in=X_train.shape[1], n_hidden=[1000], n_out=y_train.shape[1])

print('Testing...')
nums = gb.generate(np.asarray([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]))

for num in nums:
    pmatlab(num)

print('Training...')
gb.train(X_train, y_train, 1)

print('Testing...')
nums = gb.generate(np.asarray([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]))

for num in nums:
    pmatlab(num)
