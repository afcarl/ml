import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

import pylab
from PIL import Image

rng = np.random.RandomState(42)

# 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3 * 9 * 9)
W = theano.shared(np.asarray(rng.uniform(low=-1. / w_bound, high=1. / w_bound, size=w_shp), dtype=input.dtype), name='W')

# initialize shared variables for bias
b_shp = (2,)
b = theano.shared(np.asarray(rng.uniform(low=-.5, high=.5, size=b_shp), dtype=input.dtype), name='b')

# build symbolic expression for computing the convolution
conv_out = conv.conv2d(input, W)

# create output
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

# open random image
img = Image.open('images/3wolfmoon.jpg')
img = np.asarray(img, dtype='float64') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input], pool_out)

invals = np.random.RandomState(1).rand(3, 2, 5, 5)
print('With ignore_border set to True:')
print('invals[0, 0, :, :] = \n', invals[0, 0, :, :])
print('output[0, 0, :, :] = \n', f(invals)[0, 0, :, :])

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
print('With ignore_border set to False:')
print('invals[1, 0, :, :] = \n', invals[1, 0, :, :])
print('output[1, 0, :, :] = \n', f(invals)[1, 0, :, :])
