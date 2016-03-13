'''
Potentially interesting idea: Train a network to recognize a pattern, then iteratively feed it the same input (starting
at a random value) and backpropagate on the input to get a "characteristic input" for a given output. Sort of a
generative model...
'''

import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(42)
LEARNING_RATE = 0.01

def layer(n_in, n_out):
    np_array = np.asarray(rng.uniform(low=-1, high=1, size=(n_in, n_out)), dtype=theano.config.floatX)
    return theano.shared(value=np_array, name='W', borrow=True)

def nonlin(x):
    return T.nnet.sigmoid(x)

# input and output vectors
X = T.vector('X')
Y = T.vector('Y')

# weight matrices
w1 = layer(2,5)
w2 = layer(5,1)

# the model itself
a1 = T.dot(X, w1)
z1 = nonlin(a1)
a2 = T.dot(z1, w2)
z2 = nonlin(a2)

# cost = squared error
cost = T.mean(T.sqr(z2 - Y))
weight_vars = [w1, w2]
weight_updates = [(x, x - LEARNING_RATE * T.grad(cost=cost, wrt=x)) for x in weight_vars]

train = theano.function(inputs=[X,Y], outputs=cost, updates=weight_updates, allow_input_downcast=True)

# reverse encoding
te_x = theano.shared(value=np.asarray(rng.uniform(low=0, high=1, size=(2, 2))), name='te_x', borrow=True)
te_y = theano.shared(value=np.asarray([[1], [0]]))

re_output = nonlin(T.dot(nonlin(T.dot(te_x, w1)), w2))
n_cost = T.mean(T.sqr(re_output - te_y))

c_updates = [(x, x - LEARNING_RATE * T.grad(cost=n_cost, wrt=x)) for x in [te_x]]
re_encode = theano.function(inputs=[], outputs=[te_x], updates=c_updates, allow_input_downcast=True)

# testing
for i in range(1000):
    re_encode()
print('Before training:')
print(re_encode())

# training
tr_x = np.asarray([[0, 1], [1, 0], [1, 1], [0, 0]])
tr_y = np.asarray([[1], [1], [0], [0]])
for i in range(30000):
    for x, y in zip(tr_x, tr_y):
        c = train(x, y)

# testing
for i in range(1000):
    re_encode()
print('After training:')
print(re_encode())
