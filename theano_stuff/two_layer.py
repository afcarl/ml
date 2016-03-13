import theano
import theano.tensor as T
import numpy as np

X = theano.shared(value=np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]]), name='X')
y = theano.shared(value=np.asarray([[1], [0], [1], [0]]), name='y')
rng = np.random.RandomState(1234)
LEARNING_RATE = 0.01


def layer(n_in, n_out):
    np_array = np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=theano.config.floatX)
    return theano.shared(value=np_array, name='W', borrow=True)

W1 = layer(2, 3)
W2 = layer(3, 1)

output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1)), W2))
cost = T.sum((y - output) ** 2)
updates = [(x, x - LEARNING_RATE * T.grad(cost, x)) for x in [W1, W2]]

train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=[output])

for i in range(60000):
    if (i+1) % 10000 == 0:
        print(i+1)
    train()

print(test())
