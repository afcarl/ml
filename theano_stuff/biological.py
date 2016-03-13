import theano
import theano.tensor as T
import numpy as np

rng = np.random

N = 400
feats = 784

LEARNING_RATE = 0.01

D = (rng.randn(N, feats), rng.randint(size=(1, N), low=-1, high=2))
training_steps = 50000


def nonlin(x):
    return T.tanh(x)

# declare symbolic variables
x = T.dmatrix('x')
y = T.dmatrix('y')

w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

output = nonlin(T.dot(x,w)+b)
cost = T.nnet.binary_crossentropy(output, y).mean() + 0.01 * (w ** 2).sum()

learn_vars = [w, b]
updates = [(v, v - LEARNING_RATE * T.grad(cost, v)) for v in learn_vars]

train = theano.function(inputs=[x,y], outputs=[cost], updates=updates)
predict = theano.function(inputs=[x], outputs=[output])

for i in range(training_steps):
    err = train(D[0], D[1])
    if i % (training_steps/10) == 0:
        print(err)
