import theano
from theano import tensor as T
import numpy as np

import textwrap

'''
Description: Given two binary numbers, train an rnn to add them together. Lessons learned:
1. This is way faster and converges much better with minibatches instead of individual units (probably duh)
2. I don't think this can be done with one hidden unit, because we're trying to learn more complex transitions.
   It sort of makes sense in my head, when you think about mapping to a hyperplane / activation functions.

    (carry, input 1, input 2) -> (new carry, output) (inputs 1 and 2 could be swapped)
    (1, 1, 1) -> (1, 1)
    (1, 1, 0) -> (1, 0)
    (1, 0, 0) -> (0, 1)
    (0, 1, 1) -> (1, 0)
    (0, 1, 0) -> (0, 1)
    (0, 0, 0) -> (0, 0)

3. RNN's are difficult to learn. theano.scan() is starting to make some more sense though.
4. I think something like this would be interesting to try on an analog platform? Would need to figure out math though.
'''

rng = np.random.RandomState(42)

SIZE = 8
MIN, MAX = 0, 2 ** (SIZE - 1)


def to_binary(x):
    m = ('{0:b}').format(x)
    y = [int(s) for s in reversed(m + '0' * (SIZE - len(m)))]
    return np.asarray(y, theano.config.floatX)


def as_binary_string(x):
    return [s[::-1] for s in textwrap.wrap(''.join('1' if i > 0.5 else '0' for i in x.T.flatten()), SIZE)]


def func(x, y):
    return x + y


def generate_data(n, size=1):
    """ Generator to generate `n` data instances, each consisting of two input strings and one output string """
    for i in range(n):
        a, b = [rng.randint(MIN, MAX) for j in range(size)], [rng.randint(MIN, MAX) for j in range(size)]

        nums = np.asarray([[to_binary(x), to_binary(y)] for x, y in zip(a, b)], dtype=theano.config.floatX)
        sums = np.asarray([[to_binary(func(x, y)) for x, y in zip(a, b)]], dtype=theano.config.floatX)

        yield np.asarray([nums], theano.config.floatX).transpose(3,1,0,2), np.asarray([sums], theano.config.floatX).transpose(3,2,1,0)


class RNN:
    def __init__(self, n_in, n_out, n_hidden=50):
        """ Recurrent network with input dim `n_in` and output dim `n_out` """

        # input, target, initial hidden state, learning rate
        input = T.tensor4(name='input')
        target = T.tensor4(name='target')
        hidden_state = T.tensor3(name='hidden')
        learning_rate = T.scalar(name='learning_rate')

        # initialize weight matrices and biases
        w_hidden = self._get_weights(n_hidden, n_hidden, 'w_hidden')
        w_in = self._get_weights(n_in, n_hidden, 'w_in')
        w_out = self._get_weights(n_hidden, n_out, 'w_out')
        b_in = self._get_weights(1, n_hidden, 'b_in')
        b_out = self._get_weights(1, n_out, 'b_out')

        self.weights = [w_hidden, w_in, w_out, b_in, b_out]

        # step function
        def step(input, prev_h, w_hidden, w_in, w_out, b_in, b_out):
            h_new = self._transfer(T.dot(input, w_in) + T.dot(prev_h, w_hidden) + b_in)
            y_new = self._transfer(T.dot(h_new, w_out) + b_out)
            return y_new, h_new

        # construct recurrent part
        (output, intermediate_states), _ = theano.scan(step, sequences=input, outputs_info=[None, hidden_state], non_sequences=self.weights, n_steps=input.shape[0], name='network')

        # compute error (with l2 norm of weights) and updates
        err = ((output - target) ** 2).sum() / input.shape[0]

        # push the weights towards binary values (doesn't really help but it is interesting)
        # err += 0.3 * abs(output * (1 - output)).sum() / input.shape[0]
        self.updates = [(x, x - learning_rate * T.grad(err, x)) for x in self.weights]

        # functions for training and testing
        self.train = theano.function([input, target, hidden_state, learning_rate], err, updates=self.updates)
        self.test = theano.function([input, hidden_state], output)
        self.introspect = theano.function([input, hidden_state], [output, intermediate_states])

    def _transfer(self, x):
        """ Default transfer function to use """
        return T.tanh(x)

    def _get_weights(self, n_in, n_out, name, low=-0.5, high=0.5):
        """ Initialize a weight matrix of size `n_in` by `n_out` with random values from 0 to 1 """
        return theano.shared(np.asarray(rng.rand(n_in, n_out) * (high - low) - low, dtype=theano.config.floatX), name=name)


def test_net(n_epochs=1000, n_train=10000, n_test=1):

    LEARNING_RATE = 0.01
    DECAY = 0.98
    MINI_BATCH = 100

    TEST_BATCH = 10

    N_HIDDEN = 4
    HIDDEN_STATE = np.zeros(shape=(MINI_BATCH, 1, N_HIDDEN), dtype=theano.config.floatX)
    HIDDEN_STATE_TEST = np.zeros(shape=(TEST_BATCH, 1, N_HIDDEN), dtype=theano.config.floatX)

    rnn = RNN(2, 1, n_hidden=N_HIDDEN)

    for epoch in range(n_epochs):
        print('\nEpoch %d (learning rate = %f)\n-------' % (epoch, LEARNING_RATE))

        costs = 0

        for nums, sums in generate_data(n_train // MINI_BATCH, size=MINI_BATCH):
            err = rnn.train(nums, sums, HIDDEN_STATE, LEARNING_RATE)
            costs += err

        LEARNING_RATE *= DECAY

        for nums, sums in generate_data(n_test, size=TEST_BATCH):
            preds = rnn.test(nums, HIDDEN_STATE_TEST)
            for a, b, sum, pred in zip(as_binary_string(nums[:,:,:,0]), as_binary_string(nums[:,:,:,1]), as_binary_string(sums), as_binary_string(preds)):
                print('Input: %s + %s\n\t\t  Correct: %s\n\t\tPredicted: %s' % (a, b, sum, pred))

        print('Total error: %f' % costs)

if __name__ == '__main__':
    test_net()
    # for n, sum in generate_data(1, size=100):
    #     print(n.shape)
    #     print(sum.shape)
    #     print(as_binary_string(n[:,:,:,0]))
    #     print(as_binary_string(n[:,:,:,1]))
    #     print(as_binary_string(sum))