import theano
from theano import tensor as T
import numpy as np

from theano_stuff.binary_generator import NumberGenerator

rng = np.random.RandomState(42)
ng = NumberGenerator(lambda x, y: x + y)

def _transfer(x):
    return T.tanh(x)

def _get_weights(n_in, n_out, name, low=-0.5, high=0.5):
    """ Initialize a weight matrix of size `n_in` by `n_out` with random values from `low` to `high` """
    return theano.shared(np.random.uniform(size=(n_in, n_out), low=low, high=high), name=name)

def generate_rnn(n_in, n_out, n_hidden=50):
    """ Recurrent network with input dim `n_in` and output dim `n_out` """

    # input, target, initial hidden state, learning rate
    input = T.tensor4(name='input')
    target = T.tensor4(name='target')
    hidden_state_low = T.tensor3(name='hidden')
    hidden_state_high = T.tensor3(name='hidden')
    learning_rate = T.scalar(name='learning_rate')

    # initialize weight matrices and biases
    low_w_hidden = _get_weights(n_hidden, n_hidden, 'low_w_hidden')
    low_w_in, low_b_in = _get_weights(n_in, n_hidden, 'low_w_in'), _get_weights(1, n_hidden, 'low_b_in')
    low_w_out, low_b_out = _get_weights(n_hidden, n_hidden, 'low_w_out'), _get_weights(1, n_hidden, 'low_b_out')

    low_weights = [low_w_hidden, low_w_in, low_w_out, low_b_in, low_b_out]

    high_w_hidden = _get_weights(n_hidden, n_hidden, 'high_w_hidden')
    high_b_in = _get_weights(1, n_hidden, 'high_b_in')
    high_w_out, high_b_out = _get_weights(n_hidden, n_out, 'high_w_out'), _get_weights(1, n_out, 'high_b_out')

    high_weights = [high_w_hidden, high_w_out, high_b_in, high_b_out]

    def low_step(input, prev_h, w_hidden, w_in, w_out, b_in, b_out):
        h_new = _transfer(T.dot(input, w_in) + T.dot(prev_h, w_hidden) + b_in)
        y_new = _transfer(T.dot(h_new, w_out) + b_out)
        return y_new, h_new

    def high_step(input, prev_h, w_hidden, w_out, b_in, b_out):
        h_new = _transfer(input + T.dot(prev_h, w_hidden) + b_in)
        y_new = _transfer(T.dot(h_new, w_out) + b_out)
        return y_new, h_new

    # construct recurrent part
    (intermediate, _), _ = theano.scan(low_step, sequences=input, outputs_info=[None, hidden_state_low], non_sequences=low_weights, n_steps=input.shape[0], name='low_network')
    (output, _), _ = theano.scan(high_step, sequences=intermediate, outputs_info=[None, hidden_state_high], non_sequences=high_weights, n_steps=intermediate.shape[0], name='high_network')

    # compute error (with l2 norm of weights) and updates
    err = ((output - target) ** 2).sum() / input.shape[0]
    updates = [(x, x - learning_rate * T.grad(err, x)) for x in low_weights + high_weights]

    # functions for training and testing
    train = theano.function([input, target, hidden_state_low, hidden_state_high, learning_rate], err, updates=updates)
    test = theano.function([input, hidden_state_low, hidden_state_high], output)

    return train, test

def test_net(n_epochs=1000, n_train=10000, n_test=1):

    LEARNING_RATE = 0.01
    DECAY = 0.99
    MINI_BATCH = 100

    TEST_BATCH = 10

    N_HIDDEN = 4
    HIDDEN_STATE = np.zeros(shape=(MINI_BATCH, 1, N_HIDDEN))
    HIDDEN_STATE_TEST = np.zeros(shape=(TEST_BATCH, 1, N_HIDDEN))

    train, test = generate_rnn(2, 1, n_hidden=N_HIDDEN)

    for epoch in range(n_epochs):
        print('\nEpoch %d (learning rate = %f)\n-------' % (epoch, LEARNING_RATE))

        costs = 0

        for nums, sums in ng.generate_data(n_train // MINI_BATCH, size=MINI_BATCH):
            err = train(nums, sums, HIDDEN_STATE, HIDDEN_STATE, LEARNING_RATE)
            costs += err

        LEARNING_RATE *= DECAY

        for nums, sums in ng.generate_data(n_test, size=TEST_BATCH):
            preds = test(nums, HIDDEN_STATE_TEST, HIDDEN_STATE_TEST)
            for a, b, sum, pred in zip(ng.as_binary_string(nums[:,:,:,0]), ng.as_binary_string(nums[:,:,:,1]), ng.as_binary_string(sums), ng.as_binary_string(preds)):
                print('Input: %s and %s\n\t\t  Correct: %s\n\t\tPredicted: %s' % (a, b, sum, pred))

        print('Total error: %f' % costs)

if __name__ == '__main__':
    test_net()
    for nums, sums in ng.generate_data(1, size=5):
        print(ng.as_digits(nums[:,:,:,0]))
        print(ng.as_digits(nums[:,:,:,1]))
        print(ng.as_digits(sums))