import theano
import theano.tensor as T
import numpy as np

rng = np.random.RandomState(42)


def l2_cost(output, target):
    return ((output - target) ** 2).sum() / output.shape[0]


def shared_matrix(n_in, n_out, name=None, low=-1, high=1):
    return theano.shared(np.random.uniform(size=(n_in, n_out), low=low, high=high), name=name)


class RNN:
    def __init__(self, n_in, n_out, name=None, n_hidden=50, learning_rate=0.01, transfer_func=T.tanh):
        if name is None:
            name = 'rnn'

        # model variables
        input = T.tensor4(name + '_input')
        hidden_state = T.zeros(shape=(input.shape[0], 1, n_hidden))

        # parameters
        w_in = shared_matrix(n_in, n_hidden, name=name+'_w_in')
        w_hidden = shared_matrix(n_hidden, n_hidden, name=name+'_w_hidden')
        w_out = shared_matrix(n_hidden, n_out, name=name+'w_out')
        b_in = shared_matrix(1, n_hidden, name=name+'_b_in')
        b_out = shared_matrix(1, n_out, name=name+'_b_out')

        weights = [w_hidden, w_in, w_out, b_in, b_out]

        # recurrent step function
        def step(input, prev_hidden_state, w_hidden, w_in, w_out, b_in, b_out):
            h_new = transfer_func(T.dot(input, w_in) + T.dot(prev_hidden_state, w_hidden) + b_in)
            y_new = transfer_func(T.dot(h_new, w_out) + b_out)
            return y_new, h_new

        (output, _), _ = theano.scan(step, sequences=input, outputs_info=[None, hidden_state], non_sequences=weights, n_steps=input.shape[0], name=name+'_scan')

        # parameters that we want to save
        self.input = input
        self.output = output
        self.hidden_state = hidden_state
        self.weights = weights


def generate_data(epochs=10, n=1000):
    max = 10

    for epoch in range(epochs):
        a, b = rng.randint(0, max, (1, n)), rng.randint(0, max, (1, n))
        y = a == b

        m = a + b

        for i in range(0, 2*(max-1)+1):
            idx = m == i
            yield np.asarray([[[1] * int(a_x) + [0] * int(b_x)] for a_x, b_x in zip(a[idx], b[idx])]), np.asarray(y[idx], dtype=theano.config.floatX)


def build():
    layer_1 = RNN(1, 1, n_hidden=10)
    params = layer_1.weights

    # input, output and target of the network
    input = layer_1.input
    output = layer_1.output[-1]
    target = T.tensor3('target')

    # model parameters
    learning_rate = 0.01

    # cost function
    cost = l2_cost(output, target)
    updates = [(p, p - learning_rate * T.grad(cost, p)) for p in params]

    # training and testing functions
    train = theano.function([layer_1.input, target], [cost], updates=updates)
    test = theano.function([layer_1.input], [layer_1.output])

    for x, y in generate_data():
        train(x, y)

        for xx, yy in generate_data(epochs=1, n=10):
            print(test(xx, yy))



if __name__=='__main__':
    # build()
    for x, y in generate_data():
        # print(np.asarray([x]).transpose(3, 1, 0, 2))
        print(np.asarray([[y]]).transpose(2, 0, 1))