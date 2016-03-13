import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(42)


class GenerativeBackprop(object):
    def __init__(self, n_in, n_hidden, n_out, learning_rate=0.01):
        self.learning_rate = learning_rate

        X = T.vector('X')
        Y = T.vector('Y')

        self.dims = [n_in] + n_hidden + [n_out]
        self.weights = [self._layer(self.dims[i-1], self.dims[i]) for i in range(1, len(self.dims))]

        a = X
        for weight in self.weights:
            a = self._nonlin(T.dot(a, weight))

        self.output = a
        self.cost = T.mean(T.sqr(self.output - Y))
        self.updates = [(x, x - learning_rate * T.grad(cost=self.cost, wrt=x)) for x in self.weights]

        # basic multilayer perceptron
        self._train = theano.function(inputs=[X,Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
        self._test = theano.function(inputs=[X], outputs=self.output, allow_input_downcast=True)

    def train(self, data, targets, n_epochs):
        for epoch in range(n_epochs):
            for d, t in zip(data, targets):
                self._train(d, t)

    def test(self, data):
        r = []
        for d in data:
            r.append(self._test(d))
        return r

    def generate(self, output, train_len=1000):
        te_x = theano.shared(value=np.asarray(rng.uniform(low=0, high=1, size=(output.shape[0], self.dims[0]))), name='te_x', borrow=True)
        te_y = theano.shared(value=output, name='te_y')

        a = te_x
        for weight in self.weights:
            a = self._nonlin(T.dot(a, weight))

        cost = T.mean(T.sqr(a - te_y))
        updates = [(x, x - self.learning_rate * T.grad(cost, x)) for x in [te_x]]

        re_encode = theano.function(inputs=[], outputs=[], updates=updates, allow_input_downcast=True)

        for i in range(train_len):
            re_encode()

        return te_x.get_value()

    @staticmethod
    def _layer(n_in, n_out):
        np_array = np.asarray(rng.uniform(low=-1, high=1, size=(n_in, n_out)), dtype=theano.config.floatX)
        return theano.shared(value=np_array, name='W', borrow=True)

    @staticmethod
    def _nonlin(x):
        return T.tanh(x)

# testing
if __name__=='__main__':
    gb = GenerativeBackprop(n_in=2, n_hidden=[10, 10], n_out=1)
    x, y = np.asarray([[0,1], [1,0], [1,1], [0,0]]), np.asarray([[1], [1], [0], [0]])

    print('Before training:')
    print(gb.generate(np.asarray([[1], [0]])))

    # train the model
    gb.train(x, y, 10000)

    print('After training:')
    print(gb.generate(np.asarray([[1], [0]])))
