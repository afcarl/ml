"""
Two theano functions: one to train rnn-rbm, other to generate sample sequences from it
For training: Given v^t, the rnn hidden state u^t and associated b_v^t, b_h^t parameters are deterministic and can be
readily computed for each training sequence. SGD update on the parameters estimated via CD on individual time steps
"""

import glob
import os
import sys

import theano
import theano.tensor as T
import numpy as np

import pylab

def shared_normal(num_rows, num_cols, scale=1):
    return theano.shared(np.random.normal(scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))


def build_rbm(v, W, bv, bh, k):
    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=theano.config.floatX)

        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX)

        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v], n_steps=k)

    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample,  cost, monitor, updates


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    """ Build the model """

    # model parameters
    recurrent_weight_scale = 0.0001
    rbm_weight_scale = 0.01

    # model weights
    W = shared_normal(n_visible, n_hidden, rbm_weight_scale)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, recurrent_weight_scale)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, recurrent_weight_scale)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, recurrent_weight_scale)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, recurrent_weight_scale)
    bu = shared_zeros(n_hidden_recurrent)

    params = [W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu]

    v = T.matrix() # training sequence
    u0 = T.zeros((n_hidden_recurrent,)) # initial value for rnn hidden units

    def recurrence(v_t, u_tml):
        bv_t = bv + T.dot(u_tml, Wuv)
        bh_t = bh + T.dot(u_tml, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t, bh_t, k=25)

        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tml, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    (u_t, bv_t, bh_t), updates_train = theano.scan(lambda v_t, u_tml, *_: recurrence(v_t, u_tml), sequences=v, outputs_info=[u0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:], k=15)

    updates_train.update(updates_rbm)

    (v_t, u_t), updates_generate = theano.scan(lambda u_tml, *_: recurrence(None, u_tml), outputs_info=[None, u0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate)

class RnnRbm:
    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.001, r=(21, 109), dt=0.5):
        self.r = r
        self.dt = dt

        (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate) = build_rnnrbm(r[1] - r[0], n_hidden, n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params, gradient)))
        self.train_function = theano.function([v], monitor, updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)

    def train(self, files, batch_size=100, num_epochs=200):
        assert len(files) > 0, 'Training set is empty! (did you download the data file?)'

        dataset = [midiread(f, self.r, self.dt) for f in files]

        try:
            for epoch in range(num_epochs):
                np.random.shuffle(dataset)
                costs = []

                for s, sequence in enumerate(dataset):
                    for i in range(0, len(sequence), batch_size):
                        cost = self.train_function(sequence[i:i + batch_size])
                        costs.append(cost)

                print('Epoch %i/%i' % (epoch+1, num_epochs))
                print(np.mean(costs))
                sys.stdout.flush()
        except KeyboardInterrupt:
            print('Interrupted by user.')

    def generate(self, filename, show=True):
        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)

        if show:
            extent = (0, self.dt * len(piano_roll))
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto', interpolation='nearest', cmap=pylab.cm.gray_r, extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')

def test_rnnrbm(batch_size=100, num_epochs=200):
    model = RnnRbm()

    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'Nottingham', 'train', '*.mid')
    model.train(glob.glob(re), batch_size=batch_size, num_epochs=num_epochs)

    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    model.generate('sample2.mid')
    pylab.show()