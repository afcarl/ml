"""
ANALOG CIRCUIT NEURAL NETWORK SIMULATION
Simulation of training an analog circuit-based neural network using theano
"""

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt

# theano.config.exception_verbosity = 'high'

# physical parameters
v_dd =          2.5 # volts
i_th =          1 # picoamps
u_t =           0.02585 # volts, thermal voltage
kappa =         0.7 # constant (could be different for pfet / nfet)
v_th_nfet =     0.55 # volts
v_th_pfet =     0.65 # volts

max_current =   1e7

v_2 =           2.
# v_1 =           v_2 + v_dd * (1 - (1 / kappa)) - v_th_pfet - v_th_nfet
v_1 =           0.5

# learning parameters
lr =            0.001 # learning rate

np.random.seed(42)


def nfet(v_g, v_s, i_th=i_th, v_th=v_th_nfet, kappa=kappa, v_dd=v_dd, u_t=u_t):
    """
    :param v_g: voltage on gate
    :param v_s: voltage on source
    :return: current at drain
    """

    return i_th * T.exp((kappa * (v_g - v_th) - v_s) / u_t)


def pfet(v_g, v_s, i_th=i_th, v_th=v_th_pfet, kappa=kappa, v_dd=v_dd, u_t=u_t):
    """
    :param v_g: voltage on gate
    :param v_s: voltage on source
    :return: drain current
    """

    return i_th * T.dot(T.exp(T.maximum(v_dd - T.maximum(v_s, 0), 0) / u_t), T.exp((kappa * T.maximum(v_dd - T.maximum(v_g, 0) - v_th, 0)) / u_t))


def vmm(n_in, n_out, v_s, low=1, high=1.5, i_th=i_th, kappa=kappa, u_t=u_t, v_dd=v_dd):
    """
    :param n_in: number of input dimensions
    :param n_out: number of output dimensions
    :param v_s: input vector (voltages)
    :return: floating gate voltage matrix, bias current matrix, pfet current matrix (function of fg matrix)
    """

    # initialize random floating gate weights for each element in vmm part
    v_fg = theano.shared(np.random.uniform(low=low, high=high, size=(n_in, n_out)), name='v_fg')

    # bias current
    i_bias = theano.shared(np.random.uniform(low=0, high=1e-10, size=(1, n_out)), name='i_bias')

    # physical equation for the output current
    i_pfet = pfet(v_fg, v_s, i_th=i_th, kappa=kappa, v_dd=v_dd, u_t=u_t) + T.minimum(i_bias.reshape((1, 1, n_out)), max_current)

    return v_fg, i_bias, i_pfet


def sinh(v_out, v_1=v_1, v_2=v_2, i_th=i_th, v_th_p=v_th_pfet, v_th_n=v_th_nfet, kappa=kappa, v_dd=v_dd, u_t=u_t):
    return pfet(v_1, v_out) - nfet(v_2, v_out)


def sinh_approx(v_out, i_th=i_th): # approximating function, given preset parameters
    return i_th * 4.1571e-5 * T.sinh(39.1503 * (v_out - 1.285))


def asinh(i_in, i_th=i_th):
    # this seems like a pretty good approximation of the asinh circuit (given the parameters i was using)
    return T.minimum(1.285 + 0.0255426 * T.arcsinh(2.40552e4 * i_in / i_th), v_dd)


def build(test_in, test_out):
    # input and output vectors
    x = T.matrix(name='x', dtype=theano.config.floatX) # v_s
    y = T.matrix(name='y', dtype=theano.config.floatX) # v_o

    n_hidden = 3

    weights = list()

    v_1, b_1, a_1 = vmm(2, n_hidden, x)
    z_1 = asinh(a_1)
    weights += [v_1, b_1]

    v_2, b_2, a_2 = vmm(n_hidden, 1, z_1)
    z_2 = asinh(a_2)
    weights += [v_2, b_2]

    # variable to hold the final output
    y_hat = z_2

    cost = 0.5 * ((y - y_hat) ** 2).sum() # minimize l2 norm
    updates = [(p, p - lr * T.grad(cost, p)) for p in weights]

    # training and testing functions
    print('Building functions...')

    train = theano.function([x, y], cost, updates=updates)
    test = theano.function([x], [z_1, a_2, y_hat])

    print('Initial:')
    print(test(test_in))

    for i in range(10000):
        train(test_in, test_out)

    print('Final:')
    print(test(test_in))

    print('Values:')
    for v in [v_1, b_1, v_2, b_2]:
        print(v.get_value())

if __name__=='__main__':
    low_i, high_i = 1e-12, 1e-10
    low_v, high_v = 2, 2.5

    test_in = np.asarray([[low_v, high_v], [high_v, low_v], [low_v, low_v], [high_v, high_v]])
    # test_out = np.asarray([[[low_i], [high_i], [low_i], [high_i]]])
    test_out = np.asarray([[1], [1], [2], [2]])

    # below: testing the asinh function
    # v_in = T.scalar('v_in')
    # i_out = sinh(v_in)
    # v_out = asinh(i_out)
    #
    # get_i = theano.function([v_in], v_out)
    #
    # for v_i in np.arange(-2.5, 2.5, 0.01):
    #     print(v_i, get_i(v_i))

    build(test_in, test_out)
