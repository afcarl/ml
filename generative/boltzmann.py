import gzip
import pickle
import timeit

try:
    import PIL.Image as Image
except:
    import Image

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import os

# This takes a long time to train...


class RBM(object):
    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(42)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            x = np.sqrt(6. / (n_hidden + n_visible))
            initial_W = np.asarray(numpy_rng.uniform(low=-4*x, high=4*x, size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='hbias', borrow=True)

        if vbias is None:
            vbias = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), name='vbias', borrow=True)

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([pre_sigmoid_nvs, nv_mean, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(self.gibbs_hvh, outputs_info=[None] * 5 + [chain_start], n_steps=k)

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        ''' stochastic approximation of pseudo-likelihood '''
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # free energy with bit set
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)

        # free energy with bit flipped
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        return cross_entropy


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)

    if data_dir == '' and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], 'data', dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def test_rbm(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20, n_chains=20, n_samples=10, output_folder='rbm_plots', n_hidden=500):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]

    index = T.lscalar()
    x = T.matrix('x')

    rng = np.random.RandomState(42)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    rbm = RBM(input=x, n_visible=28*28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function([index], cost, updates=updates, givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]}, name='train_rbm')

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        print('Training epoch %d')

        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
            print('  -> %d / %d' % (batch_index, n_train_batches))

        print('cost is ' % np.mean(mean_cost))

        plotting_start = timeit.default_timer()
        image = Image.fromarray(tile_raster_images(X=rbm.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))

        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time

    print('Training took %f minutes' % (pretraining_time / 60.))
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(np.asarray(test_set_x.get_value(borrow=True)[test_idx:test_idx+n_chains], dtype=theano.config.floatX))

    plot_every = 1000

    ([presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples], updates) = theano.scan(rbm.gibbs_vhv, outputs_info=[None] * 5 + [persistent_vis_chain], n_steps=plot_every)

    updates.update({persistent_vis_chain: vis_samples[-1]})

    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]], updates=updates, name='sample_fn')

    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

    for idx in range(n_samples):
        vis_mf, vis_sample = sample_fn()
        print('... plotting sample %d' % idx)
        image_data[29*idx:29*idx+28, :] = tile_raster_images(X=vis_mf, img_shape=(28, 28), tile_shape=(1, n_chains), tile_spacing=(1, 1))

        image = Image.fromarray(image_data)
        image.save('samples.png')

        os.chdir('../')

if __name__=='__main__':
    test_rbm(n_hidden=10)
