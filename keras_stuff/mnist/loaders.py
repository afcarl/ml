import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """ Load MNIST data from `path` (`train` is default, otherwise `t10k`) """

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def plot_image(*imgs):
    if len(imgs) == 1:
        image = plt.imshow(imgs[0].reshape(28,28), cmap='Greys', interpolation='nearest')
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(len(imgs)):
            ax[i].imshow(imgs[i].reshape(28,28), cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()