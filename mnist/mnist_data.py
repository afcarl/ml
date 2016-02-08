from loaders import *
from neural_networks import *

MNIST_PATH = './data'
X_train, y_train = load_mnist(MNIST_PATH, kind='train')
X_test, y_test = load_mnist(MNIST_PATH, kind='t10k')
nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, l2=0.1, l1=0.0, epochs=100, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)

nn.fit(X_train, y_train, print_progress=True)

print nn.predict(X_test)
print y_test