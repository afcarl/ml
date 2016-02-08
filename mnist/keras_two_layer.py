import theano
from loaders import *

MNIST_PATH = './data'
X_train, y_train = load_mnist(MNIST_PATH, kind='train')
X_test, y_test = load_mnist(MNIST_PATH, kind='t10k')

theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

from keras.utils import np_utils

y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import SGD

np.random.seed(1)
model = Sequential()
model.add(Dense(input_dim=X_train.shape[1],
               output_dim=50,
               init='uniform',
               activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))
model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train_ohe, nb_epoch=50, batch_size=300, verbose=1, validation_split=0.1, show_accuracy=True)

print model.predict_classes(X_test, verbose=0)
print y_test