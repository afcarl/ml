from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file

import pickle
import numpy as np
import random
import sys


class CharRNN(object):
    def __init__(self, corpus=None, step=3, max_len=20, n_layers=3, n_neurons=128, dropout=0.2, weight_file='char_rnn.hdf5'):
        self.max_len = max_len
        self.step = step
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.weight_file = weight_file

        if corpus is not None:
            self.corpus = corpus

            # build dictionary
            self.chars = set(self.corpus)
            print('total chars:', len(self.chars))
            self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
            self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

            # cut the text in semi-redundant sequences of maxlen characters
            sentences = []
            next_chars = []

            for i in range(0, len(self.corpus)-self.max_len, self.step):
                sentences.append(self.corpus[i:i+self.max_len])
                next_chars.append(self.corpus[i+self.max_len])

            # vectorization
            print('Vectorization...')
            self.X = np.zeros((len(sentences), self.max_len, len(self.chars)), dtype=np.bool)
            self.y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    self.X[i, t, self.char_indices[char]] = 1
                self.y[i, self.char_indices[next_chars[i]]] = 1

            # save for future use
            self._save(self.char_indices, 'char_indices')
            self._save(self.indices_char, 'indices_char')
            self._save(self.chars, 'chars')
        else:
            # future use
            self.char_indices = self._load('char_indices')
            self.indices_char = self._load('indices_char')
            self.chars = self._load('chars')

        # build the model
        print('Building model...')
        self.model = Sequential()

        # add first layer
        self.model.add(LSTM(self.n_neurons, return_sequences=True, input_shape=(self.max_len, len(self.chars))))
        self.model.add(Dropout(self.dropout))

        # add the rest of the layres
        for i in range(self.n_layers-1):
            self.model.add(LSTM(self.n_neurons, return_sequences=False if i == self.n_layers-2 else True))
            self.model.add(Dropout(self.dropout))

        # final layer: used to predict next character
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))

        # rmsprop is a good optimizer for rnn's, apparently
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # if weights already exist, load them
        try:
            self.model.load_weights(weight_file)
        except:
            pass

    # helper function to sample an index from a probability array
    @staticmethod
    def _sample(a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    def _save(self, what, name):
        file_name = '%s.pkl' % name
        pickle.dump(what, open(file_name, 'w'))

    def _load(self, what):
        file_name = '%s.pkl' % what
        return pickle.load(open(file_name, 'r'))

    # train the model, output generated text after each iteration
    def train(self, n_iter=60, batch_size=128):
        for iteration in range(1, n_iter):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            self.model.fit(self.X, self.y, batch_size=batch_size, nb_epoch=1)
            self.evaluate()
            self.model.save_weights(self.weight_file, overwrite=True)

    # evaluate the model
    def evaluate(self):
        start_index = random.randint(0, len(self.corpus)-self.max_len-1)

        for diversity in [0.2, 0.6, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.corpus[start_index: start_index+self.max_len]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, self.max_len, len(self.chars)))
                for t, char in enumerate(sentence):
                    x[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x, verbose=0)[0]
                next_index = self._sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()

            print()

    # Generate text with user input
    def user_input(self):
        sentence = 'not none'

        while len(sentence) > 0:
            generated = ''
            sentence = raw_input('Seed text: ')
            if len(sentence) < self.max_len:
                continue
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for diversity in [0.2, 0.6, 1.0, 1.2]:
                print()
                print('----- diversity:', diversity)

                for i in range(400):
                    x = np.zeros((1, len(sentence), len(self.chars)))
                    for t, char in enumerate(sentence):
                        x[0, t, self.char_indices[char]] = 1.

                    preds = self.model.predict(x, verbose=0)[0]
                    next_index = self._sample(preds, diversity)
                    next_char = self.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()

TRAIN_ON_CORPUS = False

# download corpus
if TRAIN_ON_CORPUS:
    print('Downloading corpus...')
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('corpus length:', len(text))

    char_rnn = CharRNN(corpus=text)
    char_rnn.train(n_iter=60)
else:
    char_rnn = CharRNN()
    char_rnn.user_input()
