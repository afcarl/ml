from __future__ import print_function
# import six.modes.cPickle as pickle
import pickle

from keras.layers import GRU
from keras.models import Graph

import gensim

class Vocab:

    def __init__(self):
        self._token_counts = dict()
        self._compiled = True
        self.token2id = dict()
        self.id2token = dict()

    def add(self, text):
        self._compiled = False

        tok = gensim.utils.tokenize(text, to_lower=True)
        for t in tok:
            if t in self._token_counts:
                self._token_counts[t] += 1
            else:
                self._token_counts[t] = 1

    def _compile(self):
        if self._compiled: return

        self.token2id = dict([(t, i) for i, t in enumerate(self._token_counts)])
        self.id2token = dict([(i, t) for i, t in enumerate(self._token_counts)])

    def save(self):
        self._compile()


def getModel(vocab):
    model = Graph()
    model.add_input('question')
    model.add_node('question_lstm', GRU())

    model.add_input('answer')