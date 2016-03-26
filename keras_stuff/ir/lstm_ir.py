from __future__ import print_function

import os

# parameters
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Convolution1D, MaxPooling1D, LSTM, Dropout, Dense
from keras.models import Sequential

from keras_stuff.ir.text_utils import YahooDictionary

WEIGHTS_FILE = 'ir_rnn_weights.h5'
SOURCE_FILE = '/media/moloch/HHD/MachineLearning/data/yahoo_qa/FullOct2007.xml.part1'
VOCAB_SIZE = 20000

# make a new dictionary
d = YahooDictionary(SOURCE_FILE, vocab_size=VOCAB_SIZE)

answers, subjects, contents, categories = d.get_docs()

LOAD_WEIGHTS = False
BATCH_SIZE = 32
EPOCHS = 200

# build the network
HIDDEN_NEURONS = 50
EMBEDDING_SIZE = 50
LSTM_DROPOUT_U = 0.15
LSTM_DROPOUT_W = 0.25

# predict question given answer
print('Building network...')
aq_model = Sequential()
aq_model.add(Embedding(d.vocab_size, EMBEDDING_SIZE, input_length=d.max_ans_len))

aq_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu',
                        subsample_length=1))
aq_model.add(MaxPooling1D(pool_length=2))
aq_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))
aq_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W, go_backwards=True))
aq_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))

aq_model.add(Dropout(0.25))
aq_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu',
                           subsample_length=1))
aq_model.add(MaxPooling1D(pool_length=2))
aq_model.add(LSTM(50, return_sequences=False))
aq_model.add(Dense(d.max_cont_len))

aq_model.compile(optimizer='adam', loss='cosine_proximity')

if os.path.isfile(WEIGHTS_FILE) and LOAD_WEIGHTS:
    print('Loading weights from "%s"' % WEIGHTS_FILE)
    aq_model.load_weights(WEIGHTS_FILE)
else:
    print('Training')

    # callbacks (e.g. saving the model)
    callbacks = []
    callbacks.append(ModelCheckpoint(filepath=WEIGHTS_FILE, verbose=1, save_best_only=True))

    aq_model.fit([answers], contents, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05,
                 show_accuracy=True, callbacks=callbacks)
