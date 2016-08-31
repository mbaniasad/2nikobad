# -*- coding: utf-8 -*-
"""
Simple example using a Dynamic RNN (LSTM) to classify IMDB sentiment dataset.
Dynamic computation are performed over sequences with variable length.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

import sys
from  LSTMconfig import LSTMconfig
import traceback

def runDLstm(cl):
    print("running Dynamic_lstm_on_", cl.setting_name())
    # IMDB Dataset loading
    train, test, _ = imdb.load_data(path=cl.dataset_path, n_words=cl.number_of_words_used_in_embedding,
                                    valid_portion=0.1)
    trainX, trainY = train
    testX, testY = test

    # Data preprocessing
    # NOTE: Padding is required for dimension consistency. This will pad sequences
    # with 0 at the end, until it reaches the max sequence length. 0 is used as a
    # masking value by dynamic RNNs in TFLearn; a sequence length will be
    # retrieved by counting non zero elements in a sequence. Then dynamic RNN step
    # computation is performed according to that length.
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, 100])
    # Masking is not required for embedding, sequence length is computed prior to
    # the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
    net = tflearn.embedding(net, input_dim=cl.number_of_words_used_in_embedding, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=cl.dropout, dynamic=True)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.001,
                             loss=cl.loss)

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32, n_epoch=cl.n_epoch)

    model.save("./5-SAVED_MODELS/"+cl.setting_name())





tb = ''
try:
    print(sys.argv)
    cl = LSTMconfig(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])
    runDLstm(cl)
except:
    tb = traceback.format_exc()
    raise 
else:
    tb = "No error"
finally:
    print(tb)

