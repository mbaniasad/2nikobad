from gensim.models import Doc2Vec

# numpy
import numpy
import os
import tflearn
from tflearn.data_utils import to_categorical
import tensorflow as tf


from gensim.models import Doc2Vec

# numpy
import numpy

import tflearn
from tflearn.data_utils import to_categorical

class config:
    
    def __init__(self,dataset_path,optimizer,loss,number_of_cells):
        self.dataset_path = dataset_path
        self.loss= loss
        self.optimizer=optimizer
        self.dropout = 0.5
        self.number_of_cells=number_of_cells
    def setting_name(self):
        return 'doc2vecLSTM'+str(cl.number_of_cells)+'ds-'+self.dataset_path +'optimizer-'+self.optimizer+'loss-'+str(self.loss)


doc2vec_configs = [
                   # config( 'imdb.d2v', 'RMSProp', 'categorical_crossentropy'),
                   # config( 'imdb.d2v', 'Momentum', 'categorical_crossentropy'),
                   # config( 'imdb.d2v', 'AdaGrad', 'categorical_crossentropy'),    
                   # config( 'imdb.d2v', 'Ftrl', 'categorical_crossentropy'),
                   # config( 'imdb.d2v', 'AdaDelta', 'categorical_crossentropy'),
                   # config( 'imdb.d2v', 'sgd', 'categorical_crossentropy'),
                   # config( 'GAR-cls-acl10.d2v', 'adam', 'categorical_crossentropy',1000),
                   config( 'GAR-cls-acl10.d2v', 'adam', 'categorical_crossentropy',10000)
                   ]




for cl in doc2vec_configs:
    with tf.Graph().as_default():
        print("running doc2vecRegression_on_"+cl.setting_name())
        # loading dov2vec model
        model = Doc2Vec.load('./4-PKLED/'+cl.dataset_path)

        number_of_samples = 6000;


        trainX = numpy.zeros((number_of_samples,1, 100))
        train_labels = numpy.zeros(number_of_samples)

        for i in range(3000):
            prefix_train_pos = 'TRAIN_POS_' + str(i)
            prefix_train_neg = 'TRAIN_NEG_' + str(i)
            trainX[i,0] = model.docvecs[prefix_train_pos]
            trainX[3000 + i,0] = model.docvecs[prefix_train_neg]
            train_labels[i] = 1
            train_labels[3000 + i] = 0

        trainY = to_categorical(train_labels, nb_classes=2)
        print "train_labels",train_labels

        testX = numpy.zeros((number_of_samples, 1,100))
        test_labels = numpy.zeros(number_of_samples)

        for i in range(3000):
            prefix_test_pos = 'TEST_POS_' + str(i)
            prefix_test_neg = 'TEST_NEG_' + str(i)
            testX[i,0] = model.docvecs[prefix_test_pos]
            testX[3000 + i,0] = model.docvecs[prefix_test_neg]
            test_labels[i] = 1
            test_labels[3000 + i] = 0

        testY = to_categorical(test_labels, nb_classes=2)
        # Network building
        net = tflearn.input_data([None, 1,100])
        # net = tflearn.lstm(net, 128, dropout=cl.dropout)
        net = tflearn.lstm(net, cl.number_of_cells, dropout=cl.dropout)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.001, loss=cl.loss)

        # # Training
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=20)
