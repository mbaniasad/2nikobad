
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
    
    def __init__(self,dataset_path,optimizer,loss):
        self.dataset_path = dataset_path
        self.loss= loss
        self.optimizer=optimizer
        
    def setting_name(self):
        return 'doc2vec'+'ds-'+self.dataset_path +'optimizer-'+self.optimizer+'loss-'+str(self.loss)
datasetName="aclImdb"
d2v_dataset_file_name='./4-PKLED/'+datasetName+'.d2v'
doc2vec_configs = [
                   config( datasetName, 'RMSProp', 'categorical_crossentropy'),
                   # config( datasetName, 'Momentum', 'categorical_crossentropy'),
                   # config( datasetName, 'AdaGrad', 'categorical_crossentropy'),
                   # config( datasetName, 'Ftrl', 'categorical_crossentropy'),
                   # config( datasetName, 'AdaDelta', 'categorical_crossentropy'),
                   # config( datasetName, 'sgd', 'categorical_crossentropy'),
                   # config( datasetName, 'adam', 'categorical_crossentropy')
                   ]




for cl in doc2vec_configs:
    with tf.Graph().as_default():
        print("running doc2vecRegression_on_"+cl.setting_name())
        # loading dov2vec model
        model = Doc2Vec.load(d2v_dataset_file_name)

        print model.docvecs[0]
        number_of_samples = 6000; #number of samples in the datset. 
        half=number_of_samples/2

        trainX = numpy.zeros((number_of_samples, 100))
        train_labels = numpy.zeros(number_of_samples)

        for i in range(half):
            prefix_train_pos = 'TRAIN_POS_' + str(i)
            prefix_train_neg = 'TRAIN_NEG_' + str(i)
            trainX[i] = model.docvecs[prefix_train_pos]
            trainX[half + i] = model.docvecs[prefix_train_neg]
            train_labels[i] = 1
            train_labels[half + i] = 0

        trainY = to_categorical(train_labels, nb_classes=2)
        print "train_labels",train_labels

        testX = numpy.zeros((number_of_samples, 100))
        test_labels = numpy.zeros(number_of_samples)

        for i in range(half):
            prefix_test_pos = 'TEST_POS_' + str(i)
            prefix_test_neg = 'TEST_NEG_' + str(i)
            testX[i] = model.docvecs[prefix_test_pos]
            testX[half + i] = model.docvecs[prefix_test_neg]
            test_labels[i] = 1
            test_labels[half + i] = 0

        testY = to_categorical(test_labels, nb_classes=2)
        # Network building
        net = tflearn.input_data([None, 100])
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.001, loss=cl.loss)

        # Training
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=20)
        model.save("./5-SAVED_MODELS/"+cl.setting_name())