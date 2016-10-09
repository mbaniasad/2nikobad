import cPickle as pickle
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
import tensorflow as tf

with open("./4-PKLED/FastTextModel-GAR-cls-acl10.pkl",'rb') as fp:
    train,train_labels,test,test_labels=pickle.load(fp)

import numpy

import tflearn
from tflearn.data_utils import to_categorical

class config:
    
    def __init__(self,dataset_path,optimizer,loss):
        self.dataset_path = dataset_path
        self.loss= loss
        self.optimizer=optimizer
        
    def setting_name(self):
        	return 'fastText'+'ds-'+self.dataset_path +'optimizer-'+self.optimizer+'loss-'+str(self.loss)+"n_epoch200"
datasetName="FastTextModel-GAR-cls-acl10"
fastText_configs = [
                   # config( datasetName, 'RMSProp', 'categorical_crossentropy'),
                   # config( datasetName, 'Momentum', 'categorical_crossentropy'),
                   # config( datasetName, 'AdaGrad', 'categorical_crossentropy'),
                   # config( datasetName, 'Ftrl', 'categorical_crossentropy'),
                   # config( datasetName, 'AdaDelta', 'categorical_crossentropy'),
                   # config( datasetName, 'sgd', 'categorical_crossentropy'),
                   config( datasetName, 'adam', 'categorical_crossentropy')
                   ]




for cl in fastText_configs:
    with tf.Graph().as_default():
        print("running fastTextRegression_on_"+cl.setting_name())
        # loading dov2vec model
        

        trainX = train
        trainY = to_categorical(train_labels, nb_classes=2)
        print "train_labels",train_labels

        testX = test
        testY = to_categorical(test_labels, nb_classes=2)
        # Network building
        net = tflearn.input_data([None, 100])
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.01, loss=cl.loss)

        # Training
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=200)
        model.save("./5-SAVED_MODELS/"+cl.setting_name())