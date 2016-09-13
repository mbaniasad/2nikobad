
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
import random
import tflearn
from tflearn.data_utils import to_categorical

class config:
    
    def __init__(self,dataset_path,optimizer,loss):
        self.dataset_path = dataset_path
        self.loss= loss
        self.optimizer=optimizer
        
    def setting_name(self):
        return 'doc2vec'+'dsEQ-'+self.dataset_path +'optimizer-'+self.optimizer+'loss-'+str(self.loss)
datasetName="matelsoCalls-noUmlaut-nopunct"
d2v_dataset_file_name='./4-PKLED/'+datasetName+'.d2v'
doc2vec_configs = [
                    config( datasetName, 'RMSProp', 'categorical_crossentropy'),
                    config( datasetName, 'Momentum', 'categorical_crossentropy'),
                    config( datasetName, 'AdaGrad', 'categorical_crossentropy'),
                    config( datasetName, 'Ftrl', 'categorical_crossentropy'),
                    # config( datasetName, 'AdaDelta', 'categorical_crossentropy'),
                    config( datasetName, 'sgd', 'categorical_crossentropy'),
                    config( datasetName, 'adam', 'categorical_crossentropy')
                   ]




for cl in doc2vec_configs:
    with tf.Graph().as_default():
        print("running doc2vecRegression_on_"+cl.setting_name())
        # loading dov2vec model
        model = Doc2Vec.load(d2v_dataset_file_name)

        
        # number_of_samples = 6000;
        pos_sample_count = 135
        neg_sample_count = 462
        number_of_samples = pos_sample_count*2
        # half=number_of_samples/2

        trainX = numpy.zeros((number_of_samples, 100))
        train_labels = numpy.zeros(number_of_samples)

        for i in range(pos_sample_count):
            prefix_train_pos = 'TRAIN_POS_' + str(i)
            
            trainX[i] = model.docvecs[prefix_train_pos]
            
            train_labels[i] = 1
            
        for i in range(pos_sample_count):
            rnd = random.randint(pos_sample_count,neg_sample_count-1)
            prefix_train_neg = 'TRAIN_NEG_' + str(rnd)
            trainX[pos_sample_count+i] = model.docvecs[prefix_train_neg]
            train_labels[pos_sample_count+i] = 0

        trainY = to_categorical(train_labels, nb_classes=2)
        print "train_labels",train_labels

        crossValidationPercent = 0.15
        validationPivot = int(trainX.shape[0]*crossValidationPercent) 


        sub_train_x = trainX[:-validationPivot]  
        sub_train_y = trainY[:-validationPivot]  
        sub_test_x = trainX[-validationPivot+1:]
        sub_test_y = trainY[-validationPivot+1:] 
        
        
        # Network building
        net = tflearn.input_data([None, 100])
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.001, loss=cl.loss)

        # Training
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
        model.fit(sub_train_x, sub_train_y, validation_set=(sub_test_x, sub_test_y), show_metric=True, batch_size=32, n_epoch=20)
        model.save("./5-SAVED_MODELS/"+cl.setting_name())