import itertools
import lstm
import tflearn as tfl
import os
import sys
from  LSTMconfig import LSTMconfig



globaldataset = "GAR-cls-acl10"

classifiers= ["LSTM","DLSTM"]
datasets = ["GAR-cls-acl10","aclImdb"]
number_of_words_used_in_embeddings = [1000,10000]
dropouts =[0.5]
n_epochs =[1]
losss =['categorical_crossentropy']
optimizers =['AdaDelta','AdaGrad','adam','Ftrl','Momentum','sgd']

lstm_options =(list(tup) for tup in  itertools.product(*[classifiers,datasets,number_of_words_used_in_embeddings,dropouts,
                            n_epochs,
                            losss,
                            optimizers]))
lstm_configs = []
dlstm_configs= []
for cl in lstm_options:
    co= LSTMconfig(cl[0], cl[1], cl[2], cl[3], cl[4], cl[5],cl[6])
    if os.path.exists("./5-SAVED_MODELS/"+co.setting_name()):
        print " model found in models dir " +co.setting_name()
    else:
     print "adding "+co.setting_name()
     if co.classifier=="LSTM":
        lstm_configs.append(co)
     elif co.classifier=="DLSTM":
        dlstm_configs.append([cl[0], cl[1], cl[2], cl[3], cl[4], cl[5],cl[6]])


lstm.runLstm(lstm_configs)

for cl in dlstm_configs:
    try:
        sys.argv = [cl[0], cl[1], cl[2], cl[3], cl[4], cl[5],cl[6]]
        execfile('dynamic_lstm.py')
    except Exception, e:
        print("Unexpected error:", sys.exc_info()[0])
    
