import itertools
import lstm
import os
import sys
from  LSTMconfig import LSTMconfig
from subprocess import call

#------------------------------- configuration variations
globaldataset = "EAR-cls-acl10"

classifiers= ["DLSTM", "LSTM"]  # choose between any in  ["DLSTM", "LSTM"]
datasets = [globaldataset]      # the name of the dataset or datasets that would be used
number_of_words_used_in_embeddings = [10] #number less than max unique words in the dataset
dropouts =[0.9]                 #dropout number between 0 to 1
n_epochs =[1]                   # number of epochs 
losss =['categorical_crossentropy'] #loss function 
optimizers =['adam']            # one of possible optimizers from {SGD, RMSProp, Adam, Momentum, AdaGrad, Ftrl, AdaDelta}
#------------------------------- configuration variations

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
        # execfile('dynamic_lstm.py')
        call("python dynamic_lstm.py " + ' ' + str(cl[0])+ ' ' +  str(cl[1])+ ' ' +  str(cl[2])+ ' ' +  str(cl[3])+ ' ' +  str(cl[4])+ ' ' +  str(cl[5])+ ' ' + str(cl[6]), shell=True)
    except Exception, e:
        print e
    
