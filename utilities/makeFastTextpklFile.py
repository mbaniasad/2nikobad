# -*- coding: utf-8 -*-
import os
import glob
import string
import numpy as np
import fasttext
import cPickle as pickle


def GetVectorForSentence(sentence,model):
    words = sentence.split()
    npavg = np.zeros((len(words),100))
    i =0
    for i in range(len(words)):
        npavg[i]=model[words[i]]
        i+=1

    return np.average(npavg, axis=0)    


def readDatasFromDir(DirAddress):
    exclude = set(string.punctuation)
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            s = f.readline().strip()
            s = ''.join(ch for ch in s if ch not in exclude)
            s = s.replace('Ä','ae')
            s = s.replace('ä','ae')
            s = s.replace('Ü','ue')
            s = s.replace('ü','ue')
            s = s.replace('Ö','oe')
            s = s.replace('ö','oe')
            s = s.replace('\n','')
            sentences.append(s)
    os.chdir(currdir)
    return sentences
datasetName="aclImdb"
datasetAddress = "../2-DS/"+datasetName+"/"
destinationDir = "../3-PREPROCESSED/LOWER-NOPUNC/"+datasetName+"/"
trainPositive = [datasetAddress + "train/pos/", destinationDir+"fastTextFomatTrain.csv", "1"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"fastTextFomatTrain.csv","0"]
# trainUnsuperviesd = [datasetAddress + "train/unsup/", destinationDir+"train-unsup.csv"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"fastTextFomatTest.csv","1"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"fastTextFomatTest.csv", "0"]


exclude = set(string.punctuation)

addresses =[ trainPositive, trainNegitive ]
train = []
train_label = []
fastTextmodelAddress = "aclImdb-fasttextModel-skipGram-ws5-epoch10-word_ngrams3.bin" # file address for fasttext model bin file
fastTextmodel = fasttext.load_model(fastTextmodelAddress)
for directoryAddress,filename,label in addresses:
    with open (filename,'a') as destinationFile:
        print(filename)
        counter =0
        for line in readDatasFromDir(directoryAddress):
            line = line.lower()
            if line == "":
                line = "empty Comment"    
            train.append(GetVectorForSentence(line,fastTextmodel))
            train_label.append(label)
            counter+=1

addresses =[ testPositive, testNegitive ]
test = []
test_label = []

fastTextmodel = fasttext.load_model(fastTextmodelAddress)
for directoryAddress,filename,label in addresses:
    with open (filename,'a') as destinationFile:
        print(filename)
        counter =0
        for line in readDatasFromDir(directoryAddress):
            line = line.lower()
            if line == "":
                line = "empty Comment"    
            test.append(GetVectorForSentence(line,fastTextmodel))
            test_label.append(label)
            counter+=1            

with open("../4-PKLED/FastTextModel-GAR-cls-acl10.pkl","wb") as fp:
    pickle.dump([train,train_label,test,test_label],fp)

