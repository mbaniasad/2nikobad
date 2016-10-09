# -*- coding: utf-8 -*-
import os
import glob
import string
import numpy as np


def GetVectorForSentence(sentence):
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
datasetName="GAR-cls-acl10"
datasetAddress = "../2-DS/"+datasetName+"/"
destinationDir = "../3-PREPROCESSED/LOWER-NOPUNC/"+datasetName+"/"
trainPositive = [datasetAddress + "train/pos/", destinationDir+"fastTextFomatTrain.csv", "1"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"fastTextFomatTrain.csv","0"]
# trainUnsuperviesd = [datasetAddress + "train/unsup/", destinationDir+"train-unsup.csv"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"fastTextFomatTest.csv","1"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"fastTextFomatTest.csv", "0"]

addresses =[ trainPositive, trainNegitive , testPositive, testNegitive]
exclude = set(string.punctuation)

for directoryAddress,filename,label in addresses:
    with open (filename,'a') as destinationFile:
        print(filename)
        counter =0
        for line in readDatasFromDir(directoryAddress):
            line = line.lower()
            if line == "":
                line = "empty Comment"    
            destinationFile.write(label+","+line+"\n")
            counter+=1
        print(counter)    
