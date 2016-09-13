# -*- coding: utf-8 -*-

import os
import glob
import string
from motextpreprocessor import MoTextPreprocessor
import codecs
preprocessingConfig = dict(language = "de", stemming=False,removeUmlaut = True, lowerCase = True,removePunc=True)
def readDatasFromDir(DirAddress,number_of_samples):
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    sample_counter=0
    for ff in glob.glob("*.txt"):
        sample_counter +=1
        if sample_counter > number_of_samples:
            break
        with codecs.open(ff,'r',encoding='utf8') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    return sentences
datasetName="matelsoCalls-noUmlaut-nopunct"
number_of_samples = 3000
datasetAddress = "../3-PREPROCESSED/lower-nopunc-noumlaut/"+datasetName+"/"
destinationDir = "../3-PREPROCESSED/lower-nopunc-noumlaut/"+datasetName+"/"
if not os.path.exists(destinationDir):
    os.makedirs(destinationDir)


trainPositive = [datasetAddress + "train/pos/", destinationDir+"train-pos.txt"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"train-neg.txt"]
# trainUnsuperviesd = [datasetAddress + "train/unsup/", destinationDir+"train-unsup.txt"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"test-pos.txt"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"test-neg.txt"]

addresses =[ trainPositive, trainNegitive, testPositive, testNegitive]
exclude = set(string.punctuation)

for directoryAddress,filename in addresses:
    with open(filename,'w') as destinationFile:
        print(filename)
        for line in readDatasFromDir(directoryAddress,number_of_samples):
            line =  MoTextPreprocessor.normalize(line,**preprocessingConfig).encode("utf-8")
            destinationFile.write(line+"\n")
