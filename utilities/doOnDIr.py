# -*- coding: utf-8 -*-
import os
import glob
import string
import string
import codecs
from motextpreprocessor import MoTextPreprocessor
preprocessingConfig = dict(language = "en", stemming=True,removeUmlaut = True, lowerCase = True,removePunc=True)

datasetName="aclImdb"
datasetAddress = "/home/moo1366/Documents/uni/GuidedResearch/2nikobad/3-PREPROCESSED/lower-nopunc-noumlaut-snowball-stemmed/aclImdb/"
trainPositive = datasetAddress + "train/pos/"
trainNegitive = datasetAddress + "train/neg/"
# trainUnsuperviesd = datasetAddress + "train/unsup/"
testPositive = datasetAddress + "test/pos/"
testNegitive =datasetAddress + "test/neg/"

addresses =[ trainPositive, trainNegitive , 
             testPositive, testNegitive
           ]
exclude = set(string.punctuation)

currdir = os.getcwd()
    
for address in addresses:
    os.chdir(address)
    for filename in glob.glob("*.txt"):
        print filename
        doc = ""
        with codecs.open (filename,'r',encoding='utf8') as sourceFile:
            doc = sourceFile.read().replace('\n', '')    
        #do things on doc
        doc =  MoTextPreprocessor.normalize(doc,**preprocessingConfig)
        #write the changes into the same file
        
        with open (filename,'w') as destinationFile:
            destinationFile.write(doc.encode("utf-8"))
    os.chdir(currdir)