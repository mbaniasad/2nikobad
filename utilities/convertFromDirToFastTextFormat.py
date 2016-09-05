import os
import glob
import string

def readDatasFromDir(DirAddress):
    exclude = set(string.punctuation)
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            s = f.readline().strip()
            s = ''.join(ch for ch in s if ch not in exclude)
            sentences.append(s)
    os.chdir(currdir)
    return sentences
datasetName="EAR-cls-acl10"
datasetAddress = "../2-DS/"+datasetName+"/"
destinationDir = "../2-DS/"+datasetName+"/"
trainPositive = [datasetAddress + "train/pos/", destinationDir+"fastTextFomatTrain.txt", "__POS__"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"fastTextFomatTrain.txt","__NEG__"]
# trainUnsuperviesd = [datasetAddress + "train/unsup/", destinationDir+"train-unsup.txt"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"fastTextFomatTest.txt","__POS__"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"fastTextFomatTest.txt", "__NEG__"]

addresses =[ trainPositive, trainNegitive , testPositive, testNegitive]
for directoryAddress,filename,label in addresses:
    with open (filename,'a') as destinationFile:
        print(filename)
        counter =0
        for line in readDatasFromDir(directoryAddress):
            destinationFile.write(line+" "+label+"\n")
            counter+=1
        print(counter)    
