import os
import glob

def readDatasFromDir(DirAddress):
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    return sentences
datasetName="aclImdb"
datasetAddress = "../1-RAW-DS/"+datasetName+"/"
destinationDir = "../2-DS/"+datasetName+"/"
trainPositive = [datasetAddress + "train/pos/", destinationDir+"train-pos.txt"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"train-neg.txt"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"test-pos.txt"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"test-neg.txt"]

addresses =[ trainPositive, trainNegitive, testPositive, testNegitive]
for directoryAddress,filename in addresses:
    with open (filename,'w') as destinationFile:
        for line in readDatasFromDir(directoryAddress):
            destinationFile.write(line)
