#this would also apply 
#lower casing
#removeing punctuation

import os
import glob
import string
def readDatasFromDir(DirAddress,number_of_samples):
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    sample_counter=0
    for ff in glob.glob("*.txt"):
        sample_counter +=1
        if sample_counter > number_of_samples:
            break
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    return sentences
datasetName="EAR-cls-acl10"
number_of_samples = 3000
datasetAddress = "../2-DS/"+datasetName+"/"
destinationDir = "../3-PREPROCESSED/LOWER-NOPUNC/"+datasetName+"-"+str(number_of_samples)+"Sample/"
if not os.path.exists(destinationDir):
    os.makedirs(destinationDir)


trainPositive = [datasetAddress + "train/pos/", destinationDir+"train-pos.txt"]
trainNegitive = [datasetAddress + "train/neg/", destinationDir+"train-neg.txt"]
trainUnsuperviesd = [datasetAddress + "train/unsup/", destinationDir+"train-unsup.txt"]
testPositive = [datasetAddress + "test/pos/" ,destinationDir+"test-pos.txt"]
testNegitive =[ datasetAddress + "test/neg/",destinationDir+"test-neg.txt"]

addresses =[ trainPositive, trainNegitive,trainUnsuperviesd, testPositive, testNegitive]
exclude = set(string.punctuation)

for directoryAddress,filename in addresses:
    with open (filename,'w') as destinationFile:
        print(filename)
        for line in readDatasFromDir(directoryAddress,number_of_samples):
            line = line.lower()
            line = ''.join(ch for ch in line if ch not in exclude)
            destinationFile.write(line+"\n")
