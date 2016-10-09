from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn.utils import shuffle
import random
import scipy
# "/home/moo1366/Documents/uni/GuidedResearch/2nikobad/3-PREPROCESSED/lower-nopunc-noumlaut/aclImdb"
# "/home/moo1366/Documents/uni/GuidedResearch/2nikobad/3-PREPROCESSED/lower-nopunc-noumlaut-snowball-stemmed/aclImdb/test/neg"
datasetName = "aclImdb"
train_dir = "../3-PREPROCESSED/lower-nopunc-noumlaut-snowball-stemmed/"+datasetName+"/train"
test_dir = "../3-PREPROCESSED/lower-nopunc-noumlaut-snowball-stemmed/"+datasetName+"/test"
dataset = load_files(train_dir, shuffle=False)
testDataset = load_files(test_dir, shuffle=False)

print "train data count:at ", train_dir ,len(dataset.data), "targets count:",len(dataset.target)
print "test data count:at ", test_dir ,len(dataset.data), "targets count:",len(dataset.target)

def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def createWordFrequencyVectors(trainSentences, testSentences):
	print "createing wordCountVectors"
	cv = CountVectorizer()
	wordCountVectors = cv.fit_transform(trainSentences)
	testwordCountVectors = cv.transform(testSentences)
	#posNo=cv.vocabulary_.get(u'pos')
	# print "the id of word pos is ",posNo
	# print "count of word pos in each doc", wordCountVectors.toarray()[0][posNo],wordCountVectors.toarray()[1][posNo]

	#wordFrequencyVectors
	print "creating wordFrequencyVectors, count/numberofWords in doc"
	tfidf_transformer = TfidfTransformer()
	wordFrequencyVectors = tfidf_transformer.fit_transform(wordCountVectors)
	testwordFrequencyVectors = tfidf_transformer.transform(testwordCountVectors)

	# print "count/wordCountOfDoc pos in each doc",wordFrequencyVectors.toarray()[0][posNo],wordFrequencyVectors.toarray()[1][posNo]
	return wordFrequencyVectors,testwordFrequencyVectors



# training on 

train_x,test_x = createWordFrequencyVectors(dataset.data,testDataset.data)
print train_x.shape[0]
train_y = dataset.target
test_y= testDataset.target

#correcting destribution

numer_of_shuffles=1
finallres = []
for i in range(numer_of_shuffles): 
	res = []
	train_x,train_y = shuffle(train_x,train_y )

	numberOfPos = np.sum(train_y==1)
	while len(train_y)!=(numberOfPos*2):
		i = random.randint(0,len(train_y)-1)
		if train_y[i]==0:
			train_y= np.delete(train_y,i)
			delete_row_csr(train_x,i)
	

	crossValidationPercent = 0.15
	validationPivot = int(train_x.shape[0]*crossValidationPercent) 


	sub_train_x = train_x#[:-validationPivot]  
	sub_train_y = train_y#[:-validationPivot]  
	sub_test_x = test_x#[-validationPivot+1:]
	sub_test_y = test_y#[-validationPivot+1:]  

	# # gnb = GaussianNB()#gausian bayes needs an array as
	mnb = MultinomialNB()
	classifer = mnb.fit(sub_train_x,sub_train_y)
	y_pred = classifer.predict(sub_train_x)
	res.append(np.mean(sub_train_y == y_pred))

	y_pred_test = classifer.predict(sub_test_x)

	res.append(np.mean(sub_test_y == y_pred_test))
	finallres.append(res)
print numer_of_shuffles, np.mean(finallres,axis=0)