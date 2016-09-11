from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import numpy as np

datasetName = "mostest"
movie_reviews_data_folder = "../2-DS/"+datasetName+"/train"
movie_reviews_data_test_folder = "../2-DS/"+datasetName+"/test"
dataset = load_files(movie_reviews_data_folder, shuffle=False)
testDataset = load_files(movie_reviews_data_test_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))
print "train data count:",len(dataset.data), "targets count:",len(dataset.target)

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



train_x,test_x = createWordFrequencyVectors(dataset.data,testDataset.data)
train_y = dataset.target
test_y = testDataset.target
# # gnb = GaussianNB()#gausian bayes needs an array as
mnb = MultinomialNB()
classifer = mnb.fit(train_x,train_y)
y_pred = classifer.predict(train_x)
print np.mean(train_y == y_pred)



y_pred_test = classifer.predict(test_x)

print np.mean(test_y == y_pred_test)