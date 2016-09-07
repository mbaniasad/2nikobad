"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    datasetName = "aclImdb"
    movie_reviews_data_folder = "./txt_sentoken/"
    movie_reviews_data_folder = "../2-DS/"+datasetName+"/train"
    movie_reviews_data_test_folder = "../2-DS/"+datasetName+"/test"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    testDataset = load_files(movie_reviews_data_test_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    
    # split the dataset in training and test set:
    docs_train, docs_valid, y_train, y_valid = train_test_split(
        dataset.data, dataset.target, test_size=0.025, random_state=None)

    docs_test, _, y_test, _ = train_test_split(
        testDataset.data, dataset.target, test_size=0, random_state=None)


    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
    ])
    text_clf = text_clf.fit(docs_train, y_train)
    y_predicted = text_clf.predict(docs_valid)
    print "validation accuracy",np.mean(y_predicted == y_valid) 
    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    # Print the classification report
    print "validation restuls",(metrics.classification_report(y_valid, y_predicted,
                                         target_names=dataset.target_names))

    ###testing
    print "test"
    
    y_predicted_test = text_clf.predict(docs_test)
    print "test accuracy",np.mean(y_predicted_test == y_test)     
    print metrics.accuracy_score(y_predicted_test ,y_test)

    print "test restuls",(metrics.classification_report(y_test, y_predicted_test,
                                         target_names=dataset.target_names))  
    # plt.matshow(cm)
    # plt.show()
