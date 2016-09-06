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

if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = "./txt_sentoken/"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    print len(docs_train)

    count_vect = CountVectorizer()

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    X_train_counts = count_vect.fit_transform(docs_train)
    print "shape of vectorized train data =>", X_train_counts.shape
    print "each sentence is now a vector of words and their frequency"
    mo_index =count_vect.vocabulary_.get(u'mohammad')
    print "id of word mohammad ",mo_index



    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape
    # that are too rare or too frequent

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    clf = MultinomialNB().fit(X_train_tfidf, y_train)



    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
    ])
    text_clf = text_clf.fit(docs_train, y_train)
    predicted = text_clf.predict(docs_test)
    print np.mean(predicted == y_test) 
    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    # Print the classification report
    # print(metrics.classification_report(y_test, y_predicted,
    #                                     target_names=dataset.target_names))

    # Print and plot the confusion matrix
    # cm = metrics.confusion_matrix(y_test, y_predicted)
    # print(cm)

    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()
