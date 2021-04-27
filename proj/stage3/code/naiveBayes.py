from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import time

def naiveBayesClassifierTest(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    print('-----------------------------')    
    print('Naive Bayes Classifier Test was Called. Wait...')
    # capture the start time
    start = time.time()
    y_pred = gnb.fit(X_train, np.ravel(y_train)).predict(X_test)
    # capture the end time of calculation
    end = time.time()

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (np.ravel(y_test)!= y_pred).sum()))
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)