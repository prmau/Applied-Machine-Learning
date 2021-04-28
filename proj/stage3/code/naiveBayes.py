from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport
from sklearn import metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import time

def naiveBayesClassifierTest(X_train, X_test, y_train, y_test):
    path = Path(__file__).parent.absolute()
      #Creates a new directory under svm-linear if it doesn't exist
    Path("output/GaussianNB/").mkdir(parents=True, exist_ok=True)

    gnb = GaussianNB()
    print('-----------------------------')    
    print('Naive Bayes Classifier Test was Called. Wait...')
    # capture the start time
    start = time.time()
    y_pred = gnb.fit(X_train, np.ravel(y_train)).predict(X_test)
    # capture the end time of calculation
    end = time.time()

    print("Time taken to train model and prediction :", end-start)

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (np.ravel(y_test)!= y_pred).sum()))
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for SVM-Linear is generated in the output folder")
    #Printing the classification report
    vizualizer = ClassificationReport(gnb, classes = [0,1,2,3,4,5])
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/GaussianNB"+"/Classification Report.png"
    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()