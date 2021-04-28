# Adaboost

from sklearn.metrics import accuracy_score
import time
from pathlib import Path
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time


def adaboostTest(X_train, X_test, y_train, y_test):
    print('-----------------------------')    
    print('Adaboost Test was Called. Wait...')
    
    
    
    path = Path(__file__).parent.absolute()
    depths= [2, 5, 10, 20]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []
    
    #fit the training dataset to linear kernel model
    #capture the start time
    for i in depths:
        start = time.time()
    
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i))
        clf.fit(X_train, y_train.values.ravel())
        
        y_pred = clf.predict(X_test)
        
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred))
    
    #print(trainAccuracy)
    #print(testAccuracy)
    
    
    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for Bagging is generated in the output folder")
    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/adaboost/").mkdir(parents=True, exist_ok=True)
    plt.clf()


    #Generating Test accuracy plot


    plt.plot(depths, testAccuracy, 'ro-')
    plt.legend(['Test Accuracy'])
    plt.xlabel('max_depth values')
    plt.ylabel('Accuracy')
    plt.title("Test Accuracy")
    strFile = str(path)+"/output/adaboost"+"/Test Accuracy.png"

    print(os.path.isfile(strFile))
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Genrerating the running time plot
    plt.plot(depths, runningTime, 'ro-')
    plt.legend(['Running time(s)'])
    plt.xlabel('max_depth values')
    plt.ylabel('Running time(seconds)')
    plt.title("Running time")
    strFile = str(path)+"/output/adaboost"+"/Running Time.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()


    #Finding the max accuracy

    maxValue = max(testAccuracy)
    max_index = testAccuracy.index(maxValue)

    print("The maximum test accuracy - %.3f and the corresponding max_depth value "%(testAccuracy[max_index]))
    print(depths[max_index])

    #Printing the classification report
    vizualizer = ClassificationReport(DecisionTreeClassifier(criterion='gini', max_depth= depths[max_index]), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/adaboost"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return