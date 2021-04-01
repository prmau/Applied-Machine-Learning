#SVM-classification
#SVM-NonLinear classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

def svmLinearTest(X_train, X_test, y_train, y_test, classes):
    print('-----------------------------')
    print('SVM Linear Test was Called. Wait...')
    path = Path(__file__).parent.absolute()

    c= [0.01, 0.1, 1.0, 5, 10, 15,20]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []

    #fit the training dataset to linear kernel model
    #capture the start time
    for i in c:
        start = time.time()
        svc = SVC(kernel = 'linear', gamma=0.7, C=i, random_state=1)
        svc.fit(X_train, y_train.values.ravel())

        y_pred_linear = svc.predict(X_test)
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred_linear))

    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for SVM-Linear is generated in the output folder")

    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/svm-linear/").mkdir(parents=True, exist_ok=True)
    plt.clf()


    #Generating Test accuracy plot


    plt.plot(c, testAccuracy, 'ro-')
    plt.legend(['Test Accuracy'])
    plt.xlabel('C Param value')
    plt.ylabel('Accuracy')
    plt.title("Test Accuracy")
    strFile = str(path)+"/output/svm-linear"+"/Test Accuracy.png"

    print(os.path.isfile(strFile))
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Genrerating the running time plot
    plt.plot(c, runningTime, 'ro-')
    plt.legend(['Running time(s)'])
    plt.xlabel('C Param value')
    plt.ylabel('Running time(seconds)')
    plt.title("Running time")
    strFile = str(path)+"/output/svm-linear"+"/Running Time.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()


    #Finding the max accuracy

    maxValue = max(testAccuracy)
    max_index = testAccuracy.index(maxValue)

    print("The maximum test accuracy - %.3f and the corresponding C param value "%(testAccuracy[max_index]))
    print(c[max_index])

    #Printing the classification report
    vizualizer = ClassificationReport(SVC(kernel='linear', gamma = 0.7, C= c[max_index], random_state=1), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/svm-linear"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return
