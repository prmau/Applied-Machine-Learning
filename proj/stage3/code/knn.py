from yellowbrick.classifier import ClassificationReport
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import time


def knnTest(X_train, X_test, y_train, y_test):
    path = Path(__file__).parent.absolute()
      #Creates a new directory under svm-linear if it doesn't exist
    Path("output/knn/").mkdir(parents=True, exist_ok=True)

    print('-----------------------------')    
    print('KNN Classifier Test was Called. Wait...')

    c= [3,4,5,6,8]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []

    for i in c:
        # capture the start time
        start = time.time()

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, np.ravel(y_train))
        y_pred_knn = neigh.predict(X_test)
        # capture the end time of calculation
        end = time.time()
        #Storing the metrics
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred_knn))

    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for KNN is generated in the output folder")

    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/knn/").mkdir(parents=True, exist_ok=True)
    plt.clf()

    #Generating Test accuracy plot
    plt.plot(c, testAccuracy, 'ro-')
    plt.legend(['Test Accuracy'])
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title("Test Accuracy")
    strFile = str(path)+"/output/knn"+"/Test Accuracy.png"

    print(os.path.isfile(strFile))
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Genrerating the running time plot
    plt.plot(c, runningTime, 'ro-')
    plt.legend(['Running time(s)'])
    plt.xlabel('n_neighbors')
    plt.ylabel('Running time(seconds)')
    plt.title("Running time")
    strFile = str(path)+"/output/knn"+"/Running Time.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()


    #Finding the max accuracy

    maxValue = max(testAccuracy)
    max_index = testAccuracy.index(maxValue)

    print("The maximum test accuracy - %.3f and the corresponding n_neighbors value "%(testAccuracy[max_index]))
    print(c[max_index])

    #Printing the classification report
    vizualizer = ClassificationReport(KNeighborsClassifier(n_neighbors=3), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/knn"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return