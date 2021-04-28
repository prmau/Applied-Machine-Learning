from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
from pathlib import Path
import os
import time


def logisticRegressionTest(X_train, X_test, y_train, y_test):
    path = Path(__file__).parent.absolute()
      #Creates a new directory under svm-linear if it doesn't exist
    Path("output/logistic-regression/").mkdir(parents=True, exist_ok=True)

    print('-----------------------------')    
    print('Logistic Regression Test was Called. Wait...')

    c= ["liblinear", "sag", "lbfgs"]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []

    for i in c:
        # capture the start time
        start = time.time()
        clf = LogisticRegression(random_state=0, solver=i,class_weight='balanced', max_iter=10000).fit(X_train, np.ravel(y_train))
        y_pred_lr = clf.predict(X_test)
        # capture the end time of calculation
        end = time.time()
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred_lr))


    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for LR is generated in the output folder")

    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/knn/").mkdir(parents=True, exist_ok=True)
    plt.clf()

    #Generating Test accuracy plot
    plt.plot(c, testAccuracy, 'ro-')
    plt.legend(['Test Accuracy'])
    plt.xlabel('Solver function')
    plt.ylabel('Accuracy')
    plt.title("Test Accuracy")
    strFile = str(path)+"/output/logistic-regression"+"/Test Accuracy.png"

    print(os.path.isfile(strFile))
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Genrerating the running time plot
    plt.plot(c, runningTime, 'ro-')
    plt.legend(['Running time(s)'])
    plt.xlabel('Solver function')
    plt.ylabel('Running time(seconds)')
    plt.title("Running time")
    strFile = str(path)+"/output/logistic-regression"+"/Running Time.png"

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
    vizualizer = ClassificationReport(LogisticRegression(random_state=0, solver=c[max_index],class_weight='balanced', max_iter=10000), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/logistic-regression"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return