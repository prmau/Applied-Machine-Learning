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
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean

def svmLinearTest(X_train, X_test, y_train, y_test, classes, X_1_df, Y_1_df):
    print('-----------------------------')
    print('SVM Linear Test was Called. Wait...')
    path = Path(__file__).parent.absolute()

    c= [0.01, 0.1,0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 15,20]
    #c = [0.01, 0.1, 0.2]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []
    param = []


    #fit the training dataset to linear kernel model
    #capture the start time
    for i in c:
        start = time.time()
        svc = SVC(kernel = 'linear', gamma=0.7, C=i, random_state=1)
        svc.fit(X_train, y_train.values.ravel())

        y_pred_linear = svc.predict(X_test)
        y_train_pred_linear = svc.predict(X_train)
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        param.append(i)
        trainAccuracy.append(accuracy_score(y_train, y_train_pred_linear))
        testAccuracy.append(accuracy_score(y_test, y_pred_linear))

    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for SVM-Linear is generated in the output folder")

    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/svm-linear/").mkdir(parents=True, exist_ok=True)
    plt.clf()


    #Generating Test accuracy plot


    plt.plot(c, trainAccuracy, 'ro-', c, testAccuracy, 'bv--')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('C Param value')
    plt.ylabel('Accuracy')
    plt.title("SVMLinear-Accuracy")
    strFile = str(path)+"/output/svm-linear"+"/Accuracy.png"

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
    optimum_param = param[max_index]

    #Printing the average running time
    print("The average running time - %.3f seconds" %mean(runningTime))

    print("The maximum test accuracy  - %.3f "%maxValue)
    print("Corresponding C value for max test accuracy", optimum_param)

    pca = PCA(n_components=2)
    X_transform = pca.fit_transform(X_1_df)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(pd.DataFrame(X_transform), Y_1_df, random_state=1,
                                                            test_size=0.2)

    svc = SVC(kernel='linear', gamma=0.7, C=optimum_param, random_state=1).fit(X_train1, np.ravel(y_train1))

    # Generating decision boundary chart

    y = pd.DataFrame(Y_1_df).to_numpy()
    y = y.astype(np.int).flatten()

    plot_decision_regions(X_transform, y, clf=svc, legend=2)

    # Adding axes annotations
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    strFile = str(path) + "/output/svm-linear" + "/Decision Boundary.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Printing the classification report
    vizualizer = ClassificationReport(SVC(kernel='linear', gamma = 0.7, C= optimum_param, random_state=1), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/svm-linear"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return
