from yellowbrick.classifier import ClassificationReport
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean
import numpy as np


def knnTest(X_train, X_test, y_train, y_test, X_1_df, Y_1_df):
    path = Path(__file__).parent.absolute()
      #Creates a new directory under svm-linear if it doesn't exist
    Path("output/knn/").mkdir(parents=True, exist_ok=True)

    print('-----------------------------')    
    print('KNN Classifier Test was Called. Wait...')

    c= list(range(1, 51))
    runningTime = []
    trainAccuracy = []
    testAccuracy = []
    cv_scores = []
    param = []

    for i in c:
        # capture the start time
        start = time.time()

        neigh = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(neigh, X_1_df, Y_1_df.values.ravel(), cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        neigh.fit(X_train, np.ravel(y_train))
        y_pred_knn = neigh.predict(X_test)
        y_train_pred_knn = neigh.predict(X_train)
        # capture the end time of calculation
        end = time.time()


        #Storing the metrics
        runningTime.append(end-start)
        param.append(i)
        trainAccuracy.append((accuracy_score(y_train, y_train_pred_knn)))
        testAccuracy.append(accuracy_score(y_test, y_pred_knn))

    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for KNN is generated in the output folder")

    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/knn/").mkdir(parents=True, exist_ok=True)
    plt.clf()

    #Generating Test accuracy plot
    plt.plot(c, trainAccuracy, 'ro-', c, testAccuracy, 'bv--')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('KNN value')
    plt.ylabel('Accuracy')
    plt.title("KNN-Accuracy")
    strFile = str(path)+"/output/knn"+"/Accuracy.png"

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
    # Finding optimum k value
    maxValue = max(cv_scores)
    max_index = cv_scores.index(maxValue)
    optimum_param = param[max_index]

    # Printing the average running time
    print("The average running time - %.3f seconds" % mean(runningTime))
    print("The maximum Cross validation score - %.3f" %max(cv_scores))
    print("The maximum test accuracy  - %.3f " % max(testAccuracy))
    print("Corresponding K value ", optimum_param)


    # Training a classifier
    pca = PCA(n_components=2)
    X_transform = pca.fit_transform(X_1_df)


    X_train1, X_test1, y_train1, y_test1 = train_test_split(pd.DataFrame(X_transform),Y_1_df,random_state=1, test_size=0.2)

    clf = KNeighborsClassifier(n_neighbors=optimum_param).fit(X_train1, np.ravel(y_train1))

    #Generating decision boundary chart

    y = pd.DataFrame(Y_1_df).to_numpy()
    y = y.astype(np.int).flatten()

    plot_decision_regions(X_transform, y , clf=clf, legend=2)

    # Adding axes annotations
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    strFile = str(path)+"/output/knn"+"/Decision Boundary.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()



    #Printing the classification report
    vizualizer = ClassificationReport(KNeighborsClassifier(n_neighbors=optimum_param), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/knn"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return