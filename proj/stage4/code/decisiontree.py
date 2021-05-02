#Decision tree-classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from pathlib import Path
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean
import numpy as np

def decisionTreeTest(X_train, X_test, y_train, y_test, classes,  X_1_df, Y_1_df):
    
    print('-----------------------------')    
    print('DecisionTree Test was Called. Wait...')
    
    
    
    path = Path(__file__).parent.absolute()
    depths= list(range(1, 51))
    runningTime = []
    trainAccuracy = []
    testAccuracy = []
    param = []

    #fit the training dataset to linear kernel model
    #capture the start time
    for i in depths:
        start = time.time()
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=i)
        clf_gini.fit(X_train, y_train.values.ravel())

        y_pred_gini = clf_gini.predict(X_test)
        y_train_pred_gini = clf_gini.predict(X_train)
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        param.append(i)
        trainAccuracy.append(accuracy_score(y_train, y_train_pred_gini))
        testAccuracy.append(accuracy_score(y_test, y_pred_gini))
    
    #print(trainAccuracy)
    #print(testAccuracy)
    
    
    #Printing the metrics/Generating visualization
    print("Classification report, class prediction error, Test accuracy, Running time for DecisionTree is generated in the output folder")
    #Creates a new directory under svm-linear if it doesn't exist
    Path("output/DecisionTree/").mkdir(parents=True, exist_ok=True)
    plt.clf()


    #Generating Test accuracy plot


    plt.plot(depths, trainAccuracy, 'ro-',depths, testAccuracy, 'bv--')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('max_depth values')
    plt.ylabel('Accuracy')
    plt.title("Decison Tree - Accuracy")
    strFile = str(path)+"/output/DecisionTree"+"/Accuracy.png"

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
    strFile = str(path)+"/output/DecisionTree"+"/Running Time.png"

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
    print("Corresponding MaxDepth value for max test accuracy", optimum_param)


    pca = PCA(n_components=2)
    X_transform = pca.fit_transform(X_1_df)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(pd.DataFrame(X_transform), Y_1_df, random_state=1,
                                                            test_size=0.2)

    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=optimum_param).fit(X_train1, np.ravel(y_train1))

    # Generating decision boundary chart

    y = pd.DataFrame(Y_1_df).to_numpy()
    y = y.astype(np.int).flatten()

    plot_decision_regions(X_transform, y, clf=clf_gini, legend=2)

    # Adding axes annotations
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    strFile = str(path) + "/output/DecisionTree" + "/Decision Boundary.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    #Printing the classification report
    vizualizer = ClassificationReport(DecisionTreeClassifier(criterion='gini', max_depth= optimum_param), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/DecisionTree"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return
    
    
    
    
    
    
    
