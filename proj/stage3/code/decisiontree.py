#Decision tree-classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from pathlib import Path
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
import os
def decisionTreeTest(X_train, X_test, y_train, y_test, classes):
    
    print('-----------------------------')    
    print('DecisionTree Test was Called. Wait...')
    
    
    
    path = Path(__file__).parent.absolute()
    depths= list(range(1, 51))
    runningTime = []
    trainAccuracy = []
    testAccuracy = []
    
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

    print("The maximum test accuracy - %.3f and the corresponding max_depth value "%(testAccuracy[max_index]))
    print(depths[max_index])

    #Printing the classification report
    vizualizer = ClassificationReport(DecisionTreeClassifier(criterion='gini', max_depth= depths[max_index]), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
    vizualizer.fit(X_train, y_train.values.ravel())
    vizualizer.score(X_test, y_test)
    strFile = str(path)+"/output/DecisionTree"+"/Classification Report.png"

    if os.path.isfile(strFile):
        os.remove(strFile)
    vizualizer.show(strFile)

    plt.clf()

    return
    
    
    
    
    
    
    
    
    # Create Decision Tree classifer object
    #clf_gini = DecisionTreeClassifier(criterion='gini')

    # Train Decision Tree Classifer
    #clf = clf_gini.fit(X_train,y_train)
    
    #Predict the response for test dataset
    #y_pred = clf.predict(X_test)
    
 
    
    #print('Accuracy with gini: %.2f' % accuracy_score(y_test, y_pred))
    #print('-----------------------------')
    
    
    
    
    '''
    clf_entropy = DecisionTreeClassifier(criterion='entropy')
    
    clf2 = clf_entropy.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    
    
    
    print('Accuracy with entropy: %.2f' % accuracy_score(y_test, y_pred))
    print('-----------------------------')
    
    
    
    
    
    
    #end = time.time()
    #print('Running Time is: ', end - start, "seconds")
    '''