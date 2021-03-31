#SVM-NonLinear classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_graph(folder_name, model, X_train, y_train, X_test,y_test,c, runningTime, testAccuracy, trainAccuracy):
         #Printing the metrics/Generating visualization
        print("Classification report, class prediction error, Test accuracy, Running time for SVM-non linear is generated in the output folder")
        path = Path(__file__).parent.absolute()
        #Creates a new directory under svm-linear if it doesn't exist
        Path("output/"+folder_name+"/"+model).mkdir(parents=True, exist_ok=True)
        plt.clf()

        #Generating Test accuracy plot

        plt.plot(c, testAccuracy, 'ro-')
        plt.legend(['Test Accuracy'])
        plt.xlabel('C Param value')
        plt.ylabel('Accuracy')
        plt.title("Test Accuracy")
        strFile = str(path)+"/output/"+folder_name+"/"+model+"/Test Accuracy.png"

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
        strFile = str(path)+"/output/"+folder_name+"/"+model+"/Running Time.png"

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
        vizualizer = ClassificationReport(SVC(kernel=model, gamma = 0.7, C= c[max_index], random_state=1), classes=[0,1,2,3,4,5], support=True, size=(1400, 1000))
        vizualizer.fit(X_train, y_train.values.ravel())
        vizualizer.score(X_test, y_test)
        strFile = str(path)+"/output/"+folder_name+"/"+model+"/Classification Report.png"

        if os.path.isfile(strFile):
            os.remove(strFile)
        vizualizer.show(strFile)

        plt.clf()

def svmNonLinearTest(X_train, X_test, y_train, y_test):
    print('-----------------------------')    
    print('SVM NonLinear Test was Called. Wait...')

    #hyperparameters test set
    c= [0.01, 0.1, 1.0, 5, 10, 15,20]
    runningTime = []
    trainAccuracy = []
    testAccuracy = []

    for i in c:
        # fit the training dataset to rbf kernel model
        # capture the start time
        start = time.time()
        rbf_svc = SVC(kernel = 'rbf', gamma=0.7, C=1.0)
        rbf_svc.fit(X_train, y_train.values.ravel())

        y_pred_rbf = rbf_svc.predict(X_test)
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred_rbf))

    plot_graph("svm-nonlinear", "rbf", X_train, y_train, X_test, y_test, c, runningTime, testAccuracy, trainAccuracy)

    runningTime = []
    trainAccuracy = []
    testAccuracy = []

    for i in c:
        # fit the training dataset to poly kernel model
        # capture the start time
        start = time.time()
        poly_svc = SVC(kernel = 'poly', degree=3, C=1.0)
        poly_svc.fit(X_train, y_train.values.ravel())

        y_pred_poly = poly_svc.predict(X_test)
        # capture the end time of calculation
        end = time.time()

        #Storing the metrics
        runningTime.append(end-start)
        testAccuracy.append(accuracy_score(y_test, y_pred_rbf))

    plot_graph("svm-nonlinear", "poly", X_train, y_train,X_test, y_test,c,runningTime, testAccuracy, trainAccuracy)
    
    return 

