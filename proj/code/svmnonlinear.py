#SVM-NonLinear classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

def svmNonLinearTest(X_train, X_test, y_train, y_test):
    print('-----------------------------')    
    print('SVM NonLinear Test was Called. Wait...')

    # fit the training dataset to rbf kernel model
    # capture the start time
    start = time.time()
    rbf_svc = SVC(kernel = 'rbf', gamma=0.7, C=1.0)
    rbf_svc.fit(X_train, y_train.values.ravel())

    y_pred_rbf = rbf_svc.predict(X_test)
    # capture the end time of calculation
    end = time.time()

    print('Accuracy with rbf kernel: %.2f' % accuracy_score(y_test, y_pred_rbf))
    print('Time taken :', end-start, '(seconds)')
    print('-----------------------------')

    # fit the training dataset to poly kernel model
    # capture the start time
    start = time.time()
    poly_svc = SVC(kernel = 'poly', degree=3, C=1.0)
    poly_svc.fit(X_train, y_train.values.ravel())

    y_pred_poly = poly_svc.predict(X_test)
    # capture the end time of calculation
    end = time.time()

    print('Accuracy with poly kernel: %.2f' % accuracy_score(y_test, y_pred_poly))
    print('Time taken :', end-start, '(seconds)')
    print('-----------------------------')
    return 