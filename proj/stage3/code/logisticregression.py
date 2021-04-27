from sklearn.linear_model import LogisticRegression
import numpy as np

def logisticRegressionTest(X_train, X_test, y_train, y_test):
    print('-----------------------------')    
    print('Logistic Regression Test was Called. Wait...')
    clf = LogisticRegression(random_state=0, max_iter = 200).fit(X_train, np.ravel(y_train))
    clf.predict(X_test)
