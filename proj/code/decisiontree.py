#Decision tree-classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

def decisionTreeTest(X_train, X_test, y_train, y_test):\
    
    print('-----------------------------')    
    print('DecisionTree Test was Called. Wait...')
    
    
    
    #start = time.time()
    
    
    
    # Create Decision Tree classifer object
    clf_gini = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf_gini.fit(X_train,y_train)
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
 
    
    print('Accuracy with gini: %.2f' % accuracy_score(y_test, y_pred))
    print('-----------------------------')
    
    
    
    
    
    clf_entropy = DecisionTreeClassifier()
    
    clf2 = clf_entropy.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    
    
    
    print('Accuracy with entropy: %.2f' % accuracy_score(y_test, y_pred))
    print('-----------------------------')
    
    
    
    
    
    
    #end = time.time()
    #print('Running Time is: ', end - start, "seconds")