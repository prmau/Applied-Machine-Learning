# Adaboost

from sklearn.metrics import accuracy_score
import time
from pathlib import Path
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.model_selection import cross_val_score

def ensembleClassifier(X_train, X_test, y_train, y_test, X_1_df, Y_1_df):
    print('-----------------------------')
    print('Ensemble Vote Classifier was Called. Wait...')

    clf1 = LogisticRegression(C=5.0,  class_weight='balanced', max_iter=10000, random_state= 1 )  # C = 5.0
    clf2 = SVC(kernel = 'linear', C=1.0, random_state=1)  # linear SVM C = 1.0
    clf3 = KNeighborsClassifier(n_neighbors=1)  # optimum_k = 1
    clf4 = DecisionTreeClassifier(max_depth=21, criterion='gini')  #

    labels = ['Logistic Regression', 'Support Vector Machine', 'K Nearest Neighbor', 'Decision Tree', 'Ensemble']
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4], weights=[1, 1, 1, 1])

    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], labels):
        clf.fit(X_1_df, Y_1_df)

        scores = cross_val_score(clf, X_1_df, Y_1_df.values.ravel(),
                                                 cv=20,
                                                 scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    return