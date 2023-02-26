import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data
import sklearn.linear_model as lm
import sklearn.svm as svm
import matplotlib.pyplot as plt


train_df, test_df = read_data()

train_df = filter_data(train_df)
test_df = filter_data(test_df)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

linsvm = svm.LinearSVC(C=0.6667)

logreg = lm.LogisticRegression(C=7.5938)

linsvm.fit(X_train, y_train)

logreg.fit(X_train, y_train)

print("Linear SVM accuracy: ", linsvm.score(X_test, y_test))

print("Logistic Regression accuracy: ", logreg.score(X_test, y_test))

