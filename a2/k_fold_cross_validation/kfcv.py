import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data
import sklearn.linear_model as lm
import sklearn.svm as svm
import matplotlib.pyplot as plt
import time
'''Here we will be implementing the k-fold cross validation algorithm

1. Load data
2. Split data into k folds
3. For i in range(k):
    a. Select one fold to act as the validation set
    b. Select the remaining folds to act as the training set
    c. Fit the model on the training set and evaluate it on the validation set
    d. Compute the validation set performance metric
4. Computer the risk estimate as the average of the k validation set performance metrics

'''

# Load data
train_df, test_df = read_data()
train_df = filter_data(train_df)
test_df = filter_data(test_df)

def k_fold_separation(test_data, k=5):
    '''This function takes in a dataframe and returns a list of dataframes, each of which is a fold of the original dataframe.'''
    test_data_length = len(test_data)
    fold_length = test_data_length // k
    folds = []
    for i in range(k):
        folds.append(test_data.iloc[i*fold_length:(i+1)*fold_length])
    return folds

folds = k_fold_separation(train_df)
print(len(folds), len(folds[0]))

def k_fold_cross_validation(folds, model):
    '''this function takes in the folds and a SKL model, fits the model on each possible combination of folds and returns the average score of the k folds'''
    scores = []
    for i in range(len(folds)):
        validation_set = folds[i]
        training_set = pd.concat(folds[:i] + folds[i+1:])
        model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
        scores.append(model.score(validation_set.iloc[:, :-1], validation_set.iloc[:, -1]))
    return np.mean(scores)
lr = lm.LogisticRegression()
print(k_fold_cross_validation(folds, lr))
linsvm = svm.LinearSVC()
print(k_fold_cross_validation(folds, linsvm))


def plot_error_logistic_regression_varying_c(c_values):
    '''This function creates an SKL logistic regression model and plots the error for varying values of the regularization parameter C'''
    folds = k_fold_separation(train_df)
    errors = []
    best_error = 1
    best_c = 1
    for c in c_values:
        lr = lm.LogisticRegression(C=c)
        errors.append(1 - k_fold_cross_validation(folds, lr))
        if 1 - k_fold_cross_validation(folds, lr) < best_error:
            best_error = 1 - k_fold_cross_validation(folds, lr)
            best_c = c
    plt.plot(c_values, errors, label='Logistic Regression')
    plt.xscale("log", base=1.5)
    plt.xlabel("C")
    plt.ylabel("Error")
    plt.plot(best_c, best_error, 'ro', label='Best error LR: '
        + "{:.4f}".format(best_error) + ' at C: ' + "{:.4f}".format(best_c))
   # plt.legend(loc='best')
    plt.savefig("kfcv_logistic_error_vs_C.png")
start_time = time.perf_counter()
c_values = [1.5**i for i in range(-10, 10)]
plot_error_logistic_regression_varying_c(c_values)
print("Time taken: ", time.perf_counter() - start_time)

def plot_error_linear_svm_varying_c(c_values):
    '''This function creates an SKL linear SVM model and plots the error for varying values of the regularization parameter C'''
    folds = k_fold_separation(train_df)
    errors = []
    best_error = 1
    best_c = 1
    for c in c_values:
        linsvm = svm.LinearSVC(C=c)
        errors.append(1 - k_fold_cross_validation(folds, linsvm))
        if 1 - k_fold_cross_validation(folds, linsvm) < best_error:
            best_error = 1 - k_fold_cross_validation(folds, linsvm)
            best_c = c
    plt.plot(c_values, errors, label='Linear SVM')
    # plt.xscale("log", base=1.5)
    # plt.xlabel("C")
    # plt.ylabel("Error")
    # plt.legend(loc='best')
    plt.plot(best_c, best_error, 'ro', label='Best error SVM: '
        + "{:.4f}".format(best_error) + ' at C: ' + "{:.4f}".format(best_c))
    plt.legend(loc='best')
    plt.savefig("kfcv_linear_svm_error_vs_C.png")

start_time = time.perf_counter()
c_values = [1.5**i for i in range(-10, 10)]
plot_error_linear_svm_varying_c(c_values)
print("Time taken: ", time.perf_counter() - start_time)
