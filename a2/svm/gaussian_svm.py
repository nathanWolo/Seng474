import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data
from k_fold_cross_validation.kfcv import k_fold_cross_validation, k_fold_separation
import sklearn.svm as svm
import matplotlib.pyplot as plt

train_df, test_df = read_data()

train_df = filter_data(train_df)
test_df = filter_data(test_df)

folds = k_fold_separation(train_df)


def plot_error_gaussian_svm_varying_gamme_and_c(gamma_values):
    ''' This function creates an SKL gaussian SVM model and plots the error for varying values of the scale parameter gamma.
    For each gamma, we will use k fold cross validation to find an optimal C_gamma value. We will then use the optimal C_gamma
    To train on the entire training set and evaluate on the test set. We will then plot the test set error for each gamma value.'''
    errors = []
    best_error = 1
    best_gamma = 0
    c_gammas = []
    for gamma in gamma_values:
        print('gamma: ', gamma)
        cur_gamma_c_values = [1.5**i for i in range(-2, 3)]
        best_c = 0
        best_c_error = 1
        for c in cur_gamma_c_values:
            print('c: ', c)
            gs = svm.SVC(C=c, gamma=gamma)
            cur_error = k_fold_cross_validation(folds, gs)
            if cur_error < best_c_error:
                best_c_error = cur_error
                best_c = c
        c_gammas.append(best_c)
        gs = svm.SVC(C=best_c, gamma=gamma)
        gs.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])
        cur_error = 1 - gs.score(test_df.iloc[:, :-1], test_df.iloc[:, -1])
        errors.append(cur_error)
        if cur_error < best_error:
            best_error = cur_error
            best_gamma = gamma
        
    plt.plot(gamma_values, errors)
    plt.xlabel('Gamma')
    plt.ylabel('Error')
    plt.title('Error vs Gamma')
    plt.savefig('error_vs_gamma.png')

gamma_values = [(1/784)/(1.5**i) for i in range(-1, 2)]
plot_error_gaussian_svm_varying_gamme_and_c(gamma_values)

