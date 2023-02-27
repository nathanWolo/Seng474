import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data
from k_fold_cross_validation.kfcv import k_fold_cross_validation, k_fold_separation
import sklearn.svm as svm
import matplotlib.pyplot as plt
import time
train_df, test_df = read_data()

train_df = filter_data(train_df).sample(frac=1, random_state=1234)
test_df = filter_data(test_df)

folds = k_fold_separation(train_df)


def plot_error_gaussian_svm_varying_gamme_and_c(gamma_values):
    ''' This function creates an SKL gaussian SVM model and plots the error for varying values of the scale parameter gamma.
    For each gamma, we will use k fold cross validation to find an optimal C_gamma value. We will then use the optimal C_gamma
    To train on the entire training set and evaluate on the test set. We will then plot the test set error for each gamma value.'''
    accuracies = []
    best_acc = 0
    best_gamma = 0
    c_gammas = []
    for gamma in gamma_values:
        start_time = time.perf_counter()
        print('gamma: ', gamma)
        cur_gamma_c_values = [1.5**i for i in range(-5, 8)]
        best_c = 0
        best_c_score = 0
        for c in cur_gamma_c_values:
            print('c: ', c)
            gs = svm.SVC(C=c, gamma=gamma)
            cur_c_score = k_fold_cross_validation(folds, gs)
            if cur_c_score > best_c_score:
                best_c_score = cur_c_score
                best_c = c
        c_gammas.append(best_c)
        print(best_c)
        gs = svm.SVC(C=best_c, gamma=gamma)
        gs.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])
        cur_gamma_score = gs.score(test_df.iloc[:, :-1], test_df.iloc[:, -1])
        accuracies.append(cur_gamma_score)
        print(time.perf_counter() - start_time)
        if cur_gamma_score > best_acc:
            best_acc = cur_gamma_score
            best_gamma = gamma
        
    plt.plot(gamma_values, accuracies)
    plt.xlabel('Gamma')
    plt.xscale('log', base=1.5)
    plt.ylabel('accuracy')
    plt.title('Error vs Gamma')
    plt.plot(best_gamma, best_acc, 'ro', label='Best score: '
        + "{:.4f}".format(best_acc) + ' at gamma: ' + "{:.4f}".format(best_gamma))
    plt.legend(loc='best')
    plt.savefig('error_vs_gamma.png')

gamma_values = [(1.5**i) for i in range(-3, 8)]
plot_error_gaussian_svm_varying_gamme_and_c(gamma_values)

