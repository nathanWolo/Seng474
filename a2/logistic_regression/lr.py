import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data


train_df, test_df = read_data()
train_df = filter_data(train_df)
test_df = filter_data(test_df)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
'''We'll use sklearn's LogisticRegression to train our model. We'll use l2 regularization, and study the effect of the regularization parameter C on the accuracy of the model.'''

# model = lm.LogisticRegression(C=1, multi_class="multinomial", solver="lbfgs", penalty="l2")
# # remember that small value of C implies stronger regularization
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))



def plot_accuracy_vs_C(X_train, y_train, X_test, y_test, C_values):
    '''A function to plot the accuracy of the model as a function of C.'''
    accuracies = []
    best_accuracy = 0
    best_C = 0
    for C in C_values:
        model = lm.LogisticRegression(C=C, multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=1000)
        model.fit(X_train, y_train)
        accuracies.append(model.score(X_test, y_test))
        if model.score(X_test, y_test) > best_accuracy:
            best_accuracy = model.score(X_test, y_test)
            best_C = C
    plt.plot(C_values, accuracies)
    plt.plot(best_C, best_accuracy, 'ro', label='Best score: '
        + "{:.4f}".format(best_accuracy) + ' at C: ' + str(best_C))
    plt.xlabel("C")
    plt.xscale("log", base=2)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig("logistic_accuracy_vs_C.png")

C_values = [2**i for i in range(-8, 8)]
plot_accuracy_vs_C(X_train, y_train, X_test, y_test, C_values)