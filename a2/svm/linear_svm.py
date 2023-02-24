import sys
sys.path.append('..')
from utils.read_data import read_data, filter_data
import matplotlib.pyplot as plt
import sklearn.svm as svm
import time
train_df, test_df = read_data()

train_df = filter_data(train_df)
test_df = filter_data(test_df)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]



'''Training an skl linear svm, varying C and plotting the accuracy of the model as a function of C.'''
def plot_accuracy_vs_C(X_train, y_train, X_test, y_test, C_values):
    accuracies = []
    best_accuracy = 0
    best_C = 0
    for C in C_values:
        model = svm.LinearSVC(C=C, max_iter=10000)
        model.fit(X_train, y_train)
        accuracies.append(model.score(X_test, y_test))
        if model.score(X_test, y_test) > best_accuracy:
            best_accuracy = model.score(X_test, y_test)
            best_C = C
    plt.plot(C_values, accuracies)
    plt.plot(best_C, best_accuracy, 'ro', label='Best score: '
        + "{:.4f}".format(best_accuracy) + ' at C: ' + str(best_C))
    plt.xlabel("C")
    plt.xscale("log", base=1.5)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig("lin_svm_accuracy_vs_C.png")

start_time = time.perf_counter()
C_values = [1.5**i for i in range(-8, 8)]
plot_accuracy_vs_C(X_train, y_train, X_test, y_test, C_values)
end_time = time.perf_counter()
print("Time taken: ", end_time - start_time)