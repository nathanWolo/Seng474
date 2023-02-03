from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np
sys.path.append('..')
from utils import read_data
import matplotlib.pyplot as plt


train, test = read_data.partition_data(read_data.read_data()) 
num_features = len(train.columns) - 1
num_estimators = int(np.floor(np.sqrt(num_features)))


'''Test various depths, with entropy as the criterion, 
store and plot the scores on both training and test sets using matplotlib'''
def test_entropy_depths():
    best_depth = 0
    best_score = 0
    test_scores = []
    training_scores = []
    for i in range(1, 30):
        model = RandomForestClassifier(n_estimators=num_estimators,max_depth=i, random_state=0, criterion="entropy")
        model.fit(train.drop(columns=['income']), train['income'])
        test_scores.append(model.score(test.drop(columns=['income']), test['income']))
        training_scores.append(model.score(train.drop(columns=['income']), train['income']))
        if model.score(test.drop(columns=['income']), test['income']) > best_score:
            best_score = model.score(test.drop(columns=['income']), test['income'])
            best_depth = i
    plt.plot(range(1, 30), test_scores, label='Test')
    plt.plot(range(1, 30), training_scores, label='Training')
    plt.plot(best_depth, best_score, 'ro', label='Best score: '
        + "{:.4f}".format(best_score) + ' at depth: ' + str(best_depth))
    plt.legend(loc="best")
    plt.xlabel('Depth')
    plt.ylabel('Score')
    plt.title('Entropy, no pruning')
    #plt.show()
    plt.savefig('rf_entropy_scores_varying_depth.png')
    plt.clf()


'''Test various depths, with gini as the criterion'''

def test_gini_depths():
    best_depth = 0
    best_score = 0
    test_scores = []
    training_scores = []
    for i in range(1, 30):
        model = RandomForestClassifier(n_estimators=num_estimators,max_depth=i, random_state=0, criterion="gini")
        model.fit(train.drop(columns=['income']), train['income'])
        test_scores.append(model.score(test.drop(columns=['income']), test['income']))
        training_scores.append(model.score(train.drop(columns=['income']), train['income']))
        if model.score(test.drop(columns=['income']), test['income']) > best_score:
            best_score = model.score(test.drop(columns=['income']), test['income'])
            best_depth = i
    plt.plot(range(1, 30), test_scores, label='Test')
    plt.plot(range(1, 30), training_scores, label='Training')
    plt.plot(best_depth, best_score, 'ro', label='Best score: '
        + "{:.4f}".format(best_score) + ' at depth: ' + str(best_depth))
    plt.legend(loc="best")
    plt.xlabel('Depth')
    plt.ylabel('Score')
    plt.title('Gini, no pruning')
    #plt.show()
    plt.savefig('rf_gini_scores_varying_depth.png')
    plt.clf()

# test_entropy_depths()
# test_gini_depths()

'''test various number of estimators, with gini as the criterion, depth 15'''

def test_gini_estimators():
    best_num_estimators = 0
    best_score = 0
    test_scores = []
    training_scores = []
    for i in range(1, 100):
        model = RandomForestClassifier(n_estimators=i,max_depth=15, random_state=0, criterion="gini")
        model.fit(train.drop(columns=['income']), train['income'])
        test_scores.append(model.score(test.drop(columns=['income']), test['income']))
        training_scores.append(model.score(train.drop(columns=['income']), train['income']))
        if model.score(test.drop(columns=['income']), test['income']) > best_score:
            best_score = model.score(test.drop(columns=['income']), test['income'])
            best_num_estimators = i
    plt.plot(range(1, 100), test_scores, label='Test')
    plt.plot(range(1, 100), training_scores, label='Training')
    plt.plot(best_num_estimators, best_score, 'ro', label='Best score: '
        + "{:.4f}".format(best_score) + ' at num estimators: ' + str(best_num_estimators))
    plt.legend(loc="best")
    plt.xlabel('Number of estimators')
    plt.ylabel('Score')
    plt.title('Gini, depth 15')
    #plt.show()
    plt.savefig('rf_gini_scores_varying_estimators.png')
    plt.clf()

#test_gini_estimators()

'''test various bootstrap values, with gini as the criterion, depth 15, 19 estimators'''

def test_gini_bootstrap():
    best_bootstrap = False
    best_score = 0
    test_scores = []
    training_scores = []
    for i in range(1,100):
        model = RandomForestClassifier(n_estimators=19,max_depth=15, random_state=0, criterion="gini", bootstrap=True, max_samples=i/100)
        model.fit(train.drop(columns=['income']), train['income'])
        test_scores.append(model.score(test.drop(columns=['income']), test['income']))
        training_scores.append(model.score(train.drop(columns=['income']), train['income']))
        if model.score(test.drop(columns=['income']), test['income']) > best_score:
            best_score = model.score(test.drop(columns=['income']), test['income'])
            best_bootstrap = i
    plt.plot(range(1,100), test_scores, label='Test')
    plt.plot(range(1,100), training_scores, label='Training')
    plt.plot(best_bootstrap, best_score, 'ro', label='Best score: '
        + "{:.4f}".format(best_score) + ' at bootstrap: ' + str(best_bootstrap))
    plt.legend(loc="best")
    plt.xlabel('Bootstrap')
    plt.ylabel('Score')
    plt.title('Gini, depth 15, 19 estimators')
    #plt.show()
    plt.savefig('rf_gini_scores_varying_bootstrap.png')
    plt.clf()

test_gini_bootstrap()