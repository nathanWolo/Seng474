import sys
sys.path.append('..')
from utils import read_data
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


train, test = read_data.partition_data(read_data.read_data())


#create dt
dt = DecisionTreeClassifier(random_state=0, criterion="entropy").fit(train.drop(columns=['income']), train['income'])

#test dt
print(dt.score(test.drop(columns=['income']), test['income']))

#prune dt
path = dt.cost_complexity_pruning_path(train.drop(columns=['income']), train['income'])
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
i = 0
for ccp_alpha in ccp_alphas:
    i+=1
    print("Iteration: ", i)
    dt = DecisionTreeClassifier(random_state=0, criterion="entropy", ccp_alpha=ccp_alpha)
    dt.fit(train.drop(columns=['income']), train['income'])
    trees.append(dt)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(trees[-1].tree_.node_count, ccp_alphas[-1]))