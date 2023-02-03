import sys
sys.path.append('..')
from utils import read_data
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


train, test = read_data.partition_data(read_data.read_data())


#create dt
dt = DecisionTreeClassifier(random_state=0, criterion="entropy").fit(train.drop(columns=['income']), train['income'])

#prune dt
path = dt.cost_complexity_pruning_path(train.drop(columns=['income']), train['income'])
ccp_alphas, impurities = path.ccp_alphas, path.impurities

depths = []
scores = []
i = 0
best_alpha = ccp_alphas[0]
best_score = 0
for ccp_alpha in ccp_alphas:
    if i % 100 == 0:
        print(i)
    dt_pruned = DecisionTreeClassifier(random_state=0, criterion="entropy", ccp_alpha=ccp_alpha).fit(train.drop(columns=['income']), train['income'])
    if dt_pruned.score(test.drop(columns=['income']), test['income']) > best_score:
            best_score = dt_pruned.score(test.drop(columns=['income']), test['income'])
            best_alpha = ccp_alpha
    depths.append(dt_pruned.get_depth())
    scores.append(dt_pruned.score(test.drop(columns=['income']), test['income']))
    i+=1

'''Plot the scores on both the training and test sets using matplotlib from varying alpha values'''

def plot_alpha_scores():
    plt.plot(ccp_alphas, scores, label='Scores')
    plt.plot(best_alpha, best_score, 'ro', label='Best score: '
        + "{:.4f}".format(best_score) + ' at alpha: ' + str(best_alpha))
    plt.legend(loc="upper right")
    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.title('Entropy, pruning')
    plt.savefig('entropy_pruning_scores_varying_alpha.png')
    plt.clf()

'''plot the depths as alpha varies'''
def plot_alpha_depths():
    plt.plot(ccp_alphas, depths, label='Depth')
    plt.legend(loc="upper right")
    plt.xlabel('Alpha')
    plt.ylabel('Depth')
    plt.title('Entropy, pruning')
    plt.savefig('entropy_pruning_depths_varying_alpha.png')
    plt.clf()

plot_alpha_scores()
plot_alpha_depths()