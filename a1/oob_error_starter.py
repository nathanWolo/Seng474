from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state  
import numpy as np

## This is some dummy data just so you have a complete working example
X = [[0, 0], [1, 1], [0, 0], [1, 1],[0, 0], [1, 1], [0, 0], [1, 1]]
Y = [0, 1, 1, 0, 0, 1, 1, 0]
M = 10 # number of trees in random forest
rf = RandomForestClassifier(n_estimators = M, random_state = 0)
rf = rf.fit(X, Y)
n_samples = len(X)
n_samples_bootstrap = n_samples


## THE ACTUAL STARTER CODE YOU SHOULD GRAB BEGINS BELOW

## Assumptions
#    - n_samples is the number of examples
#    - n_samples_bootstrap is the number of samples in each bootstrap sample
#      (this should be equal to n_samples)
#    - rf is a random forest, obtained via a call to
#      RandomForestClassifier(...) in scikit-learn

unsampled_indices_for_all_trees= []
for estimator in rf.estimators_:
    random_instance = check_random_state(estimator.random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength = n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    unsampled_indices_for_all_trees += [unsampled_indices]

## Result:
#    unsampled_indices_for_all_trees is a list with one element for each tree
#    in the forest. In more detail, the j'th element is an array of the example
#    indices that were \emph{not} used in the training of j'th tree in the
#    forest. For examle, if the 1st tree in the forest was trained on a
#    bootstrap sample that was missing only the first and seventh training
#    examples (corresponding to indices 0 and 6), and if the last tree in the
#    forest was trained on a boostrap sample that was missing the second,
#    third, and sixth training examples (indices 1, 2, and 5), then
#    unsampled_indices_for_all_trees would begin like:  
#        [array([0, 6]),
#         ...
#         array([1, 2, 5])]

print(unsampled_indices_for_all_trees)
