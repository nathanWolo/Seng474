from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state  
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import read_data
## This is some dummy data just so you have a complete working example
train, test = read_data.partition_data(read_data.read_data(), train_size=0.5, random_state=0)
X = train.drop(columns=['income']).values
Y = train['income'].values
n_samples = len(X)
n_samples_bootstrap = n_samples



## THE ACTUAL STARTER CODE YOU SHOULD GRAB BEGINS BELOW


def get_unsampled_indices(rf, n_samples, n_samples_bootstrap):
    '''input: 
    - n_samples is the number of examples
    - n_samples_bootstrap is the number of samples in each bootstrap sample
     (this should be equal to n_samples)
    - rf is a random forest, obtained via a call to
     RandomForestClassifier(...) in scikit-learn
    
    output:
    unsampled_indices_for_all_trees is a list with one element for each tree
    in the forest. In more detail, the j'th element is an array of the example
    indices that were \emph{not} used in the training of j'th tree in the
    forest. For examle, if the 1st tree in the forest was trained on a
    bootstrap sample that was missing only the first and seventh training
    examples (corresponding to indices 0 and 6), and if the last tree in the
    forest was trained on a boostrap sample that was missing the second,
    third, and sixth training examples (indices 1, 2, and 5), then
    unsampled_indices_for_all_trees would begin like:  
        [array([0, 6]),
            ...
        array([1, 2, 5])]


    '''

    unsampled_indices_for_all_trees= []
    for estimator in rf.estimators_:
        random_instance = check_random_state(estimator.random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(sample_indices, minlength = n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]
        unsampled_indices_for_all_trees += [unsampled_indices]
    return unsampled_indices_for_all_trees

''' a function that returns all the trees that were not trained on a particular example'''

def get_trees_not_trained_on_example(rf, n_samples, n_samples_bootstrap, example_index):
    '''input: 
    - n_samples is the number of examples
    - n_samples_bootstrap is the number of samples in each bootstrap sample
     (this should be equal to n_samples)
    - rf is a random forest, obtained via a call to
     RandomForestClassifier(...) in scikit-learn
    - example_index is an integer in the range [0, n_samples - 1]
    
    output:
    trees_not_trained_on_example is a list of the indices of the trees in the
    forest that were \emph{not} trained on the example with index
    example_index. For example, if the 1st tree in the forest was trained on a
    bootstrap sample that was missing only the first and seventh training
    examples (corresponding to indices 0 and 6), and if the last tree in the
    forest was trained on a boostrap sample that was missing the second,
    third, and sixth training examples (indices 1, 2, and 5), then
    trees_not_trained_on_example would be [0, 9] (since the 1st and last trees
    in the forest were not trained on the example with index 3).
    '''

    unsampled_indices_for_all_trees = get_unsampled_indices(rf, n_samples, n_samples_bootstrap)
    trees_not_trained_on_example = []
    for i in range(len(unsampled_indices_for_all_trees)):
        if example_index in unsampled_indices_for_all_trees[i]:
            trees_not_trained_on_example += [i]
    return trees_not_trained_on_example

#print(get_trees_not_trained_on_example(rf, n_samples, n_samples_bootstrap, 3))

''' a function that returns the average prediction of all the trees that were not trained on a particular example'''

def get_average_prediction(rf, n_samples, n_samples_bootstrap, example_index):
    '''input: 
    - n_samples is the number of examples
    - n_samples_bootstrap is the number of samples in each bootstrap sample
     (this should be equal to n_samples)
    - rf is a random forest, obtained via a call to
     RandomForestClassifier(...) in scikit-learn
    - example_index is an integer in the range [0, n_samples - 1]
    
    output:
    average_prediction is the average prediction of the trees in the forest
    that were \emph{not} trained on the example with index example_index. For
    example, if the 1st tree in the forest was trained on a bootstrap sample
    that was missing only the first and seventh training examples (corresponding
    to indices 0 and 6), and if the last tree in the forest was trained on a
    boostrap sample that was missing the second, third, and sixth training
    examples (indices 1, 2, and 5), then average_prediction would be the
    average prediction of the 1st and last trees in the forest (since these are
    the trees that were not trained on the example with index 3).
    '''
    print("Example index: ", example_index)
    trees_not_trained_on_example = get_trees_not_trained_on_example(rf, n_samples, n_samples_bootstrap, example_index)
    predictions = []
    for i in trees_not_trained_on_example:
        predictions += [rf.estimators_[i].predict(X[example_index].reshape(1, -1))[0]]
    average_prediction = np.mean(predictions)
    return average_prediction

# print(get_unsampled_indices(rf, n_samples, n_samples_bootstrap))
# print(get_trees_not_trained_on_example(rf, n_samples, n_samples_bootstrap, 4))
# print(get_average_prediction(rf, n_samples, n_samples_bootstrap, 4))

'''Function that gets average oob error for all examples'''

def get_average_oob_error(rf, n_samples, n_samples_bootstrap):
    '''input: 
    - n_samples is the number of examples
    - n_samples_bootstrap is the number of samples in each bootstrap sample
     (this should be equal to n_samples)
    - rf is a random forest, obtained via a call to
     RandomForestClassifier(...) in scikit-learn
    
    output:
    average_oob_error is the average out-of-bag error of the random forest.
    '''

    average_oob_error = 0
    for i in range(n_samples):
        average_oob_error += (get_average_prediction(rf, n_samples, n_samples_bootstrap, i) != Y[i])
    average_oob_error /= n_samples
    return average_oob_error

# print(get_average_oob_error(rf, n_samples, n_samples_bootstrap))

'''Function that plots the oob error for ensembles of different sizes'''
def plot_oob_error():
    ensemble_sizes = [ 2, 5, 10, 20, 50, 100]
    oob_errors = []
    for i in ensemble_sizes:
        rf = RandomForestClassifier(n_estimators = i, random_state=0)
        rf.fit(X, Y)
        oob_errors.append(get_average_oob_error(rf, n_samples, n_samples_bootstrap))
    plt.plot(ensemble_sizes, oob_errors)
    plt.xlabel('Ensemble Size')
    plt.ylabel('Average OOB Error')
    # plt.show()
    plt.savefig('oob_error.png')
plot_oob_error()