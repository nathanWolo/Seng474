import sys
sys.path.append('..')
from utils import get_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data1 = get_data.load_dataset(1)
data2 = get_data.load_dataset(2)
# print(data1.head())
# print(data2.head())

import numpy as np

def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two points, a and b.

    Args:
        a (numpy.ndarray): A point in n-dimensional space.
        b (numpy.ndarray): Another point in n-dimensional space.

    Returns:
        float: The Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)

def random_initialization(data, k):
    """
    Initialize k centroids with uniform random initialization.

    Args:
        data (numpy.ndarray): The dataset containing n data points.
        k (int): The number of centroids to initialize.

    Returns:
        numpy.ndarray: k randomly chosen initial centroids.
    """
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]

def kmeans_plus_plus_initialization(data, k):
    """
    Initialize k centroids with k-means++ initialization.

    Args:
        data (numpy.ndarray): The dataset containing n data points.
        k (int): The number of centroids to initialize.

    Returns:
        numpy.ndarray: k initial centroids using k-means++ initialization.
    """
    # Randomly choose the first centroid
    centroids = [data[np.random.choice(data.shape[0])]]
    
    # Choose remaining centroids based on their squared distances to the closest existing centroid
    for _ in range(1, k):
        squared_distances = np.array([min([euclidean_distance(c, x)**2 for c in centroids]) for x in data])
        probabilities = squared_distances / np.sum(squared_distances)
        centroids.append(data[np.random.choice(data.shape[0], p=probabilities)])
    
    return np.array(centroids)

def kmeans_clustering(data, k, initialization_method='random', n_init=10, max_iter=100, tol=1e-4):
    """
    Run k-means clustering on the given data using the specified initialization method.

    Args:
        data (numpy.ndarray): The dataset containing n data points.
        k (int): The number of clusters to form.
        initialization_method (str): 'random' for uniform random initialization or 'k-means++' for k-means++ initialization.
        n_init (int): The number of times the algorithm will be run with different centroid seeds.
        max_iter (int): The maximum number of iterations for a single run of the algorithm.
        tol (float): The tolerance for convergence.

    Returns:
        (numpy.ndarray, numpy.ndarray): A tuple containing the best centroids and labels found after running the algorithm n_init times.
    """
    best_centroids = None
    best_labels = None
    lowest_cost = float('inf')
    best_iters = None
    for _ in range(n_init):
        iters_k = 0
        # Initialize centroids based on the specified method
        if initialization_method == 'random':
            centroids = random_initialization(data, k)
        elif initialization_method == 'k-means++':
            centroids = kmeans_plus_plus_initialization(data, k)
        else:
            raise ValueError("Invalid initialization method")
        
        # Iterate until convergence or max_iter is reached
        for j in range(max_iter):
            iters_k += 1
            old_centroids = centroids.copy()
            
            # Assign each point to the nearest centroid
            distances = np.array([[euclidean_distance(x, c) for c in centroids] for x in data])
            labels = np.argmin(distances, axis=1)
            
            # Update the centroids based on the mean of the assigned points
            for i in range(k):
                centroids[i] = np.mean(data[labels == i], axis=0)

            # Check for convergence
            if np.linalg.norm(centroids - old_centroids) < tol:
                break
        else:
            print(f"Warning: k-means did not converge within {max_iter} iterations")

        # Compute the cost for the current run
        cost = np.sum([distances[i, labels[i]] for i in range(data.shape[0])])

        # Update the best centroids and labels if the current cost is lower
        if cost < lowest_cost:
            best_centroids = centroids
            best_labels = labels
            lowest_cost = cost
            best_iters = iters_k

    return best_centroids, best_labels, lowest_cost, best_iters


#2d scatterplot the data1

plt.scatter(data1['x'], data1['y'], c='b', marker='o', s=40, alpha=0.5)
plt.savefig('2d_scatter.png')
#3d scatterplot the data2
def plot_3d_scatter(df):
    """
    Creates a 3D scatterplot for the given dataset.
    
    Args:
        df (pandas.DataFrame): The dataset containing 'x', 'y', and 'z' columns.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['x']
    y = df['y']
    z = df['z']

    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('3d_scatter.png')
    plt.close()
# Load the second dataset
df = get_data.load_dataset(2)

# Plot the 3D scatterplot
plot_3d_scatter(df)

def compute_cost(data, centroids, labels):
    """
    Compute the cost of a k-means clustering solution.

    Args:
        data (numpy.ndarray): The dataset containing n data points.
        centroids (numpy.ndarray): The k centroids.
        labels (numpy.ndarray): The cluster assignment for each data point.

    Returns:
        float: The total cost of the clustering solution.
    """
    return np.sum([euclidean_distance(data[i], centroids[labels[i]])**2 for i in range(data.shape[0])])

def plot_clusters(data, centroids, labels, title, output_filename):
    fig = plt.figure()
    
    if data.shape[1] == 3:  # Check if the data has 3 dimensions
        ax = fig.add_subplot(111, projection='3d')
        for cluster_idx, centroid in enumerate(centroids):
            points = data[labels == cluster_idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=30)
            ax.scatter(centroid[0], centroid[1], centroid[2], marker='x', s=100, c='k', linewidths=3)
    else:
        ax = fig.add_subplot(111)
        for cluster_idx, centroid in enumerate(centroids):
            points = data[labels == cluster_idx]
            ax.scatter(points[:, 0], points[:, 1], marker='o', s=30)
            ax.scatter(centroid[0], centroid[1], marker='x', s=100, c='k', linewidths=3)

    ax.set_title(title)
    plt.savefig(output_filename)
    plt.clf()
    plt.close()


def plot_cost_vs_k(ks, costs_random, costs_pp, title, output_filename):
    plt.clf()
    print('ks:', ks)
    print('costs_random:', costs_random)
    print('costs_pp:', costs_pp)
    plt.plot(ks, costs_random, marker='o', label='Random initialization')
    plt.plot(ks, costs_pp, marker='o', label='k-means++ initialization')
    plt.xlabel('k')
    plt.ylabel('Cost')
    plt.title(title)
    plt.legend()
    plt.savefig(output_filename)
    plt.clf()
    plt.close()

def run_kmeans_experiment(data, ks, initialization_method, dataset_name):
    data = data.to_numpy()  # Convert the input data to a NumPy array
    costs = []
    iters = [] 
    for k in ks:
        print(f'Running k-means with k = {k} and {initialization_method} initialization, on {dataset_name}...')
        centroids, labels, cost, iters_k = kmeans_clustering(data, k, initialization_method)
        costs.append(cost)
        iters.append(iters_k)
        plot_clusters(data, centroids, labels, f'k = {k} ({initialization_method} initialization)',
                      f'clusters_{dataset_name}_k_{k}_{initialization_method}.png')
    return costs, iters
def plot_iterations_vs_k(ks, iters_random, iters_pp, title, output_filename):
    plt.plot(ks, iters_random, marker='o', label='Random initialization')
    plt.plot(ks, iters_pp, marker='o', label='k-means++ initialization')
    plt.xlabel('k')
    plt.ylabel('Iterations')
    plt.title(title)
    plt.legend()
    plt.savefig(output_filename)
    plt.clf()
    plt.close()
def main():
    # Load dataset 1
    data1 = get_data.load_dataset(1)
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25]

    # Run experiments for dataset 1
    costs_random1, iters_random1 = run_kmeans_experiment(data1, ks, 'random', 'dataset1')
    costs_pp1, iters_pp1 = run_kmeans_experiment(data1, ks, 'k-means++', 'dataset1')

    # Plot cost vs k for dataset 1
    plot_cost_vs_k(ks, costs_random1, costs_pp1, 'Cost vs K (Dataset 1)', 'cost_vs_k_dataset1.png')
    # Plot iterations vs k for dataset 1
    plot_iterations_vs_k(ks, iters_random1, iters_pp1, 'Iterations vs K (Dataset 1)', 'iterations_vs_k_dataset1.png')
    # Load dataset 2
    data2 = get_data.load_dataset(2)

    # Run experiments for dataset 2
    costs_random2, iters_random2 = run_kmeans_experiment(data2, ks, 'random', 'dataset2')
    costs_pp2, iters_pp2 = run_kmeans_experiment(data2, ks, 'k-means++', 'dataset2')

    # Plot cost vs k for dataset 2
    plot_cost_vs_k(ks, costs_random2, costs_pp2, 'Cost vs K (Dataset 2)', 'cost_vs_k_dataset2.png')
    # Plot iterations vs k for dataset 2
    plot_iterations_vs_k(ks, iters_random2, iters_pp2, 'Iterations vs K (Dataset 2)', 'iterations_vs_k_dataset')

if __name__ == '__main__':
    main()
