import sys
sys.path.append('..')
from utils import get_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
# Load the datasets
data1 = get_data.load_dataset(1).to_numpy()
data2 = get_data.load_dataset(2).to_numpy()

def plot_hac_clusters(data, hac_labels, linkage, dataset_number, threshold):
    num_clusters = len(np.unique(hac_labels))

    if dataset_number == 2:
        # Create a 3D scatter plot of the data points, with colors determined by the labels
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=hac_labels, cmap='viridis')
    else:
        # Create a scatter plot of the data points, with colors determined by the labels
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=hac_labels, cmap='viridis')

    # Add title and labels
    ax.set_title(f'{linkage.capitalize()} Linkage, Dataset {dataset_number}, Threshold: {threshold}, Clusters: {num_clusters}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if dataset_number == 2:
        ax.set_zlabel('Z')

    # Save the plot as an image
    plt.savefig(f'hac_{linkage}_dataset{dataset_number}_threshold{threshold}.png')
    plt.clf()
    plt.close()

def plot_dendrogram(data, linkage_method, dataset_number):
    # Calculate the linkage matrix
    Z = hierarchy.linkage(data, method=linkage_method)

    # Plot the dendrogram
    plt.figure()
    hierarchy.dendrogram(Z, truncate_mode='lastp', p=30)  # You can adjust 'p' to show a different number of clusters
    plt.title(f'Dendrogram for Dataset {dataset_number} ({linkage_method.capitalize()} Linkage)')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')

    # Save the plot as an image
    plt.savefig(f'dendrogram_{linkage_method}_dataset{dataset_number}.png')
    plt.clf()
    plt.close()

# Define the linkage methods and datasets
linkage_methods = ['single', 'average']
datasets = [(data1, 1), (data2, 2)]

# Define a function to compute the HAC with a specific linkage method and threshold
def run_hac(data, linkage, threshold):
    # Create the agglomerative clustering model
    model = AgglomerativeClustering(linkage=linkage, n_clusters=None, distance_threshold=threshold)
    
    # Fit the model and predict the cluster labels
    hac_labels = model.fit_predict(data)
    
    return hac_labels

# Iterate through the datasets and linkage methods
for data, dataset_number in datasets:
    for linkage in linkage_methods:
        # Define a list of reasonable threshold values to test
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        plot_dendrogram(data, linkage, dataset_number)
        for threshold in thresholds:
            # Run HAC with the specified linkage method and threshold
            hac_labels = run_hac(data, linkage, threshold)
            
            # Plot the clusters
            plot_hac_clusters(data, hac_labels, linkage, dataset_number, threshold)