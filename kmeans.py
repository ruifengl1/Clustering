import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2, visual_plt=False, seed = 0):
    '''
    X: n-dimensional data
    k: number of clusters
    centroids: kmeans++ indicator
    max_iter: number of iterations to build clusters
    tolerance: A value to see if algorithm has converged. If the error is greater than the give tolerance value, continue running the algorithm until it gets below the tolerance value
    '''
    np.random.seed(seed)
    if centroids  == 'kmeans++':
        # initially select centorids using kmeans++
        centroids = select_centroids(X, k)
    else:
        # randomly select k centroids
        centroids = X[np.random.choice(X.shape[0], k, replace = False)]
    
    for i in range(max_iter):
        labels = []
        
        # compute distance between x and each centroid.
        for data in X:
            distances = []
            for centroid in centroids:
                distances.append(np.sum((data - centroid) ** 2))
            labels.append(np.argmin(distances))

        # plot clustering for the iteration
        if visual_plt:
            visualize_plot(X, centroids, labels, i)

        # recompute centroids based on the cluster for the iteration
        prev_centroids = centroids
        centroids = np.vstack([np.mean(X[np.array(labels)==i], axis=0) for i in range(k)])

        # check the tolerance
        centroids_square_diff = np.sum((prev_centroids - centroids) ** 2)
        if tolerance > 0.0 and centroids_square_diff <= tolerance:
            break
    
    return centroids, np.array(labels)



def select_centroids(X,k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    # pick initial first point randomly
    centroids = []
    idx = np.random.choice(X.shape[0], 1, replace = False)
    centroids.append(X[idx])

    # pick the next k-1 points
    for _ in range(k-1):
        dist = []
        for x in X:
            temp_dist = []
            for center in centroids:
                temp_dist.append(np.sum((x - center) ** 2))
            dist.append(min(temp_dist))
        centroids.append(X[np.argmax(dist)])
    
    return np.vstack(centroids)


def likely_confusion_matrix(y, labels):
    '''
    y: predicted value
    labels: true value
    '''
    # transform the predicted y to match the level of labels
    labels = np.array(labels)
    y_pred = np.zeros(len(y))
    for i in np.unique(y):
        val = stats.mode(labels[y==i])[0]
        y_pred[labels == val] = i
    
    # compute accuracy
    accuracy = np.mean(y == y_pred)
    print('clustering accuracy: ', accuracy)

    # construct confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)
    counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # plot confusion matrix
    sns.set(rc = {'figure.figsize':(10,5)})
    sns.heatmap(cf_matrix, annot=labels, annot_kws={"fontsize":20},  fmt='', cmap='Blues', cbar=False)
    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.title('Confusion Matrix', fontsize=17)


def visualize_plot(X, centroids, labels, i=None):
    '''
    X: data points
    centroids: centers of clusters
    labels: predicted y
    '''
    if i:
        print(f'{i} iteration')
    for i in np.unique(labels):
        plt.scatter(X[labels == i, 0] , X[labels == i, 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()

