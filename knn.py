import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y): # fit the training samples
        self.X_train = X
        self.y_train = y

    def predict(self, X):              # capital X means multiple samples
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):             # simple example
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]