"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Theresa Wakonig
"""

import numpy as np
import math

# implements the methods of a k-NN classifier

class KNNClassifier:
    '''
    A class object that implements the methods of a k-Nearest Neighbor classifier
    The class assumes there are only two labels, namely POS and NEG

    Attributes of the class
    -----------------------
    k : Number of neighbors
    X : A matrix containing the data points (train set)
    y : A vector with the labels
    dist : Distance metric used. Possible values are: 'euclidean'
    '''

    def __init__(self, X, y, metric):
        '''
        Constructor when X and Y are given.
        
        Parameters
        ----------
        X : Matrix with data points from training set (point cloud)
        Y : Vector with class labels
        metric : Name of the distance metric to use
        '''
        # Default values
        self.verbose = False
        self.k = 1

        # Parameters
        self.X = X
        self.y = y
        self.metric = metric


    def debug(self, switch):
        '''
        Method to set the debug mode.
        
        Parameters
        ----------
        switch : String with value 'on' or 'off'
        '''
        self.verbose = True if switch == "on" else False


    def set_k(self, k):
        '''
        Method to set the value of k.
        
        Parameters
        ----------
        k : Number of nearest neighbors
        '''
        self.k = k


    def _compute_distances(self, X, x):
        # returns vector with euclidean distances between x and all points in X
        euclidean_dist_vec = np.zeros(len(X))
        if self.metric == 'euclidean':
            for i in range(0, len(X)):
                euclidean_dist_vec[i] = math.sqrt(np.sum((x - X[i][:]) ** 2))
            return euclidean_dist_vec
        else:
            print('This metric is unknown to me. Try metric "euclidean"!')
            return 0


    def _count_nn_labels(self, labels_knn):

        num_pos = np.count_nonzero(labels_knn == 1)
        num_neg = np.count_nonzero(labels_knn == -1)

        if num_pos > num_neg:
            return 1
        elif num_pos < num_neg:
            return -1
        else:
            return 0


    def predict(self, x):
        # elements have same order as y_train
        distance_vec = self._compute_distances(self.X, x)

        # find order of sorted elements from nearest to furthest neighbour
        sorted_indeces = np.argsort(distance_vec, kind='mergesort')
        labels_knn = np.zeros(self.k)

        if self.k <= len(sorted_indeces):
            # check the labels of the k-NN and find the mode
            for i in range(0, self.k):
                labels_knn[i] = self.y[sorted_indeces[i]]

            pred_label = self._count_nn_labels(labels_knn)
            if pred_label == 0:
                return self._count_nn_labels(labels_knn[0:-1])
            else:
                return pred_label
        else:
            print(f'Less than {self.k} neighbours available!')
            return 0
