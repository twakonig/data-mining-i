#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import math
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import KFold


def split_data(X, y, attribute_index, theta):
    # split X and y into two subsets accarding to attr_idx and theta
    X1 = X[X[:, attribute_index] < theta]
    X2 = X[X[:, attribute_index] >= theta]

    y1 = y[np.argwhere(X[:, attribute_index] < theta)]
    y2 = y[np.argwhere(X[:, attribute_index] >= theta)]
    return X1, y1, X2, y2


def compute_information_content(y):
    # info of dataset before splitting (probability = frequ. of class in dataset)
    info_content = 0
    for i in range(3):
        instances_i = np.count_nonzero(y == i)
        frequ_i = instances_i/len(y)
        if frequ_i > 0:
            info_content += frequ_i * math.log2(frequ_i)

    return -info_content


def compute_information_a(X, y, attribute_index, theta):
    # information content after being split according to attribute A
    X1, y1, X2, y2 = split_data(X, y, attribute_index, theta)

    info_1 = compute_information_content(y1)
    info_2 = compute_information_content(y2)
    return (len(y1)*info_1 + len(y2)*info_2)/len(y)


def compute_information_gain(X, y, attribute_index, theta):
    # information gain of the split according to attribute & theta
    return compute_information_content(y) - compute_information_a(X, y, attribute_index, theta)


def classification_kfold(X, y):
    # classification with DecisionTreeClassifier and cross-validation
    accuracy_sum = 0
    cross_val = KFold(n_splits=5, shuffle=True)
    # Generates indices to split data into training and test sets (30 elements each)
    for train_idx, test_idx in cross_val.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        # create a classifier and fit to training data
        tree_clf = tree.DecisionTreeClassifier()
        tree_clf.fit(X_train, y_train)
        y_pred = tree_clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        accuracy_sum += score
        print(tree_clf.feature_importances_)
    print('')
    return accuracy_sum/cross_val.get_n_splits() * 100




if __name__ == '__main__':

    # load the data into X (150x4) and labels into y (150x1)
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names
    num_features = len(set(feature_names))


    print('Exercise 2.b')
    print('------------')
    vals_theta = np.array([5.0, 3.0, 2.5, 1.5])
    for i in range(4):
        print(f'Split ({feature_names[i]} < {vals_theta[i]}): '
              f'information gain = {round(compute_information_gain(X, y, i, vals_theta[i]), 2)}')

    print('')

    print('Exercise 2.c')
    print('------------')
    print('See attached PDF.')

    print('')

    print('Exercise 2.d')
    print('------------')

    # make results identical by setting the random_state (for shuffling)
    np.random.seed(42)

    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')
    accuracy_mean = classification_kfold(X, y)

    print('The mean accuracy is {0:.2f}'.format(accuracy_mean))
    print('')

    X_reduced = X[y != 2]
    y_reduced = y[y != 2]

    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')
    acc_reduced = classification_kfold(X_reduced, y_reduced)
    print('The mean accuracy is {0:.2f}'.format(acc_reduced))
