'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('\n')


# function performing logistic regression
def logistic_regr(X_train, Y_train, X_test):
    scaler = StandardScaler()
    logisticRegr = LogisticRegression()

    # fit the scaler using the training data set
    scaler.fit(X_train)
    standardized_X_train = scaler.transform(X_train)
    standardized_X_test = scaler.transform(X_test)

    # training the model on train data
    logisticRegr.fit(standardized_X_train, Y_train)

    # printing estimated coefficients
    print('Coefficients learned from training set:')
    weights = logisticRegr.coef_
    for i in range(weights.shape[1]):
        print(f'Coeff. {i+1}: {weights[0][i]}')
    print('\n')

    # return predicted labels for test data
    return logisticRegr.predict(standardized_X_test)


if __name__ == "__main__":

    test_file = 'data/diabetes_test.csv'
    train_file = 'data/diabetes_train.csv'
    df_test = pd.read_csv(test_file)
    df = pd.read_csv(train_file)

    # extract first 7 columns to data matrix X (332 x 7)
    X_test = df_test.iloc[:, 0:7].values
    # extract 8th column (labels) to vector Y (332 x 1)
    Y_test = df_test.iloc[:, 7].values

    # extract first 7 columns to data matrix X (200 x 7)
    X_train = df.iloc[:, 0:7].values
    # extract 8th column (labels) to vector Y (200 x 1)
    Y_train = df.iloc[:, 7].values

# -----------------------------------------------------EXECISE 1a)------------------------------------------------------

    print('Exercise 1.a')
    print('------------')

    # make predictions for test data via logistic regression
    predictions = logistic_regr(X_train, Y_train, X_test)
    print('# Logistic Regression performance on diabetes dataset.\n')
    compute_metrics(Y_test, predictions)


    print('Exercise 1.b')
    print('------------')

    print('See attached PDF.')
    print('\n')

    print('Exercise 1.c')
    print('------------')

    print('See attached PDF.')
    print('\n')

#-----------------------------------------------------EXECISE 1d)-------------------------------------------------------

    # attribute skin: column index 3
    X_test_posthoc = df_test.iloc[:, [True, True, True, False, True, True, True, False]].values
    X_train_posthoc = df.iloc[:, [True, True, True, False, True, True, True, False]].values

    print('Exercise 1.d')
    print('------------')

    print('See attached PDF.')
    print('\n')

    # make predictions for test data via logistic regression
    predictions_posthoc = logistic_regr(X_train_posthoc, Y_train, X_test_posthoc)
    compute_metrics(Y_test, predictions_posthoc)




