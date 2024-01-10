"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)
Theresa Wakonig

Auxiliary functions.

This file implements the metrics that are invoked from the main program.

Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import numpy as np

# Labels:
# POS = 1
# NEG = -1

def confusion_matrix(y_true, y_pred):
    '''
    Function for calculating TP, FP, TN, and FN.
    The input includes the vector of true labels
    and the vector of predicted labels
    '''

    pred_pos = np.sum(y_pred == 1)
    pred_neg = len(y_pred) - pred_pos
    true_pos = 0
    true_neg = 0

    for i in range(0, len(y_true)):
        if y_true[i] == 1 and y_true[i] == y_pred[i]:
            true_pos += 1

    for i in range(0, len(y_true)):
        if y_true[i] == -1 and y_true[i] == y_pred[i]:
            true_neg += 1

    false_pos = pred_pos - true_pos
    false_neg = pred_neg - true_neg

    return np.array([[true_pos, false_pos], [false_neg, true_neg]])


def compute_precision(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return contingency[0][0]/(contingency[0][0] + contingency[0][1])



def compute_recall(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return contingency[0][0] / (contingency[0][0] + contingency[1][0])



def compute_accuracy(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    correct_pred = np.trace(contingency)
    all_pred = np.sum(contingency)

    return correct_pred/all_pred

