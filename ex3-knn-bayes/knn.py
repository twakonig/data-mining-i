"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)
Theresa Wakonig

Main program for k-NN.
Predicts the labels of the test data using the training data.
The k-NN algorithm is executed for different values of k (user-entered parameter)


Original author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import argparse
import os
import sys
import numpy as np
import math
from scipy import stats

# Import the file with the performance metrics 
import evaluation

# Class imports
from knn_classifier import KNNClassifier


# Constants
# 1. Files with the datapoints and class labels
DATA_FILE  = "matrix_mirna_input.txt"
PHENO_FILE = "phenotype.txt"

# 2. Classification performance metrics to compute
PERF_METRICS = ["accuracy", "precision", "recall"]


def load_data(dir_path):
    # np.ndarray (#patients x 489) for mi_rna and (#patients x 2) for pheno: all of type 'str', incl. header and ID
    mi_rna = np.loadtxt("{}/{}".format(dir_path, DATA_FILE), dtype=object)
    pheno = np.loadtxt("{}/{}".format(dir_path, PHENO_FILE), dtype=object)

    # (#patients x 1) and (#patients x 488) respectively
    y = np.zeros(np.shape(pheno)[0] - 1)
    X = np.zeros(shape=(np.shape(mi_rna)[0] - 1, np.shape(mi_rna)[1] - 1))

    for i in range (1, np.shape(mi_rna)[0]):
        # match patientID and enter class label in y vector
        row_in_pheno = np.argwhere(pheno == mi_rna[i][0])
        if pheno[row_in_pheno[0][0]][1] == '+':
            y[i - 1] = 1
        else:
            y[i - 1] = -1
        # put float entries of data points into X matrix: same rows in X and y belong to same patient
        for j in range(1, np.shape(mi_rna)[1]):
            X[i - 1, j - 1] = float(mi_rna[i, j])

    return y, X


def obtain_performance_metrics(y_true, y_pred):

    return np.array([evaluation.compute_accuracy(y_true, y_pred),
                    evaluation.compute_precision(y_true, y_pred), evaluation.compute_recall(y_true, y_pred)])


#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Classify data points with k-NN classifier"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to directory where training data are stored (miRNA and phenotype)"
    )
    parser.add_argument(
        "--testdir",
        required=True,
        help="Path to directory where test data are stored (miRNA and phenotype)"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_knn.txt will be created"
    )
    parser.add_argument(
        "--mink",
        required=True,
        help="Minimum value of k for k-NN algorithm"
    )
    parser.add_argument(
        "--maxk",
        required=True,
        help="Maximum value of k for k-NN algorithm"
    )

    args = parser.parse_args()

    # If the output directory does not exist, then create it
    os.makedirs(args.outdir, exist_ok=True)

    # Read the training and test data. For each dataset, get also the true labels.
    # y...true labels, X...data points
    y_train, X_train = load_data(args.traindir)
    y_test, X_test = load_data(args.testdir)

    # Create the output file
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    # Write header for output file
    f_out.write('{}\t\t{}\n'.format(
        'Value of k',
        '\t'.join(['{}'.format(metr) for metr in PERF_METRICS])))

    ############################## KNN algorithm ####################################

    # Create the k-NN object. (Hint about how to do it in the homework sheet)
    knn = KNNClassifier(X_train, y_train, metric = 'euclidean')
    y_true = y_test

    k_min = int(args.mink)
    k_max = int(args.maxk)

    for k in range(k_min, k_max + 1):
        knn.set_k(k)
        y_pred = np.zeros(len(X_test))
        # for one specific k (k-neighbours in train), predict the labels of all test data points and store in y_pred
        for i in range(0, len(X_test)):
            x = X_test[i][:]
            # predicted label for data vector x
            y_pred[i] = knn.predict(x)
        # compute performance metrics given the true-labels and the predicted-labels
        eval_performance_vec = obtain_performance_metrics(y_true, y_pred)
        # formatting, transform float to string
        str_acc = '\t{0:.2f}'.format(eval_performance_vec[0])
        str_prec = '\t{0:.2f}'.format(eval_performance_vec[1])
        str_rec = '\t{0:.2f}'.format(eval_performance_vec[2])

        # write performance results in the output file
        f_out.write(
            '{}\t\t\t{}\t{}\t{}\n'.format(
                k, str_acc, str_prec, str_rec)
        )

    f_out.close()