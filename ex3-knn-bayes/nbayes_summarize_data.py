"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for Naive Bayes.


Theresa Wakonig
"""

import argparse
import os
import sys
import numpy as np

FEATURES = ["clump", "uniformity", "marginal", "mitoses"]

# computes likelihood P(x|y_i)
def compute_likelihood(data_class_i, val):

    likelihood_vec = np.zeros(4)
    # iterate over all features (columns)
    for feature in range(0, 4):
        count_vals = np.count_nonzero(data_class_i[:, feature] == val)
        count_nz = np.count_nonzero(data_class_i[:, feature])
        likelihood_vec[feature] = count_vals / count_nz

    return likelihood_vec


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Classify data points with k-NN classifier"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to directory where training data are stored"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_summary_class_<label>.txt will be created"
    )

    args = parser.parse_args()

    # If the output directory does not exist, then create it
    os.makedirs(args.outdir, exist_ok=True)

    # put data from tumor_info.txt into array, account for missing values
    tumor_train = np.genfromtxt("{}/{}".format(args.traindir, 'tumor_info.txt'), delimiter='\t', dtype=float, filling_values=0)


    # Create the output files for classes 2 and 4
    try:
        file_name = "{}/output_summary_class_2.txt".format(args.outdir)
        f_out2 = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

        # Write header for output file
    f_out2.write('{}\t{}\n'.format(
        'Value',
        '\t'.join(['{}'.format(feat) for feat in FEATURES])))

    try:
        file_name = "{}/output_summary_class_4.txt".format(args.outdir)
        f_out4 = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

        # Write header for output file
    f_out4.write('{}\t{}\n'.format(
        'Value',
        '\t'.join(['{}'.format(feat) for feat in FEATURES])))


    # -----------------------------MAIN PROGRAM-----------------------------------------------
    # create array of class 2 and class 4 members -> 0 ENTRIES for missing values!
    data_c2 = np.vstack(tumor_train[np.argwhere(tumor_train[:, 4] == 2), 0:4])
    data_c4 = np.vstack(tumor_train[np.argwhere(tumor_train[:, 4] == 4), 0:4])

    # for data point classification in task
    mat_c2 = np.zeros(shape=(10, 4))
    mat_c4 = np.zeros(shape=(10, 4))

    # iterate over possible feature values
    for val in range(1, 11):
        summary_c2 = compute_likelihood(data_c2, val)
        summary_c4 = compute_likelihood(data_c4, val)
        mat_c2[val - 1, :] = summary_c2.round(decimals=3)
        mat_c4[val - 1, :] = summary_c4.round(decimals=3)

        # transform the vector of likelihoods to a string
        str_c2= '\t\t'.join('{0:.3f}'.format(p2) for p2 in summary_c2)
        str_c4 = '\t\t'.join('{0:.3f}'.format(p4) for p4 in summary_c4)

        f_out2.write(
            '{}\t\t{}\n'.format(
                val, str_c2)
        )
        f_out4.write(
            '{}\t\t{}\n'.format(
                val, str_c4)
        )

    f_out2.close()
    f_out4.close()

    # ------------------------------------------------------------------------------------------------------------
    # classification of given data point
    # TO-DO: CHOOSE HOW PRIOR SHOULD BE COMPUTED! possible values: 'random', 'label_frequ'
    prior = 'random'

    if prior == 'random':
        p_y2 = p_y4 = 0.5
    elif prior == 'label_frequ':
        p_y2 = len(data_c2)/len(tumor_train)
        p_y4 = len(data_c4)/len(tumor_train)

    # given data point (will be hard coded): clump=6, uniformity=2, marginal=2, mitoses=1
    prob_c2 =p_y2 * mat_c2[5][0] * mat_c2[1][1] * mat_c2[1][2] * mat_c2[0][3]
    prob_c4 = p_y4 * mat_c4[5][0] * mat_c4[1][1] * mat_c4[1][2] * mat_c4[0][3]

    print(f'probablility for label 2, prior calculated with "{prior}": {prob_c2}')
    print(f'probablility for label 4, prior calculated with "{prior}": {prob_c4}')

    predicted_label = max(prob_c2, prob_c4)
    print(predicted_label)






