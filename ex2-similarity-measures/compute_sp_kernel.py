"""
Homework  : Similarity measures on time series and graphs
Course    : Data Mining (636-0018-00L)

Compute similarity between nodes and graphs.
Floyd-Warshall and SP-Kernel.
"""
# Author: Theresa Wakonig

import argparse
import os
import sys
import numpy as np
import scipy.io
from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import sp_kernel


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute shortest path kernel."
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file MUTAG.mat"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where timeseries_output.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)


    # Load data from .mat file
    mat = scipy.io.loadmat("{}/{}".format(args.datadir, 'MUTAG.mat'))
    label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
    data = np.reshape(mat['MUTAG']['am'], (len(label), ))


    # Create the output file
    try:
        file_name = "{}/graphs_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    # create dictionary for classes
    cdict = {}
    cdict['mutagenic'] = 1
    cdict['non-mutagenic'] = -1
    lst_group = ['mutagenic', 'non-mutagenic']

    # Write header for output file
    f_out.write('{}\t\t{}\n'.format(
        'Pair of classes',
        'SP'))


    # Iterate through groups and write results to file
    for idx_g1 in range(len(lst_group)):
        for idx_g2 in range(idx_g1, len(lst_group)):
            # 1st iteration: mutagenic:mutagenic; 2nd: mutagenic:non-mutagenic; 3rd: non-mutagenic:non-mutagenic
            group1 = data[label[:] == cdict[lst_group[idx_g1]]]
            group2 = data[label[:] == cdict[lst_group[idx_g2]]]

            # get average similarities
            count = 0
            sim = 0

            # swap if group2 has more elements than group1 because of execution order of following for-loop
            if len(group2) > len(group1):
                temp = group1
                group1 = group2
                group2 = temp

            # sp_kernel is symmetric, therefore I do not check both: sp_kernel(Sx, Sy) & sp_kernel(Sy, Sx)
            # however: sp_kernel(Sx, Sx) is checked (NOT redundant)
            for i in range(len(group1)):
                for j in range(i, len(group2)):
                    sim += sp_kernel(floyd_warshall(group1[i]), floyd_warshall(group2[j]))

                    count += 1

            avg_sim = sim/count

            # Transform the distance to a string
            str_sim = '\t{0:.2f}'.format(avg_sim)

            # Save the output
            f_out.write(
                '{}:{}\t{}\n'.format(
                    lst_group[idx_g1], lst_group[idx_g2], str_sim)
            )
    f_out.close()