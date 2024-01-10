"""
Homework  : Similarity measures on time series and graphs
Course    : Data Mining (636-0018-00L)

Compute similarity between nodes and graphs.
Floyd-Warshall and SP-Kernel.
"""
# Author: Theresa Wakonig

import numpy as np


def floyd_warshall(A):
    # shortest-path matrix S
    S = A.astype(float)
    # unconnected nodes are initialized with 'inf' distance
    no_link = np.argwhere(S < 1)
    for elem in no_link:
        if elem[0] != elem[1]: S[elem[0], elem[1]] = float('inf')

    # three nested for loops iterate over all nodes
    for k in range(len(S)):
        for i in range(len(S)):
            for j in range(len(S)):
                S[i, j] = min(S[i, j], S[i, k] + S[k, j])
    return S


def sp_kernel(S1, S2):
    # extract upper triangular part, ignore diagonal (zeros)
    S1_triu = np.triu(S1, 1)
    S2_triu = np.triu(S2, 1)
    # if infinity values exist, set to zero
    S1_triu[S1_triu == np.inf] = 0
    S2_triu[S2_triu == np.inf] = 0
    # determine longest edge walk from S1 and S2
    max_edgewalk = np.max([np.max(S1_triu), np.max(S2_triu)])
    # 1D arrays to fill in frequency of specific edge walk
    phi_G1 = np.zeros(shape=(int(max_edgewalk), 1))
    phi_G2 = np.zeros(shape=(int(max_edgewalk), 1))

    # insert occurrences of different edge walks
    for i in range(1, len(phi_G1) + 1):
        phi_G1[i - 1] = np.count_nonzero(S1_triu == i)
        phi_G2[i - 1] = np.count_nonzero(S2_triu == i)

    # calculate inner product (kernel)
    k = np.dot(np.transpose(phi_G1), phi_G2)
    return float(k)
