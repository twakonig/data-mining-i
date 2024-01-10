"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy as np
import math


def manhattan_dist(v1, v2):
    distance = 0.0
    # Check dimensionality of vectors, otherwise do not compute the distance
    if len(v1) == len(v2):
        for i in range(len(v1)):
            distance += abs(v1[i] - v2[i])
    return distance


def hamming_dist(v1, v2):
    distance = 0.0
    # Check dimensionality of vectors, otherwise do not compute the distance
    if len(v1) == len(v2):
        for i in range(len(v1)):
            # Account for binarization (indirectly to avoid resetting values of single elements)
            if (v1[i] > 0) and (v2[i] > 0):
                continue
            elif (v1[i] > 0) or (v2[i] > 0):
                distance += 1
            else:
                continue
    return distance


def euclidean_dist(v1, v2):
    distance = 0.0
    # Check dimensionality of vectors, otherwise do not compute the distance
    if len(v1) == len(v2):
        for i in range(len(v1)):
            distance += (v1[i] - v2[i])**2
    # Take square root of sum of squared differences
    return math.sqrt(distance)


def chebyshev_dist(v1, v2):
    distance = 0.0
    # Check dimensionality of vectors, otherwise do not compute the distance
    if len(v1) == len(v2):
        for i in range(len(v1)):
            temp = abs(v1[i] - v2[i])
            # Find maximum difference across all dimensions
            if temp > distance:
                distance = temp
    return distance


def minkowski_dist(v1, v2, d):
    distance = 0.0
    # Check dimensionality of vectors and exponent d, otherwise do not compute the distance
    if (len(v1) == len(v2)) and (d > 0):
        for i in range(len(v1)):
            distance += abs(v1[i] - v2[i])**d
    # Take dth root of the sum
    return distance**(1/d)
