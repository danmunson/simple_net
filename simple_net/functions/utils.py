"""
Helper Functions
"""
import numpy as np
from random import choice


def vectorize(function):
    """ Return a function that computes argument function
        for each component in a vector """
    vec_func = lambda vector : np.array(
        list(map(function, vector))
    )
    return vec_func


def random_subset(features, labels, size):
    """ Function for selecting a random subset of training data """
    assert(len(features) == len(labels)), "Features and labels must be same length"
    subset_ftrs, subset_labels = list(), list()
    indices = choice(list(range(len(labels))), size)
    for i in indices:
        subset_ftrs.append(features[i])
        labels.append(labels[i])
    return (subset_ftrs, subset_labels)

