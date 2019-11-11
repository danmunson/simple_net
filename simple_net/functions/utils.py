"""
Helper Functions
"""
import numpy as np

def vectorize(function):
    """ Return a function that computes argument function
        for each component in a vector """
    vec_func = lambda vector : np.array(
        list(map(function, vector))
    )
    return vec_func





