""" Cost & Gradient Functions """
import numpy as np

"""
Cost
"""

def mean_squared_error(arr_actual, arr_pred):
    """ Mean squared error """
    return sum(
        [(y0 - y1) ** 2 for y0, y1 in zip(arr_actual, arr_pred)]
    )

"""
Gradient
"""

def mse_gradient(arr_pred, arr_actual):
    """ Gradient of MSE with respect to predictions """
    return arr_pred - arr_actual

"""
Gradient Aggregation
"""

def arithmetic_mean(gradients):
    """ Return the arithmetic mean at each position
        for a given list of gradients """
    return np.array([*gradients]).mean(axis=0)