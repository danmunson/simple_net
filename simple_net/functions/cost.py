""" Cost & Gradient Functions """
import numpy as np

"""
Cost
"""

class MSE:

    @staticmethod
    def average_cost(predictions, actuals):
        """ Mean squared error - aka mean of the squared Euclidian distance """
        return sum([
            (y0 - y1) ** 2 
            for arr_pred, arr_actual in zip(predictions, actuals)
            for y0, y1 in zip(arr_actual, arr_pred)
        ]) / len(predictions)

    @staticmethod
    def gradient(prediction, actual):
        """ Gradient of MSE with respect to predictions 
            Order matters! Actual must be subtracted from prediction """
        return prediction - actual


class CrossEntropy:
    """
    Big thanks to Michael Nielsen
    http://neuralnetworksanddeeplearning.com/chap3.html
    """

    @staticmethod
    def average_cost(predictions, actuals):
        """ Cross Entropy cost 
            NORM-SUM over predictions, actuals (P, A)
                SUM over output components (c)
                    Ac•ln(Pc) + (1-Ac)•ln(1-Pc)
            Order matters!
        """
        return sum([
            (Ac * np.log(Pc)) + ((1-Ac) * np.log(1-Pc))
            for P, A in zip(predictions, actuals)
            for Pc, Ac, in zip(P, A)
        ]) / len(predictions)
    
    @staticmethod
    def gradient(prediction, actual):
        """ Gradient for cross entropy cost
            Note - can be optimized in conjunction with the
            final output layer, assuming Sigmoid or Softmax """
        return np.array([
            (a / p) - ((1 - a) / (1 - p))
            for p, a in zip(prediction, actual)
        ])



"""
Gradient Aggregation
"""
def arithmetic_mean(gradients):
    """ Return the arithmetic mean at each position
        for a given list of gradients """
    return np.array([*gradients]).mean(axis=0)