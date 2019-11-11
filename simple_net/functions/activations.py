""" Activation Functions """
import numpy as np


def sigmoid(x):
    """ Sigmoid function 
        courtesy of Michael Nielsen
        `http://neuralnetworksanddeeplearning.com/` """
    return 1.0/(1.0+np.exp(-x))

def ddx_sigmoid(x):
    """ Derivative of sigmoid function 
        courtesy of Michael Nielsen
        `http://neuralnetworksanddeeplearning.com/` """
    return sigmoid(x)*(1-sigmoid(x))