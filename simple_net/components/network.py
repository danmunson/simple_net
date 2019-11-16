""" Define implementation of simple neural network """
import numpy as np
from copy import deepcopy

from simple_net.components.layer import Layer

class Network:

    def __init__(
        self, 
        layer_dims, 
        activations, 
        activation_derivatives,
        cost_function,
        cost_gradient,
        learning_rate,
        gradient_aggregator
    ):
        """ Instantiate layers of Neural Network """
        self._layers = [
            Layer(
                layer_dims[i], 
                layer_dims[i-1], 
                activations[i-1],
                activation_derivatives[i-1]
            )
            for i in range(1, len(layer_dims))
        ]
        self.cost_function = cost_function
        self.cost_gradient = cost_gradient
        self.gradient_aggregator = gradient_aggregator
        self._learning_rate = learning_rate
        self._epochs = list()

    def feed_forward(self, input_vector):
        """ Feeds input data through the network, returns output """
        for layer in self._layers:
            input_vector = layer.ff(input_vector)
        return input_vector

    def feed_batch(self, inputs):
        """ Feeds a batch of inputs through the network, collects results """
        return [self.feed_forward(input_vector) for input_vector in inputs]

    def compute_cost(self, predictions, actuals):
        """ Computes cost and gradient of cost function with respect
            to the output of the network """
        costs, cost_gradients = zip(*[
            (
                self.cost_function(prediction, actual),
                self.cost_gradient(prediction, actual)
            )
            for prediction, actual in
            zip(predictions, actuals)
        ])
        agg_cost_gradient = self.gradient_aggregator(cost_gradients)
        return agg_cost_gradient, costs, cost_gradients

    def back_prop(self, cost_gradient):
        """ Updates the paramters of the network after a batch """
        for i in range(1, len(self._layers)+1):
            layer = self._layers[-i]
            cost_gradient = layer.update_params(
                cost_gradient,
                self._learning_rate
            )
        return cost_gradient
    
    def run_epoch(self, features, actuals):
        """ Run a training epoch on a set of training data """
        inputs, resps = deepcopy(features), deepcopy(actuals)
        outputs = self.feed_batch(inputs)
        agg_gradient, costs, gradients = self.compute_cost(outputs, resps)
        gradient_wrt_input = self.back_prop(agg_gradient)
        self._epochs.append({
            'err_gradient' : agg_gradient,
            'input_gradient' : gradient_wrt_input,
            'average_cost' : np.mean(costs),
            'err_gradient_magnitude' : np.sum([x**2 for x in agg_gradient]),
            'err_gradient_magnitude' : np.sum([x**2 for x in gradient_wrt_input])
        })
        return outputs