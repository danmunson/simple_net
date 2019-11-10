"""Define implementation for a single neural layer that consumes a preceding layer"""

import numpy as np

class Layer:

    def __init__(self, self_size, consume_size, activation, activation_derivative):
    """ Create random M x N matrix, where N is the dimension
        of the preceding layer, and M is the dimension of this
        layer. I.e. each row in _weights represents the weights
        for a single node in the layer """
        self._weights = np.random.rand(
            self_size,
            consume_size
        )
        self._bias = np.random.rand(self_size)
        self._activtion = activation
        self._activation_derivative = activation_derivative
    
    """
    Feed forward utilities
    """

    def feed(self, input_vector):
    """ Compute the new value of the layer """
        self._input = input_vector
        self._product = np.dot(self._weights, input_vector) + self._bias
        self._output = self._activation(self._product)
    
    def forward(self):
    """ Return the value of the layer """
        return self._output
    
    def ff(self, input_vector):
    """ Concise chaining function """
        self.feed(input_vector)
        return self.forward()

    """
    Gradient descent utilities
    """

    def adjust(self, weight_adjustment, bias_adjustment):
    """ Adjust the paramters of the layer """
        self._weights += weight_adjustment
        self._bias += bias_adjustment

    def compute_gradient(self, current_cost_gradient):
    """ Calculates the gradient with respect to weights and bias """
        
    """ `activation_gradient` is the pointwise product of the gradient
        of the activation function with respect to its input (self._product)
        and the passed-back gradient, which is the gradient of the cost function
        with respect to the output of this layer

        This is the first step of applying the chain rule component-wise,
        i.e. for each node in the layer

        Note that this vector also serves as the bias adjustment """

        activation_gradient = np.multiply(
            self._activation_derivative(self._product),
            current_cost_gradient
        )
        assert activation_gradient.shape == self._bias.shape, (
            f"Activation gradient is size {activation_gradient.shape} "
            f"but layer size is {self._bias.shape}"
        )

    """ `weight_adjs` is the outer product of the activation gradient
        and the input vector, which serves as the weight adjustment
        
        This follows from the fact that the partial derivative of `Wx`
        with respect to a given weight W[i,j], where `W` is the 
        weight matrix and `x` is the input vector (from prev. layer),
        is equal to:
            
            activation_gradient[i] * input_vector[j]
        
        Thus, the outer product of these two vectors yields the exact
        update matrix for `W` """
        
        weight_adjs = np.outer(activation_gradient, self._input)
        assert weight_adjs.shape == self._weights.shape, (
            f"Weight matrix is size {weight_adjs.shape} "
            f"but weight adjustment matrix is size {self._weights.shape}"
        )

    """ `cost_gradient_wrt_input` represents the gradient of the cost 
        function with respect to the input to this layer, and is calculated
        as the matrix product of the activation gradient and and the weight
        matrix `W`
        
        This follows from the fact that the the partial derivative of the 
        output `activation(Wx + b)` with respect to a given component 
        of the input vector `x[i]` is equal to:
        
            np.dot(activation_gradient, W[:,i])
        
        i.e. the dot product of the activation_gradient and the column i of `W`
        
        Thus, the operation can be condensed into a matrix multiplication with 
        the righthand operand being the weight matrix `W` """

        cost_gradient_wrt_input = np.dot(
            activation_gradient,
            self._weights
        )
        assert cost_gradient_wrt_input.shape == self._input.shape, (
            f"New cost gradient is size {weight_adjs.shape} "
            f"but input vector is size {self._weights.shape}"
        )

        return (
            cost_gradient_wrt_input,
            weight_adjs,
            activation_gradient
        )
    
    def update_params(self, current_cost_gradient):
    """ Updates the function paramters and returns the updated 
        current_cost_gradient """
        computations = self.compute_gradient(current_cost_gradient)
        cost_gradient_wrt_input, weight_adj, bias_adj = computations
        self.adjust(weight_adjs, bias_adj)
        return cost_gradient_wrt_input