"""Define implementation for a single neural layer that consumes a preceding layer"""

import numpy as np

class Layer:

    def __init__(self, self_size, consume_size, output_map):
    """ Create random M x N matrix, where M is the dimension
        of the preceding layer, and N is the dimension of this
        layer """
        self._weights = np.random.rand(
            consume_size,
            self_size
        )
        self._bias = np.random.rand(self_size)
        self._output_map = output_map
    
    def feed(self, input_vector):
    """ Compute the new value of the layer """
        product = np.dot(input_vector, self._weights) + self._bias
        self._output = self._output_map(product)
    
    def forward(self):
    """ Return the value of the layer """
        return self._output
    
    def ff(self, input_vector):
    """ Concise chaining function """
        self.feed(input_vector)
        return self.forward()

    def adjust(self, weight_adjustment, bias_adjustment):
    """ Adjust the paramters of the layer """
        self._weights += weight_adjustment
        self._bias += bias_adjustment