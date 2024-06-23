import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self.gradient_weights = None
        self.pre_input = None
        self.next_layer_input = None

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        bias_input = np.ones((batch_size, 1))
        input_with_bias = np.hstack((input_tensor, bias_input))
        
        self.pre_input = input_with_bias
        self.next_layer_input = np.dot(input_with_bias, self.weights)
        return self.next_layer_input

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
    def backward(self, error_tensor):
        gradient_dx = np.dot(error_tensor, self.weights[:-1].T)
        self.gradient_weights = np.dot(self.pre_input.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        
        return gradient_dx
