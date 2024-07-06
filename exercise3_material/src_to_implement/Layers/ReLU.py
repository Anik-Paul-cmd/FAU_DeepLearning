import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.next_layer_input = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        gradient_relu = error_tensor.copy()
        gradient_relu[self.next_layer_input <= 0] = 0
        return gradient_relu
