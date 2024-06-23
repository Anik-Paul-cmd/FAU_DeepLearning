import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        shifted_log = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_log)
        prob= exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_tensor = prob
        return prob
    
    def backward(self, error_tensor):
        elementwise = error_tensor * self.output_tensor
        sum_elementwise= np.sum(elementwise, axis=1, keepdims=True)
        adjusted_error_tensor = error_tensor - sum_elementwise
        softmax_grad = self.output_tensor * adjusted_error_tensor
        
        return softmax_grad
        