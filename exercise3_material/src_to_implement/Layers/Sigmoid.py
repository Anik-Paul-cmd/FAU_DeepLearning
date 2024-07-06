import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation  

    def backward(self, error_tensor):
        sigmoid_der = self.activation * (1 - self.activation)
        gradient_sig = sigmoid_der * error_tensor
        return gradient_sig    
        