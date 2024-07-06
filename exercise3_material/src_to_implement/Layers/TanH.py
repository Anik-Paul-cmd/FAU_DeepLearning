import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation
    
    def backward(self, error_tensor):
        tanh_der = 1 - np.square(self.activation)
        gradient_tanH = tanh_der * error_tensor
        return gradient_tanH