import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        
    def forward(self, input_tensor):
        if not self.testing_phase:
            random_matrix = np.random.rand(*input_tensor.shape)
            self.dropout_mask = ((random_matrix < self.probability).astype(np.float32) / self.probability)        
        else:
            self.dropout_mask = np.ones_like(input_tensor)    
            
        return input_tensor * self.dropout_mask    
    
    def backward(self, error_tensor):
        gardient_dropout = error_tensor * self.dropout_mask
        return gardient_dropout
        
            