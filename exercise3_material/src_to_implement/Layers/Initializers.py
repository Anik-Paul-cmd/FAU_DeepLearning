import numpy as np

class Constant():
    def __init__(self, constant = 0.1):
        self.constant = constant
        
        
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.zeros(weights_shape) + self.constant

class UniformRandom():
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size = weights_shape)

class Xavier():
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.randn(*weights_shape) * np.sqrt(2./(fan_out+fan_in))

class He():
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.randn(*weights_shape) * np.sqrt(2/fan_in)