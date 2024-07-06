import numpy as np
from Layers import Base, Helpers 
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self._optimizer = None
        self.moving_mean = None
        self.moving_variance = None
        self.decay = 0.8
    
    def initialize(self, weights_initializer = None, bias_initializer = None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    
    def forward(self, input_tensor):
        
        is_conv_layer = len(input_tensor.shape) == 4
        if not is_conv_layer:
            self.input_tensor = input_tensor
        else:
            self.input_tensor = self.reformat(input_tensor)
 
        if self.testing_phase:
            self.mean = self.moving_mean
            self.variance = self.moving_variance 
        else:
            self.mean, self.var = np.mean(self.input_tensor, axis=0, keepdims=True), np.var(self.input_tensor, axis=0, keepdims=True)
            if self.moving_mean is not None:
                self.moving_mean = (self.moving_mean * self.decay) + (self.mean * (1 - self.decay))
                self.moving_variance = (self.moving_variance * self.decay) + (self.var * (1 - self.decay))
            else:
                self.moving_mean, self.moving_variance = self.mean, self.var
                
        self.normalized_input  = (self.input_tensor - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
        output = self.gamma * self.normalized_input + self.beta
        
        if is_conv_layer:
            output = self.reformat(output)
            
        return output  

    def backward(self, error_tensor):
        is_conv_layer = len(error_tensor.shape) == 4
        if not is_conv_layer:
            self.error_tensor = error_tensor
        else:
            self.error_tensor = self.reformat(error_tensor)
            
        derivative_gamma = np.sum(self.error_tensor * self.normalized_input, axis=0)
        derivative_beta = np.sum(self.error_tensor, axis=0)
        
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, derivative_gamma)
            self._optimizer.bias.calculate_update(self.beta, derivative_beta)
            
        gradient_input = Helpers.compute_bn_gradients(self.error_tensor, self.input_tensor, self.gamma, self.mean, self.var)
        
        if is_conv_layer:
            gradient_input = self.reformat(gradient_input)
            
        self.gradient_weights = derivative_gamma
        self.gradient_bias = derivative_beta    
        
        return gradient_input
            
    def reformat(self, tensor):
        is_conv_layer = len(tensor.shape) == 4
        if not is_conv_layer:
            batch, height, width, channels = self.stored_shape
            reshaped = tensor.reshape(batch, width * channels, height)
            transposed = reshaped.transpose(0, 2, 1)
            reformatted = transposed.reshape(batch, height, width, channels)
            return reformatted
        else:
            self.stored_shape = tensor.shape
            batch, height, width, channels = tensor.shape
            reshaped = tensor.reshape(batch, height, width * channels)
            transposed = reshaped.transpose(0, 2, 1)
            reformatted = transposed.reshape(batch * width * channels, height)
            return reformatted
            
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
        
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    

    