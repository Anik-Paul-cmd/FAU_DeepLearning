import numpy as np
class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        return weight_tensor 
        
class SgdWithMomentum():
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
        v = self.learning_rate * gradient_tensor + self.momentum_rate * self.v
        weight_tensor = weight_tensor - v
        self.v = v
        return weight_tensor

class Adam():
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
        if self.r is None or self.r.shape != weight_tensor.shape:
            self.r = np.zeros_like(weight_tensor)
        
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        self.k += 1
        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        return weight_tensor

