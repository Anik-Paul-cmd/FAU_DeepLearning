import numpy as np

class Optimizer():
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate:float):
        super().__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        w_array = np.asarray(weight_tensor).copy()
    
        weight_tensor -= self.learning_rate * gradient_tensor
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(w_array) 
        return weight_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum_vector = 0.
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        w_array = np.asarray(weight_tensor).copy()
        momentum_vector = self.learning_rate * gradient_tensor + self.momentum_rate * self.momentum_vector
        weight_tensor -= momentum_vector
        self.momentum_vector = momentum_vector
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(w_array)
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.moving_avg = 0.
        self.moving_avg_sq = 0.
        self.time_step = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_array = np.asarray(weight_tensor).copy()
        self.moving_avg = self.mu * self.moving_avg + (1 - self.mu) * gradient_tensor
        self.moving_avg_sq = self.rho * self.moving_avg_sq + (1 - self.rho) * np.power(gradient_tensor, 2)
        moving_avg_corrected = self.moving_avg / (1 - np.power(self.mu, self.time_step))
        moving_avg_sq_corrected = self.moving_avg_sq / (1 - np.power(self.rho, self.time_step))
        weight_tensor -= self.learning_rate * (moving_avg_corrected / (np.sqrt(moving_avg_sq_corrected) + np.finfo(float).eps))
        self.time_step += 1
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_array)
        return weight_tensor        
