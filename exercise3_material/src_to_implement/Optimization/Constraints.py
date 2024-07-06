import numpy as np

class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha
        
    def norm(self, weights):
        l2_norm = self.alpha * np.sum(np.square(weights))
        return l2_norm    
         
    def calculate_gradient(self, weights):
        l2_gradient = self.alpha * weights
        return l2_gradient
    
class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weights):
        l1_norm = self.alpha * np.sum(np.abs(weights))
        return l1_norm
    
    def calculate_gradient(self, weights):
        l1_gradient = np.sign(weights) * self.alpha
        return l1_gradient
        
 