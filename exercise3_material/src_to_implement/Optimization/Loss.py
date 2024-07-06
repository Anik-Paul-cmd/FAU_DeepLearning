import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def forward(self, prediction_tensor, label_tensor):
        self.y_hat = prediction_tensor
        log_predictions = np.log(prediction_tensor + np.finfo(float).eps)
        loss = - np.sum(label_tensor * log_predictions)
        return loss
    
    def backward(self, label_tensor):
        output = -(label_tensor / (self.y_hat + np.finfo(float).eps))
        return output
    
        
        
    