import numpy as np
import copy

class NeuralNetwork():
    
    def __init__(self, optimizer,weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)
    
    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        
        layer_index = 0
        while layer_index < len(self.layers):
            input_tensor = self.layers[layer_index].forward(input_tensor)
            layer_index += 1
        
        loss = self.loss_layer.forward(input_tensor, copy.deepcopy(self.label_tensor))
        return loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(copy.deepcopy(self.label_tensor))
        
        layer_index = len(self.layers) - 1
        while layer_index >= 0:
            error_tensor = self.layers[layer_index].backward(error_tensor)
            layer_index -= 1
        
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    def train(self, iterations):
        iteration = 0
        while iteration < iterations:
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
            iteration += 1

    def test(self, input_tensor):
        layer_index = 0
        while layer_index < len(self.layers):
            input_tensor = self.layers[layer_index].forward(input_tensor)
            layer_index += 1
        return input_tensor
        